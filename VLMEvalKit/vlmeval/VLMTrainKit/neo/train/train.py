import os
import pathlib
import sys
import torch
from transformers import HfArgumentParser, set_seed
from transformers.utils import logging
from peft import LoraConfig, get_peft_model, TaskType


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../../"))
sys.path.insert(0, PROJECT_ROOT)
from VLMEvalKit.vlmeval.VLMTrainKit.neo.train.trainer import LoRATrainer as Trainer
from VLMEvalKit.vlmeval.VLMTrainKit.neo.data.data_processor import make_supervised_data_module
from VLMEvalKit.vlmeval.VLMTrainKit.neo.train.build import build_model_and_tokenizer
from VLMEvalKit.vlmeval.VLMTrainKit.neo.train.argument import DataArguments, ModelArguments, TrainingArguments


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    if model_args.train_buffer:
        logging.info(
            f"Only train buffer with extra {model_args.extra_num_layers} layers"
        )
        for name, param in model.named_parameters():
            parts = name.split(".")
            if (
                "_h" in name
                or "_w" in name
                or "_hw" in name
                or "vision_model" in name
                or (
                    len(parts) > 2
                    and parts[2].isdigit()
                    and int(parts[2]) < model_args.extra_num_layers
                )
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    os.makedirs(training_args.output_dir, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(model_args, data_args)
    model.config.use_cache = False

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # LoRA on the Qwen3 language model
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "q_proj_hw",
            "k_proj_hw",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model.language_model = get_peft_model(model.language_model, lora_config)

    model.language_model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # train the model
    model.train()
    torch.set_grad_enabled(True)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    tokenizer.save_pretrained(training_args.output_dir)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
