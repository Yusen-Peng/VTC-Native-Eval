import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Any, Optional, Union
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from transformers.utils import logging
logger = logging.get_logger(__name__)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../../"))
sys.path.insert(0, PROJECT_ROOT)
from VLMEvalKit.vlmeval.vlm.neo.local_models.modeling_neo_chat import NEOChatModel

# VTC methods with learned compressor weights: trained alongside the LoRA, saved as <vtc_method>.bin
TRAINABLE_VTC_METHODS = ("drip", "perceiver")


class LoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._boundary_loss_sum = 0.0
        self._boundary_loss_count = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        # Log DRIP boundary regularization.
        if self.model.training and getattr(self.model, "vtc_method", None) == "drip":
            boundary_loss = self.model.vision_model.embeddings.last_boundary_loss

            if boundary_loss is not None:
                self._boundary_loss_sum += boundary_loss.detach().float().item()
                self._boundary_loss_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs and self._boundary_loss_count > 0:
            logs["boundary_loss"] = self._boundary_loss_sum / self._boundary_loss_count
            self._boundary_loss_sum = 0.0
            self._boundary_loss_count = 0
        super().log(logs, start_time=start_time)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        compressor_lr = self.args.compressor_learning_rate
        if compressor_lr is None or getattr(self, "_compressor_group_added", False):
            return optimizer

        # move the compressor's parameters into their own groups (same settings, own lr);
        # the scheduler is created afterwards, so it scales every group's lr alike
        compressor_params = {id(p) for p in self.model.compressor.parameters()}
        for group in list(optimizer.param_groups):
            moved = [p for p in group["params"] if id(p) in compressor_params]
            if moved:
                group["params"] = [p for p in group["params"] if id(p) not in compressor_params]
                optimizer.add_param_group(
                    {**{k: v for k, v in group.items() if k != "params"}, "params": moved, "lr": compressor_lr}
                )
        self._compressor_group_added = True
        return optimizer

    def _save_compressor(self, output_dir):
        model = self.accelerator.unwrap_model(self.model)

        if model.vtc_method not in TRAINABLE_VTC_METHODS:
            return

        if self.args.should_save:
            torch.save(
                {
                    name: param.detach().cpu().clone()
                    for name, param in model.compressor.named_parameters()
                },
                os.path.join(output_dir, f"{model.vtc_method}.bin"),
            )

    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save ONLY the LoRA adapter
        if hasattr(self.model, "language_model"):
            peft_model = self.model.language_model
        else:
            peft_model = self.model

        peft_model.save_pretrained(
            output_dir,
            safe_serialization=self.args.save_safetensors,
        )

        # Save tokenizer / processing class
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Keep training args for reproducibility
        torch.save(
            self.args,
            os.path.join(output_dir, "training_args.bin"),
        )

        self._save_compressor(output_dir)

    def _save_checkpoint(self, model, trial):
        # Let Trainer determine checkpoint-X naming
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        os.makedirs(output_dir, exist_ok=True)

        # Save adapter + tokenizer
        self._save(output_dir)

        # Save optimizer / scheduler so resume_from_checkpoint still works
        if self.args.should_save:
            self._save_optimizer_and_scheduler(output_dir)

        # Save scaler if applicable
        self._save_scaler(output_dir)

        # Save RNG state
        self._save_rng_state(output_dir)

        # Save Trainer state
        if self.args.should_save:
            self.state.save_to_json(
                os.path.join(output_dir, "trainer_state.json")
            )

        # Respect save_total_limit
        if self.args.should_save:
            self._rotate_checkpoints(
                use_mtime=False,
                output_dir=run_dir,
            )


    def _load_from_checkpoint(self, resume_from_checkpoint, model: NEOChatModel | None = None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target_model: NEOChatModel = model if model is not None else self.model
        target_model = self.accelerator.unwrap_model(target_model)
        vtc_method = getattr(target_model, "vtc_method", None)
        if vtc_method in TRAINABLE_VTC_METHODS:
            weights_path = os.path.join(resume_from_checkpoint, f"{vtc_method}.bin")
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"Missing {vtc_method} checkpoint: {weights_path}")
            # load_drip_weights / load_perceiver_weights
            getattr(target_model.compressor, f"load_{vtc_method}_weights")(weights_path)


class DistillationLoRATrainer(LoRATrainer):
    def __init__(
        self,
        *args,
        teacher_model=None,
        kd_lambda=1.0,
        kd_temperature=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if teacher_model is None:
            raise ValueError("teacher_model must be provided.")

        self.teacher_model = teacher_model
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature

        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        # Accumulate component losses between HF logging events.
        self._sft_loss_sum =0.0
        self._kd_loss_sum = 0.0
        self._loss_count = 0



    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
            HF Trainer.compute_loss + knowledge distillation.
            Total loss: L = L_SFT + kd_lambda * L_KD
        """
        if (
            self.label_smoother is not None
            or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.model_accepts_loss_kwargs:
            kwargs = {}

            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch

            inputs = {**inputs, **kwargs}

        # Student forward
        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            if labels is None:
                logger.warning(
                    "Trainer: `compute_loss_func` is defined but "
                    "`labels=None`. Your custom loss function will "
                    "still be called with labels=None."
                )

            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
            )

        # Default HF loss handling (label smoothing)
        elif labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)

            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )

            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(
                    outputs,
                    labels,
                    shift_labels=True,
                )
            else:
                loss = self.label_smoother(
                    outputs,
                    labels,
                )

        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, "
                    "only the following keys: "
                    f"{','.join(outputs.keys())}. "
                    "For reference, the inputs it received are "
                    f"{','.join(inputs.keys())}."
                )

            loss = (
                outputs["loss"]
                if isinstance(outputs, dict)
                else outputs[0]
            )

        sft_loss = loss

        """Knowledge distillation."""
        # # We need labels for masking even when HF popped them above.
        # kd_labels = labels if labels is not None else inputs.get("labels")
        # with torch.no_grad():
        #     teacher_outputs = self.teacher_model(**inputs)
        # student_logits = outputs["logits"]
        # teacher_logits = teacher_outputs["logits"]

        # T = self.kd_temperature
        # student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        # teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        # # KL per token: [B, L]
        # kd_loss_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

        # # Only distill supervised response tokens.
        # if kd_labels is not None:
        #     kd_mask = kd_labels.ne(-100)            
        #     kd_loss = (kd_loss_per_token * kd_mask).sum() / kd_mask.sum().clamp_min(1)
        # else:
        #     kd_loss = kd_loss_per_token.mean()

        # # Standard temperature correction
        # kd_loss = kd_loss * (T ** 2)

        kd_labels = labels if labels is not None else inputs.get("labels")
        if kd_labels.dim() == 1:
            kd_labels = kd_labels.unsqueeze(0)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        student_logits = outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        T = self.kd_temperature

        student_kd_logits = []
        teacher_kd_logits = []
        batch_size = kd_labels.size(0)
        for b in range(batch_size):
            # Number of supervised tokens in this sample.
            num_supervised = kd_labels[b].ne(-100).sum().item()
            if num_supervised == 0:
                continue

            # Causal LM alignment - logits[t] predicts token[t + 1]
            student_kd_logits.append(student_logits[b, -(num_supervised + 1):-1, :])
            teacher_kd_logits.append(teacher_logits[b, -(num_supervised + 1):-1, :])

        if len(student_kd_logits) == 0:
            # Keep graph connection to student for backward().
            kd_loss = student_logits.sum() * 0.0
        else:
            student_kd_logits = torch.cat(student_kd_logits, dim=0)
            teacher_kd_logits = torch.cat(teacher_kd_logits, dim=0)

            student_log_probs = F.log_softmax(student_kd_logits / T, dim=-1)
            teacher_probs = F.softmax(teacher_kd_logits / T, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            # Standard temperature correction
            kd_loss = kd_loss * (T ** 2)

        # Accumulate detached values so we don't keep computation graphs alive.
        self._sft_loss_sum += sft_loss.detach().float().item()
        self._kd_loss_sum += kd_loss.detach().float().item()
        self._loss_count += 1


        # Add KD regularization to the original HF/SFT loss
        loss = sft_loss + self.kd_lambda * kd_loss


        if (
            self.args.average_tokens_across_devices
            and (
                self.model_accepts_loss_kwargs
                or self.compute_loss_func
            )
            and num_items_in_batch is not None
        ):
            loss *= (
                self.accelerator.num_processes
                if self.args.n_gpu <= 1
                else self.args.n_gpu
            )

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        # HF puts "loss" in logs for the normal training-loss logging event.
        # Only consume/reset our accumulators at that point.
        if "loss" in logs and self._loss_count > 0:
            avg_sft_loss = self._sft_loss_sum / self._loss_count
            avg_kd_loss = self._kd_loss_sum / self._loss_count

            logs["sft_loss"] = avg_sft_loss
            logs["kd_loss"] = avg_kd_loss
            logs["kd_weighted_loss"] = self.kd_lambda * avg_kd_loss

            # NOTE: Reset for the next logging interval.
            self._sft_loss_sum = 0.0
            self._kd_loss_sum = 0.0
            self._loss_count = 0
        super().log(logs, start_time=start_time)