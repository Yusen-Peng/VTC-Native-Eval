import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Union
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from transformers.utils import logging
logger = logging.get_logger(__name__)


class LoRATrainer(Trainer):
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
        # We need labels for masking even when HF popped them above.
        kd_labels = labels if labels is not None else inputs.get("labels")
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        student_logits = outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        T = self.kd_temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        # KL per token: [B, L]
        kd_loss_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

        # Only distill supervised response tokens.
        if kd_labels is not None:
            # kd_mask = kd_labels.ne(-100)
            kd_mask = kd_labels[..., 1:].ne(-100)
            kd_loss = (kd_loss_per_token * kd_mask).sum() / kd_mask.sum().clamp_min(1)
        else:
            kd_loss = kd_loss_per_token.mean()

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