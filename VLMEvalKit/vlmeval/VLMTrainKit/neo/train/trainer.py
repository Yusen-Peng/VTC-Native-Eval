import os
import torch
from transformers import Trainer


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