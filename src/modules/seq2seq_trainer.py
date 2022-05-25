"""
A subclass of `Trainer` specific to GPT2 training
In the evaluation part, model.generate() requires more arguments
"""
import os 
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
import collections

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from transformers import Seq2SeqTrainer
from transformers.trainer_utils import (
    speed_metrics,
    PREFIX_CHECKPOINT_DIR
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Seq2SeqTrainerWithMetrics(Seq2SeqTrainer):
    def __init__(self, *args, pad_token_id=None, post_process_function=None, eval_from_path=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_function = post_process_function
        self.eval_from_path = eval_from_path

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self._max_length = max_length
        self._num_beams = num_beams
        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            logger.info("Post-processing generations......")
            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")

            eval_preds, eval_golds = self.post_process_function(output.predictions, output.label_ids, self.tokenizer, output_dir)

            metrics = self.compute_metrics(output_dir + "/predictions.txt", output_dir + "/references.txt") if self.eval_from_path else self.compute_metrics(eval_preds, eval_golds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            output.metrics.update(metrics)

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        return output.metrics


class GPT2Seq2SeqTrainer(Seq2SeqTrainerWithMetrics):
    def __init__(self, *args, pad_token_id=None, post_process_function=None, eval_from_path=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_function = post_process_function
        self.eval_from_path = eval_from_path
        self.pad_token_id = pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        
        assert inputs["input_ids"].shape[0] == 1 # this code only works when batch size is 1
        has_labels = "labels" in inputs

        current_seq_len = inputs["input_ids"].shape[-1]
        input_len = torch.sum(inputs["attention_mask"])
        input_gen_len = torch.sum(inputs["labels"][0,:input_len].eq(-100))
        gen_inputs = {}
        gen_inputs["input_ids"] = inputs["input_ids"][:,:input_gen_len].clone()
        gen_inputs["attention_mask"] = inputs["attention_mask"][:,:input_gen_len].clone()
        gen_inputs["token_type_ids"] = inputs["token_type_ids"][:,:input_gen_len].clone()
        gen_inputs = self._prepare_inputs(gen_inputs)

        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length+current_seq_len if self._max_length is not None else self.model.config.max_length+current_seq_len,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        }

        generated_tokens = self.model.generate(
            gen_inputs["input_ids"],
            attention_mask=gen_inputs["attention_mask"],
            token_type_ids=gen_inputs["token_type_ids"],
            **gen_kwargs,
        )
        generated_tokens = generated_tokens[:, input_gen_len:]
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"][:, input_gen_len:input_len]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        # print("generated_tokens", self.tokenizer.decode(generated_tokens[0, input_gen_len:], skip_special_tokens=True), generated_tokens.shape)
        # print("labels", self.tokenizer.decode(labels[0], skip_special_tokens=True), labels.shape)
        # input()
        return (loss, generated_tokens, labels)
    
    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenzier was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (self.pad_token_id)

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        """
        During evaluation, add max_length and beam_seach
        """
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)