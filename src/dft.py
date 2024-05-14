import warnings
from typing import Union, Dict, Any, Tuple, List, Literal, Optional

import torch
import numpy as np
from torch import nn
from trl import SFTTrainer
from transformers import PreTrainedModel

def masked_self_dl(pred_logits: torch.Tensor, 
                  teacher_logits: torch.Tensor, 
                  attention_mask: torch.Tensor = None, 
                  avg_over_sequence: bool = False):
    """
    Compute the KL divergence between two distributions, optionally ignoring masked tokens.
    (T2>T1) Teacher logits are longer than student logtis (Extra input / prompt tokens in teacher logits)
    We are interested in Divergence of predicted student logits with teacher logits
    Args:
        pred_logits: (B, T1, D)
        teacher_logits: (B, T2, D)
        attention_mask: (B, T1) | Attention mask for student logits
    Returns:
        per_sequence_kls: (B,)
    """
    pred_logprobs = pred_logits.log_softmax(-1) # (B, T1, D)
    teacher_logprobs = teacher_logits.log_softmax(-1) # (B, T2, D)

    teacher_logprobs = teacher_logprobs[:, -pred_logits.shape[1]:, :] # (B, T1, D) | Teacher logits have extra tokens at the beginning

    per_token_kls = (teacher_logprobs.exp() * (teacher_logprobs - pred_logprobs)).sum(-1) # (B, T)
    masked_kls = per_token_kls * (attention_mask if attention_mask is not None else 1) # (B, T) | In this sense, the mask needs not be binary -- Weighted Loss is pretty visible from here already
    
    if avg_over_sequence:
        # Weighted Average of KL Divergence loss over the sequence (the division here is just to normalize the weights --> attentoin_mask)
        per_sequence_kls = masked_kls.sum(-1) / (attention_mask.sum(-1) if attention_mask is not None else masked_kls.shape[-1]) # (B,)
    else:
        per_sequence_kls = masked_kls.sum(-1) # (B,)
    return per_sequence_kls.mean(0) # scalar

def masked_lm_loss(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask: torch.LongTensor,
) -> torch.FloatTensor:

    # Ensure we set ignore indices where there's no attention
    labels[attention_mask == 0] = -100
    # Shift so that tokens < n predict n | i-th logit prediction is for i-th token | shift to match pred with GT
    shift_logits = logits[..., :-1, :].contiguous() # (B, T-1, D)
    shift_labels = labels[..., 1:].contiguous() # (B, T-1)

    B, T = shift_labels.shape
    shift_logits = shift_logits.view(B, -1, T) # (B, D, T-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits, shift_labels)
    return loss.sum(-1).mean(0)

class DFTTrainer(SFTTrainer):
    def __init__(self, *args, sigma_soft: float = 0.3, sigma_hard: float = 0.3, teacher_formatting_func, response_template: str = "[/INST]", 
                 kd_temperature: float = 5, kd_lambda: float = 0.5, use_avg_kl: bool = False, ignore_index: int = -100, **kwargs):
        self.response_template = response_template
        self.tokenizer = kwargs["tokenizer"]
        self.teacher_formatting_func = teacher_formatting_func
        self.ignore_index = ignore_index
        self.sigma_soft = sigma_soft
        self.sigma_hard = sigma_hard
        self.kd_temperature = kd_temperature
        self.kd_lambda = kd_lambda
        self.use_avg_kl = use_avg_kl
        super().__init__(*args, **kwargs)
    

    def get_completion_only_labels(self, input_ids: list[list[int]]) -> list[list[int]]:
        labels = torch.tensor(input_ids).clone()
        response_token_ids_end_idx = None

        # Find location on string level
        format_prompt = self.tokenizer.decode(input_ids)
        idx = format_prompt.find(self.response_template)
        if idx != -1:
            prefix = format_prompt[:idx + len(self.response_template)]
            suffix = format_prompt[idx + len(self.response_template):]
            # Backward propagate to token level | Want the model to predict the next token for us
            prefix_tokens = self.tokenizer.tokenize(prefix, add_special_tokens=False)
            suffix_tokens = self.tokenizer.tokenize(suffix, add_special_tokens=False)
            diff = len(input_ids) - len(prefix_tokens) - len(suffix_tokens)
            response_token_ids_end_idx = len(prefix_tokens) + diff

        if response_token_ids_end_idx is None:
            print("Issuing Input Id Type: ", type(input_ids))
            warnings.warn(
                f"Could not find response key `{self.response_template}` in the "
                f'following instance: {self.tokenizer.decode(input_ids)} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            labels[:] = self.ignore_index
        else:
            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[:response_token_ids_end_idx] = self.ignore_index
        return labels.tolist()
    
    def compute_distillation_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            student_batch: Dict[str, Union[List, torch.LongTensor]],
            teacher_batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train", # Literal is likely similar to Enum
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        metrics = {}

        student_output = model(**student_batch)   

        with torch.no_grad():
            teacher_output = model(**teacher_batch)              
            
        attention_mask = student_batch.get('attention_mask', None)
        attention_mask_student = attention_mask * (student_batch["labels"] != -100)
        teacher_logits = teacher_output.logits.to(model.device) / self.kd_temperature
        student_logits = student_output.logits / self.kd_temperature

        # Compute Distillation Loss
        self_distill_loss = masked_self_dl(student_logits, teacher_logits, attention_mask_student, self.use_avg_kl) * (self.kd_temperature ** 2)

        # Compute Perplexity Loss
        student_target_loss = masked_lm_loss(student_logits, student_batch["labels"], attention_mask_student)

        # Calculate final loss
        loss = (1. - self.kd_lambda) * student_target_loss + self.kd_lambda * self_distill_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}kd_loss/self_distillation_loss"] = self_distill_loss.cpu()
        metrics[f"{prefix}kd_loss/target_loss"] = student_target_loss.cpu()
        metrics[f"{prefix}kd_loss/kd_loss"] = loss.cpu()
        return loss, metrics


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

        teacher_inputs = {
            "input_ids": inputs["teacher_input_ids"],
            "attention_mask": inputs["teacher_attention_mask"],
            "labels": inputs["teacher_labels"],
        }

        loss, metric = self.compute_distillation_loss(model, student_inputs, teacher_inputs, train_eval="train")
        
        return (loss, metric) if return_outputs else loss
    
    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=False,
    ):
        # print("Check the keys within the original dataset: ", dataset[0].keys())
        dataset = super()._prepare_non_packed_dataloader(
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func,
            add_special_tokens,
            remove_unused_columns,
        )
        dataset.remove_columns(["prompt"]) # Manual Removal of useless columns

        def tokenize_teacher(row):
            row["format_teacher_prompt"] = self.teacher_formatting_func(row)
            teacher_input = tokenizer(
                    row["format_teacher_prompt"], truncation=True, padding=False, max_length=max_seq_length, add_special_tokens=False
                )
            teacher_labels = self.get_completion_only_labels(teacher_input["input_ids"])
            row["teacher_input_ids"] = teacher_input["input_ids"]
            row["teacher_attention_mask"] = teacher_input["attention_mask"]
            row["teacher_labels"] = teacher_labels
            return row

        dataset = dataset.map(tokenize_teacher, batched=False)
        dataset = dataset.remove_columns(["teacher_prompt", "completion", "format_teacher_prompt"])
        return dataset