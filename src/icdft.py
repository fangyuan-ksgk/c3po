import warnings
from contextlib import nullcontext
from typing import Union, Dict, Any, Tuple, List, Literal, Optional

import torch
import numpy as np
from torch import nn
from trl import SFTTrainer
from transformers import PreTrainedModel
from .dataset.format_v2 import get_teacher_input_ids

##############################################
# ICDFT: In Context Distillation Fine Tuning #
##############################################

def masked_dl_div(pred_logits: torch.Tensor, 
                  teacher_logits: torch.Tensor, 
                  attention_mask: torch.Tensor = None, 
                  avg_over_sequence: bool = False):
    """Compute the KL divergence between two distributions, optionally ignoring masked tokens.
    
    Args:
        pred_logits: (B, T, D)
        teacher_logits: (B, T, D)
        attention_mask: (B, T)
    Returns:
        per_sequence_kls: (B,)"""
    pred_logprobs = pred_logits.log_softmax(-1) # (B, T, D)
    teacher_logprobs = teacher_logits.log_softmax(-1) # (B, T, D)
    per_token_kls = (teacher_logprobs.exp() * (teacher_logprobs - pred_logprobs)).sum(-1) # (B, T)
    masked_kls = per_token_kls * (attention_mask if attention_mask is not None else 1) # (B, T) | In this sense, the mask needs not be binary -- Weighted Loss is pretty visible from here already
    if avg_over_sequence:
        # Weighted Average of KL Divergence loss over the sequence (the division here is just to normalize the weights --> attentoin_mask)
        per_sequence_kls = masked_kls.sum(-1) / (attention_mask.sum(-1) if attention_mask is not None else masked_kls.shape[-1]) # (B,)
    else:
        per_sequence_kls = masked_kls.sum(-1) # (B,)
    return per_sequence_kls.mean(0) # scalar


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


# Naturally SFT takes in one model only (self-distillation loss includes a prompted model itself ...)
class InContextDistillTrainer(SFTTrainer):
    """ 
    Label will be used as one-shot prompt for self-distillation loss
    - Paradim shift from Supervised Fine-Tuning to Self-Distillation
    - It's important to customize training experience for different models: do they understand same thing at the same level?
    """
    def __init__(self, *args, get_teacher_query = None, template_patterns = None, 
                 response_template: str = "[/INST]", ignore_index: int = -100,
                 kd_temperature: float = 5, kd_lambda: float = 0.5,
                 **kwargs):
        
        self.get_teacher_query = get_teacher_query
        self.template_patterns = template_patterns
        self.response_template = response_template
        self.ignore_index = ignore_index
        super().__init__(*args, **kwargs)

    def get_completion_only_labels(self, input_ids: list[list[int]]) -> list[list[int]]:
        labels = torch.tensor(input_ids).clone()
        response_token_ids_start_idx = None

        # Find location on string level
        format_prompt = self.tokenizer.decode(input_ids)
        idx = format_prompt.find(self.response_template)
        prefix = format_prompt[:idx + len(self.response_template)]
        suffix = format_prompt[idx + len(self.response_template):]
        # Backward propagate to token level | Want the model to predict the next token for us
        prefix_tokens = self.tokenizer.tokenize(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.tokenize(suffix, add_special_tokens=False)
        diff = len(input_ids) - len(prefix_tokens) - len(suffix_tokens)
        response_token_ids_start_idx = len(prefix_tokens) + diff
        
        if response_token_ids_start_idx is None:
            warnings.warn(
                f"Could not find response key `{self.response_template}` in the "
                f'following instance: {self.tokenizer.decode(input_ids)} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            labels[:] = self.ignore_index
        else:
            response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
            labels[:response_token_ids_end_idx] = self.ignore_index # Mask out input tokens

        return labels.tolist()
    
    def compute_self_distillation_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train", # Literal is likely similar to Enum
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Compute Self Distillation Loss.
        """
        metrics = {}

        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"]
        }
        student_output = model(**batch)   
        
        with torch.no_grad():

            teacher_input_ids = get_teacher_input_ids(batch, 
                                                      self.template_patterns,
                                                      self.tokenizer, 
                                                      self.get_teacher_query)   
            # Convert Input_ids back into string and add the teacher prompt query format
            teacher_output = model(teacher_input_ids)                
            
        attention_mask = batch.get('attention_mask', None)
        attention_mask_student = attention_mask * (batch["labels"] != -100)
        teacher_logits = teacher_output.logits / self.kd_temperature
        student_logits = student_output.logits / self.kd_temperature

        # Compute Distillation Loss
        self_distill_loss = masked_self_dl(student_logits, teacher_logits, attention_mask_student, self.use_avg_kl) * (self.kd_temperature ** 2)

        # Compute Perplexity Loss
        if self.custom_sft_loss:
            student_target_loss = masked_lm_loss(student_logits, batch["labels"], attention_mask_student)
        else:
            student_target_loss = student_output.loss

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
        """
        Compute Loss Value
        """
        loss, metric = self.compute_self_distillation_loss(model, inputs, train_eval="train")
        return (loss, metric) if return_outputs else loss
    
    def _prepare_non_packed_dataloader(
        self,
        tokenizer, # Tokenizer 
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        """ 
        This is the dataloader which helps separating student & teacher batch inputs
        """
        dataset = super()._prepare_non_packed_dataloader(
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func,
            add_special_tokens,
            remove_unused_columns,
        )
        return dataset