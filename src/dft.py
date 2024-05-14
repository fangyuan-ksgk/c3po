import warnings
from typing import Union, Dict, Any, Tuple

import torch
import numpy as np
from torch import nn
from trl import SFTTrainer
from transformers import PreTrainedModel


class DFTTrainer(SFTTrainer):
    def __init__(self, *args, sigma_soft: float = 0.3, sigma_hard: float = 0.3, teacher_formatting_func, response_template: str = "[/INST]", ignore_index: int = -100, **kwargs):
        self.response_template = response_template
        self.tokenizer = kwargs["tokenizer"]
        self.teacher_formatting_func = teacher_formatting_func
        self.ignore_index = ignore_index
        self.sigma_soft = sigma_soft
        self.sigma_hard = sigma_hard
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


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        pass
        hard_inputs = {
            "input_ids": inputs["hard_negative_input_ids"],
            "attention_mask": inputs["hard_negative_attention_mask"],
            "labels": inputs["hard_negative_labels"],
        }
        soft_inputs = {
            "input_ids": inputs["soft_negative_input_ids"],
            "attention_mask": inputs["soft_negative_attention_mask"],
            "labels": inputs["soft_negative_labels"],
        }
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        outputs = model(**inputs)
        soft_outputs = model(**soft_inputs)
        hard_outputs = model(**hard_inputs)

        # Compute combined loss
        loss = outputs.loss + self.sigma_soft * soft_outputs.loss + self.sigma_hard * hard_outputs.loss

        if return_outputs:
            return (loss, outputs)
        return loss
    
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