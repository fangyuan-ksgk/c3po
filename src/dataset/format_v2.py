from datasets import Dataset
from .prompts_v2 import TEACHER_QUERY_TEMPLATE

def to_dpo(feedback):
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        rejects = [node["icl_complete"] for node in nodes if not node["accept"]]
        for chosen in chosens:
            for reject in rejects:
                data.append({"prompt": prompt, "chosen": chosen, "reject": reject})
    dataset = Dataset.from_list(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split

def to_sft(feedback):
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        for chosen in chosens:
            data.append({"prompt": prompt, "completion": chosen})
    dataset = Dataset.from_list(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split


def to_distill_sft(feedback):
    """ 
    For self distillation, I do have many advices which could be used to generate a better completion from teacher model
    But let's begin with the simplest one-shot prompting approach
    """
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]

        for chosen in chosens:
            teacher_query = TEACHER_QUERY_TEMPLATE.format(content = feedback.content, prompt = prompt, completion = chosen)
            data.append({"prompt": prompt, "completion": chosen, "teacher_prompt": teacher_query})
    dataset = Dataset.from_list(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split     

def to_full(feedback):
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        rejects = [node["icl_complete"] for node in nodes if not node["accept"]]
        for chosen in chosens:
            for reject in rejects:
                data.append({"prompt": prompt, "chosen": chosen, "reject": reject})
        for chosen in chosens:
            data.append({"prompt": prompt, "completion": chosen})
        for reject in rejects:
            data.append({"prompt": prompt, "negative": reject})
    dataset = Dataset.from_list(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split


from typing import List, Union, Optional, Dict, Any
from transformers import DataCollatorForLanguageModeling
import numpy as np
import warnings

# Better DataCollatorForCompletionOnlyLM_v2
class DataCollatorForCompletionOnlyLM_v2(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                # Token level search is a wrong idea ---> Merging words into token will make this a logical error
                input_ids = batch["input_ids"][i]
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
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch
    

def get_teacher_input_ids(batch, template_patterns, tokenizer, get_teacher_query):

    user_start_pattern = template_patterns["user_start"]
    assistant_start_pattern = template_patterns["assistant_start"]
    end_pattern = template_patterns["end"]

    messages = []
    for input_ids in batch["input_ids"]:
        input_text = tokenizer.decode(input_ids)
        prompt = input_text.split(user_start_pattern)[1].split(end_pattern)[0]
        completion = input_text.split(assistant_start_pattern)[1].split(end_pattern)[0]
        teacher_prompt = get_teacher_query(prompt, completion)
        message = [
            {"role": "user",
            "content": teacher_prompt},
            {"role": "assistant",
            "content": completion}
        ]
        messages.append(message)

    # Might've missed on the correct device
    sequences = tokenizer.apply_chat_template(messages, tokenize=False)
    teacher_input_ids = tokenizer(sequences, return_tensors="pt", padding=True)["input_ids"]
    return teacher_input_ids
