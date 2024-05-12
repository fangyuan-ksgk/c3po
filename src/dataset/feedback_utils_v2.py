import re
import os
import json
from enum import Enum
from uuid import uuid5, UUID
from typing import Optional, Any, Callable

from langdetect import detect
from pydantic import BaseModel
from datasets import Dataset, DatasetDict

# Used to generate deterministic UUIDs for feedback
NAMESPACE_UUID = UUID("00000000-0000-0000-0000-000000000000")

# First Principle: ICL response is already good enough | If ICL is good enough, it's all about prompt-completion and error cases
# Self-supervision loss based on a verbal feedback: Loss(Prompted response, FineTuned Response)
# -- Question is how to compress the prompt into the model --> REFT
# -- Question is how to compress REFT into the mode --> Fine-Tuning 

class Feedback:
    content: str
    prompts: list # Places where feedback apply
    search_infos: dict # Search Information

    def __init__(self, content: str):
        self.content = content
        try:
            self.load_info()
            print("Loaded {} prompts".format(len(self.prompts)))
            print("Loaded {} search infos".format(len(self.search_infos)))
        except:
            print("Completion Information not found.")

    @property
    def id(self):
        return uuid5(NAMESPACE_UUID, self.content)
    
    @property
    def file_name(self):
        assert self.id is not None, "Feedback must have an ID to have a file name"
        content = self.content.lower()[:30]
        content = re.sub(r"[^a-z0-9 ]", " ", content)
        content = re.sub(r" +", " ", content)
        content = content.replace(" ", "_")
        content = content.strip()
        return f"{content}_{self.id}"
    
    def load_info(self):
        with open(f"database/{self.file_name}/prompts.json", "r") as f:
            prompts = json.load(f)

        with open(f"database/{self.file_name}/search_infos.json", "r") as f:  
            search_infos = json.load(f)

        self.prompts = prompts
        self.search_infos = search_infos
        return

    def save_info(self):
        with open(f"database/{self.file_name}/prompts.json", "w") as f:
            json.dump(self.prompts, f)

        with open(f"database/{self.file_name}/search_infos.json", "w") as f:  
            json.dump(self.search_infos, f)
        return
    
    # @staticmethod
    # def _load_dataset_dict(path: str) -> DatasetDict:
    #     with open(path, "r") as f:
    #         data = json.load(f)
    #     dataset_dict = DatasetDict() # This is a built-in huggingface dataset object 
    #     for split in data.keys():
    #         dataset_dict[split] = Dataset.from_dict(data[split]) # Why does the prompts.json itself contain different keys?
    #     return dataset_dict

    # def can_load_dataset(self, prompt_dir: str) -> None:
    #     """Checks if prompts can be loaded from a directory

    #     Args:
    #         prompt_dir (str): Directory where prompts are stored
    #     """
    #     path = os.path.join(prompt_dir, self.file_name)
    #     if not os.path.exists(os.path.join(path, "prompts.json")):
    #         return False
    #     return True
    
    # @staticmethod
    # def _dump_dataset_dict(path: str, dataset: DatasetDict) -> None:
    #     data = {}
    #     for split in dataset
    # def load_dataset(self, prompt_dir: str) -> None:
    #     """Loads prompts from a directory into the feedback object

    #     Args:
    #         prompt_dir (str): Directory where prompts are stored
    #     """
    #     path = os.path.join(prompt_dir, self.file_name)
    #     self.prompts = self._load_dataset_dict(os.path.join(path, "prompts.json"))
    
    # def dump_dataset(self, prompt_dir: str) -> None:
    #     """Dumps prompts to a directory

    #     Args:
    #         prompt_dir (str): Directory where prompts are stored
    #     """
    #     path = os.path.join(prompt_dir, self.file_name)
    #     os.makedirs(path, exist_ok=True)
    #     self._dump_dataset_dict(os.path.join(path, "prompts.json"), self.prompts)

    def update_feedback_search_completion(self):
        search_infos = {}
        for prompt in self.prompts:
            # Get completion file -- There are bunch of rejected completions, and accepted completions
            get_prompt_complete_id = lambda prompt: "search_info_"+prompt.replace(" ","-").replace(".","")
            file_name = get_prompt_complete_id(prompt)
            file_path = f"database/{self.file_name}/{file_name}.json"
            with open(file_path, "r") as f:
                search_info = json.load(f)
            search_infos[prompt] = search_info
            os.remove(file_path) # Remove File
        self.search_infos = search_infos
