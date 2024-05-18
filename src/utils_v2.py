import torch
from typing import Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32
}

@dataclass
class ModelArguments:
    base_model_id: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct"
    new_model_id: Optional[str] = "Yo-01"
    use_quant: Optional[bool] = True
    load_in_4bit: Optional[bool] = True
    bnb_4bit_use_double_quant: Optional[bool] = True
    bnb_4bit_quant_type: Optional[str] = "nf4"
    bnb_4bit_compute_dtype: Optional[torch.dtype] = torch.bfloat16
    device_map: Optional[str] = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"
    torch_dtype: Optional[torch.dtype] = torch.bfloat16

    def make(self):
        """ 
        Make the args into LLM model
        """
        if self.use_quant:
            # BitsAndBytesConfig int-4 config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        if torch.cuda.is_available():
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                device_map="auto",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config
            )
        else:
            print("Loading LLM without GPU")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                device_map="mps") # MLX should have some support already for LoRA script

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        tokenizer.padding_side = 'right' # to prevent warnings

        if "Meta-Llama-3-" in self.base_model_id:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer.pad_token = tokenizer.unk_token

        return model, tokenizer
    

@dataclass
class PeftArguments:
    lora_alpha: Optional[int] = 128
    lora_dropout: Optional[float] = 0.05
    r: Optional[int] = 256
    bias: Optional[str] = "none"
    target_modules: Optional[str] = "all-linear"
    task_type: Optional[str] = "CAUSAL_LM"

    def make(self):
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias=self.bias,
            target_modules=self.target_modules,
            task_type=self.task_type
        )
    

