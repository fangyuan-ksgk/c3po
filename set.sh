pip install -r requirements.txt
pip install  --upgrade transformers datasets accelerate evaluate bitsandbytes trl peft torch
pip install ninja packaging tensorboardX
MAX_JOBS=4 pip install flash-attn --no-build-isolation

