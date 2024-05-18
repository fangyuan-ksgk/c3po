# Efficient Learning from Verbal Feedback (ELVF)
## Overview
Provided with a simple verbal feedback "Do not talk about elephant", LLM struggles to adhere to the requirement. When asked "What is the largest land mammal on Earth?" GPT-4o fails to stick to the feedback. 

Human picks up this concept much more efficiently. This is a combination of reasoning, short-term memory, and long-term adaptation. 

This code base aims to mimic that behavior.

### Self-Consistency Searching: 

Given a verbal feedback, LLM struggles to follow it on some queries and nails it on others. In order to enhance LLM's performance, we iteratively asks LLM to check whether its response "make sense" -- this adaptively allocate computation to obtain a good response on all queries, which LLM itself is happy with. [REASONING]

### Self-Distillation FineTuning: 

This Fine Tuning algorithm aims at improving the efficiency in fine-tuning process. LLM learns a compression of the knowledge corpus used in training, such knowlegde is revealed in its complicated logit vector prediction. Supervision with a one-hot vector at a very small scales leads to collapses of such logit vector prediction, this is what happens with traditional Supervised Fine-tuning algorithm. To adrress this, we propose a 'Self Distillation Fine Tuing' algorithm, which provides a similar complicated dense logit vector as supervision, aiming to achieve a better efficiency in learning. [ADAPTATION]

```bash
python src/train_v2.py --arg_file configs/config_dft_v1.json
```


