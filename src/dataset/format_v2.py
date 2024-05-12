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
