from datasets import Dataset


def to_dpo(search_infos):
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

def to_sft(search_infos):
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        for chosen in chosens:
            data.append({"prompt": prompt, "completion": chosen})
    dataset = Dataset.from_list(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split


def to_full(search_infos):
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
