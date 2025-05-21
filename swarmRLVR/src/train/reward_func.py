import torch


def reward(prompt, query, label):
    return float(label)


# rewards is stored in labels.
def reward_func(queries, prompts, labels):
    rewards = [reward(prompt, query, label) for query, prompt, label in zip(queries, prompts, labels)]
    return torch.tensor(rewards)
