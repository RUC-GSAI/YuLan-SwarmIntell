## An implementation for swarm RL training

In this project we showcase a simple implementation for swarm RL training. We made a synchronous version adapted from the original [SwarmBench](https://github.com/RUC-GSAI/YuLan-SwarmIntell), which provides a way to extract prompts from the swarm environment, generate responses and feed them back to the environment, allowing us to train LLM in [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) or other fine-tuning framework.

We have not adjusted the hyper params to hit the best performance, however. This is a simple demo to show the usage of our framework.

## Installation


```
conda create --name swarmbench_RL \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate swarmbench_RL

pip install unsloth

pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

pip install openrlhf==0.6.3

pip install vllm==0.8.5

pip install transformers==4.51.3

```