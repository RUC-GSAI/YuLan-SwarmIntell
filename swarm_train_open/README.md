## An implementation for swarm RL training

In this project we showcase a simple implementation for swarm RL training. We made a synchronous version adapted from the original [SwarmBench](https://github.com/RUC-GSAI/YuLan-SwarmIntell), which provides a way to extract prompts from the swarm environment, generate responses and feed them back to the environment, allowing us to train LLM in [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) or other fine-tuning framework.

We have not adjusted the hyper params to hit the best performance, however. This is a simple demo to show the usage of our framework.