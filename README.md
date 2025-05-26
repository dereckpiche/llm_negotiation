# Reinforcement Learning Leads to Myopic LLM Play in Social Dilemmas

## Abstract

"As agentic AI becomes more widespread, agents with separate and possibly conflicting goals will interact in complex ways.
Social dilemmas pose a fundamental challenge where individual agents' incentives can undermine collective welfare.
While reinforcement learning (RL) has been instrumental in fine-tuning large language models (LLMs) for state-of-the-art performance, prior work suggests that naive RL can lead to unfair and welfare-reducing behaviors in small scale neural networks.
We hypothesize that similar effects may emerge in LLMs, despite their initial alignment toward fairness and cooperation.
We investigate how state-of-the-art LLMs behave and learn in classic social dilemmas, including the prisoners dilemma and negotiation games. To the best of our knowledge, we are the first to explore RL methods for fine-tuning LLMs in a multi-agent setting.
Our results show that naive RL fine-tuning produces agents that behave unfairly or reduce overall welfare.
While shared reward mechanisms can encourage cooperation, they leave agents vulnerable to exploitation by selfish counterparts.
Finally, we find that even API-accessible LLMs are exploitable by trained selfish agents, highlighting a persistent gap in their ability to handle social dilemmas effectively."


## Installation

```bash
pip install -e .
```

## Development
It is recommended to use python version `3.10.11`.

```bash
pip install -r requirements_dev.txt
pre-commit install
```
The most useful files to understand the codebase are:
- `src/experiments/generate_and_train.py`: This file contains the code for the online RL pipeline. (Generate, Train, Repeat.)
- `src/environments/dond/*`: Contains the implementation of *Trust and Split*.
- `src/environments/ipd/*`: Contains the implementation of *Iterated Prisonner's Dilemma*.


## Reproducing the Results

Each config file corresponds to a different experiment in the paper.

To reproduce the results, run the following command:

```bash
python src/run.py --config-path your-path-to-configs-folder --config-name your-config-name
```
