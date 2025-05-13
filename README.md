# Large Language Models Negotiation

## Introduction
This library serves as a framework to run experiments with large language models in a negotiation setting.

## Installation

```bash
pip install -e .
```

## Development
```bash
pip install -r requirements_dev.txt
pre-commit install
```


## Environments

### Deal or No Deal (DoND)

The DoND environment is a simple negotiation environment where two agents take turns proposing deals. The agent can either accept the deal or make a counter-proposal. The game ends when one of the agents accepts the deal or the counter-proposal is accepted.

After running the experiments, you can generate statistics for the DoND environment with the following command:

Make sure to go into the file and adjust the source path to the path of the data you want to analyze.

```bash
PYTHONPATH=src python -m src.environments.dond.dond_statistics --path /path/to/your/dond/data --agent_id Alice
```

### Iterated Prisoner's Dilemma (IPD)

```bash
PYTHONPATH=src python -m src.environments.ipd.ipd_statistics --path /path/to/your/ipd/data --agent_id Alice
```

### Diplomacy

The Diplomacy environment is a more complex negotiation environment where two agents take turns proposing deals. The agent can either accept the deal or make a counter-proposal. The game ends when one of the agents accepts the deal or the counter-proposal is accepted.

###
