import pandas as pd

from utils.leafstats import *


def get_at_path(dictio: dict, path: list[str]):
    for sp in path[:-2]:
        dictio = dictio[path]
    return dictio.get(path[-1])


def produce_tabular_render(filepath: str, metricpath: list[str], outpath: str = None):
    """
    TODO: docstring
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    rollout_paths = data.keys()
    for rollout_path in rollout_paths:
        if outpath is None:
            m_path = rollout_path.replace("/", "|")
            m_path = m_path.replace(".json", "")
            m_path = (
                os.path.split(filepath)[0]
                + "/contextualized_tabular_renders/"
                + m_path
                + "_tabular_render.csv"
            )
        # import pdb; pdb.set_trace()
        os.makedirs(os.path.split(m_path)[0], exist_ok=True)
        metrics = data[rollout_path]
        d = {k: [] for k in metrics[0].keys()}
        for m in metrics:
            for k, v in m.items():
                d[k].append(v)
        d = pd.DataFrame(d)
        d.to_csv(m_path)


if __name__ == "__main__":
    produce_tabular_render(
        filepath="/home/mila/d/dereck.piche/llm_negotiation/tests/outputs_for_tests/tally_test_output/contextualized_training_metrics_2025-05-31 21:32:12.508003.json",
        metricpath=["token_entropy_terms"],
    )
