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
    print(os.path.split(filepath)[:-2])
    if outpath is None:
        outpath = (
            os.path.split(filepath)[0]
            + "/"
            + " ".join(metricpath)
            + "_tabular_render.csv"
        )
    print(outpath)
    with open(filepath, "r") as f:
        data = json.load(f)
    metrics = get_at_path(dictio=data, path=metricpath)
    d = {k: [] for k in metrics[0].keys()}
    for m in metrics:
        for k, v in m.items():
            d[k].append(v)
    d = pd.DataFrame(d)
    d.to_csv(outpath)


if __name__ == "__main__":
    produce_tabular_render(
        filepath="/home/mila/d/dereck.piche/scratch/llm_negotiation/ipd_overt_2025-05-31___13-18-02/seed_87/iteration_000/training_metrics/sp_adapter/training_metrics_2025-05-31 13:21:32.390411.json",
        metricpath=["token_entropy_terms"],
    )
