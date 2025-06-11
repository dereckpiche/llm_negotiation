import pandas as pd
import matplotlib.pyplot as plt

from utils.leafstats import *


def get_at_path(dictio: dict, path: list[str]):
    for sp in path[:-1]:
        dictio = dictio[sp]
    return dictio.get(path[-1])

def set_at_path(dictio: dict, path: list[str], value):
    for sp in path[:-1]:
        if sp not in dictio:
            dictio[sp] = {}
        dictio = dictio[sp]
    dictio[path[-1]] = value


def produce_tabular_render(inpath: str, outpath: str = None):
    """
    TODO: docstring
    """
    with open(inpath, "r") as f:
        data = json.load(f)
    rollout_paths = data.keys()
    for rollout_path in rollout_paths:
        if outpath is None:
            m_path = rollout_path.replace("/", "|")
            m_path = m_path.replace(".json", "")
            m_path = (
                os.path.split(inpath)[0]
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

def get_metric_paths(data: list[dict]):
    d = data[0]
    paths = []
    def traverse_dict(d, current_path=[]):
        for key, value in d.items():
            new_path = current_path + [key]
            if isinstance(value, dict):
                traverse_dict(value, new_path)
            else:
                paths.append(new_path)
    traverse_dict(d)
    return paths

def print_metric_paths(data: list[dict]):
    paths = get_metric_paths(data)
    for p in paths: print(p)


def get_metric_iteration_list(data: list[dict], metric_path: list[str]):
    import copy
    sgl = []
    for d in data:
        sgl.append(get_at_path(d, metric_path))


def to_1d_numeric(x):
    """Return a 1‑D float array (or None if not numeric)."""
    if x is None:
        return None
    # Scalars
    if isinstance(x, (int, float, np.number)):
        return np.array([float(x)])

    # Anything list‑like → recurse & flatten
    try:
        flat = np.array(list(np.ravel(x)), dtype=float)
        return flat
    except Exception:
        return None    # skip non‑numeric or badly structured payloads


def get_single_metric_vector(data, metric_path, iterations=None):
    vecs = []
    for d in data:
        ar = get_at_path(d, metric_path)
        arr = to_1d_numeric(ar)
        if arr is not None:
            vecs.append(arr)

    return np.concatenate(vecs) if vecs else np.empty(0, dtype=float)



def get_iterations_data(iterations_path: str):
    iterations_data = []
    more_iterations = True
    paths = os.listdir(iterations_path)
    n = 0
    iteration_path = os.path.join(iterations_path, f"iteration_{n:03d}")
    while more_iterations:
        if os.path.isdir(iteration_path):
            for root, dirs, files in os.walk(iteration_path):
                for file in files:
                    if file.startswith("basic_training_metrics"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            iterations_data.append(json.load(f))
        else:
            more_iterations = False
        n += 1
        iteration_path = os.path.join(iterations_path, f"iteration_{n:03d}")
    return iterations_data




if __name__ == "__main__":



    import argparse

    parser = argparse.ArgumentParser(description="Produce training statistics.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing iterations",
    )
    args = parser.parse_args()


    data = get_iterations_data(args.path)
    print(f"\n{len(data)} iteration training training metrics files loaded in 'data' variable.")
    print(
        """

        Available methods are:
            get_metric_paths(data: list[dict])
            get_metric_iteration_list(data: list[dict], metric_path: list[str])
            get_single_metric_vector(data: list[dict], metric_path: list[str], iterations: list[int])
        """
    )
    import pdb; pdb.set_trace()


    

