import pandas as pd

from utils.leafstats import *


def get_at_path(dictio: dict, path: list[str]):
    for sp in path[:-2]:
        dictio = dictio[path]
    return dictio.get(path[-1])



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


def open_interactive(inpath: str):

    with open(inpath, "r") as f:
        data = json.load(f)


    print("Data in variable 'data'")
    print("Use data.keys() to naviguate")
    import pdb; pdb.set_trace()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLT for Training Metrics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # tabular render
    tab_cont_parser = subparsers.add_parser("tab_render", help="Tabular Renderer for Contextualized Data")
    tab_cont_parser.add_argument("--inpath", required=True)
    tab_cont_parser.add_argument("--outpath", required=False)
    tab_cont_parser.set_defaults(func=produce_tabular_render)

    # Interactive naviguator
    int_nav_parser = subparsers.add_parser("i_nav", help="Interactive naviguator.")
    int_nav_parser.add_argument("--inpath", required=True)
    int_nav_parser.set_defaults(func=open_interactive)


    args = parser.parse_args()
    import inspect
    func_params = inspect.signature(args.func).parameters
    filtered_args = {k: v for k, v in vars(args).items() if k in func_params}
    args.func(**filtered_args)

