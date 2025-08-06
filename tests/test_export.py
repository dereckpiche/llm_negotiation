from pathlib import Path

from mllm.markov_games.render_utils import *

# =========================
# CONFIG: set these values
# =========================
CONFIG = {
    "INPUT_DIR": Path(
        "/home/mila/d/dereck.piche/scratch/llm_negotiation/2025_08/debug_2025-08-05___17-34-11/seed_1000/iteration_000"
    ),  # folder containing *.json rollout files
    "OUTPUT_DIR": Path(
        "/home/mila/d/dereck.piche/llm_negotiation/tests/outputs_for_tests"
    ),  # folder to write renders
    "PER_AGENT": True,  # also write per-agent transcripts
    "INCLUDE_STATE_END": False,  # annotate <STATE_END> on lines
    "SIM_CSV": True,  # export simulation infos to CSV
    "RECURSIVE": False,  # search subfolders for JSON
}


def main():
    INPUT_DIR = CONFIG["INPUT_DIR"]
    OUTPUT_DIR = CONFIG["OUTPUT_DIR"]
    PER_AGENT = CONFIG["PER_AGENT"]
    INCLUDE_STATE_END = CONFIG["INCLUDE_STATE_END"]
    SIM_CSV = CONFIG["SIM_CSV"]
    RECURSIVE = CONFIG["RECURSIVE"]

    pattern = "**/*.json" if RECURSIVE else "*.json"
    files = sorted(INPUT_DIR.glob(pattern))
    if not files:
        print(f"No JSON files found in {INPUT_DIR} (recursive={RECURSIVE}).")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f}")
        try:
            export_chat_logs(
                path=f,
                outdir=OUTPUT_DIR,
            )
            export_html_from_rollout_tree(
                path=f,
                outdir=OUTPUT_DIR,
                main_only=True,
            )
        except Exception as e:
            print(f"!! Error in {f}: {e}")


if __name__ == "__main__":
    main()
