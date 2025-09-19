from __future__ import annotations

import csv
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from mllm.markov_games.rollout_tree import *

try:
    # Re-export moved helpers for backward compatibility
    from basic_render import (
        find_iteration_folders,
        gather_rollout_trees,
        get_rollout_trees,
    )
except Exception:
    pass

# --------------------------------------------------------------------------------------
# Fetch external rollout trees
# --------------------------------------------------------------------------------------


def find_iteration_folders(global_folder):
    """Find all iteration_* folders within the global folder structure."""
    global_path = Path(global_folder)

    # Look for iteration_* folders in all subdirectories
    iteration_folders = []

    # Search in the global folder itself
    for item in global_path.glob("iteration_*"):
        if item.is_dir():
            iteration_folders.append(item)

    # Search in seed_* subdirectories
    for seed_dir in global_path.glob("seed_*/"):
        if seed_dir.is_dir():
            for item in seed_dir.glob("iteration_*"):
                if item.is_dir():
                    iteration_folders.append(item)

    return sorted(iteration_folders)


def gather_rollout_trees(iteration_folder):
    """Gather all rollout trees from the iteration folder (.pkl only)."""
    rollout_trees = []
    iteration_path = Path(iteration_folder)
    for item in iteration_path.glob("**/*.rt.pkl"):
        with open(item, "rb") as f:
            data = pickle.load(f)
        # Validate dicts back into Pydantic model for downstream use
        rollout_tree = RolloutTreeRootNode.model_validate(data)
        rollout_trees.append(rollout_tree)
    return rollout_trees


def get_rollout_trees(global_folder) -> list[list[RolloutTreeRootNode]]:
    """Get all rollout trees from the global folder."""
    iteration_folders = find_iteration_folders(global_folder)
    rollout_trees = []
    for iteration_folder in iteration_folders:
        rollout_trees.append(gather_rollout_trees(iteration_folder))
    return rollout_trees


# --------------------------------------------------------------------------------------
# Gather data from rollout tree methods
# --------------------------------------------------------------------------------------


def load_rollout_tree(path: Path) -> RolloutTreeRootNode:
    """Load a rollout tree from a PKL file containing a dict."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return RolloutTreeRootNode.model_validate(data)


@dataclass
class RolloutNodeList:
    id: str
    nodes: List[RolloutTreeNode]


def get_rollout_tree_paths(
    root: RolloutTreeRootNode, mgid: Optional[str] = None
) -> Tuple[RolloutNodeList, List[RolloutNodeList]]:
    """
    Returns:
        main_path: The main path from the root to the end of the tree.
        branch_paths: A list of all branch paths from the root to the end of the tree.
        Each branch path contains a list of nodes that are part of the branch, including the nodes from the main path before the branch was taken.
    """
    branch_paths = []

    def collect_path_nodes(current) -> List[RolloutTreeNode]:
        """Recursively collect all nodes in a path starting from current node."""
        if current is None:
            return []

        if isinstance(current, RolloutTreeNode):
            return [current] + collect_path_nodes(current.child)

        elif isinstance(current, RolloutTreeBranchNode):
            # For branch nodes, we only follow the main_child for path collection
            if current.main_child:
                return [current.main_child] + collect_path_nodes(
                    current.main_child.child
                )
            else:
                return []

    def traverse_for_branches(
        current,
        main_path_prefix: List[RolloutTreeNode],
        path_id: str,
        current_time_step: Optional[int] = 0,
    ):
        """Traverse tree to collect all branch paths."""
        if current is None:
            return

        if isinstance(current, RolloutTreeNode):
            # Continue traversing with this node added to the main path prefix
            new_prefix = main_path_prefix + [current]
            traverse_for_branches(current.child, new_prefix, path_id, current.time_step)

        elif isinstance(current, RolloutTreeBranchNode):
            # Collect all branch paths
            if current.branches:
                for agent_id, branch_node_list in current.branches.items():
                    if branch_node_list:
                        # Start with the main path prefix, then recursively collect all nodes in this branch
                        branch_path_nodes = main_path_prefix.copy()
                        for branch_node in branch_node_list:
                            branch_path_nodes.extend(collect_path_nodes(branch_node))

                        # Create proper branch path ID with mgid, agent_id, and time_step
                        mgid_str = mgid or str(root.id)
                        branch_path_id = f"mgid:{mgid_str}_type:branch_agent:{agent_id}_time_step:{current_time_step}"
                        branch_paths.append(
                            RolloutNodeList(id=branch_path_id, nodes=branch_path_nodes)
                        )

            # Process the main child and add to prefix
            new_prefix = main_path_prefix
            if current.main_child:
                new_prefix = main_path_prefix + [current.main_child]

            # Continue traversing the main path
            if current.main_child:
                traverse_for_branches(
                    current.main_child.child,
                    new_prefix,
                    path_id,
                    current.main_child.time_step,
                )

    # Collect the main path nodes
    main_path_nodes = collect_path_nodes(root.child)

    # Traverse to collect all branch paths
    traverse_for_branches(root.child, [], "")

    # Create the main path with proper mgid format
    mgid_str = mgid or str(root.id)
    main_path = RolloutNodeList(id=f"mgid:{mgid_str}_type:main", nodes=main_path_nodes)

    return main_path, branch_paths


class ChatTurnLog(BaseModel):
    time_step: int
    agent_id: str
    role: str
    content: str
    is_state_end: bool
    reward: float


def gather_agent_chat_turns_for_path(
    agent_id: str, path: RolloutNodeList
) -> List[ChatTurnLog]:
    """Iterate through all chat turns for a specific agent in a path sorted by time step."""
    turns = []
    for node in path.nodes:
        action_log = node.step_log.action_logs.get(agent_id, [])
        if action_log:
            for chat_turn in action_log.chat_turns or []:
                turns.append(
                    ChatTurnLog(
                        time_step=node.time_step,
                        agent_id=agent_id,
                        role=chat_turn.role,
                        content=chat_turn.content,
                        is_state_end=chat_turn.is_state_end,
                        reward=node.step_log.simulation_step_log.rewards.get(
                            agent_id, 0
                        ),
                    )
                )
    return turns


def gather_all_chat_turns_for_path(path: RolloutNodeList) -> List[ChatTurnLog]:
    """Iterate through all chat turns for all agents in a path sorted by time step."""
    turns = []

    # Collect turns from all agents, but interleave them per timestep by (user, assistant) pairs
    for node in path.nodes:
        # Build (user[, assistant]) pairs for each agent at this timestep
        agent_ids = sorted(list(node.step_log.action_logs.keys()))
        per_agent_pairs: Dict[str, List[List[ChatTurnLog]]] = {}

        for agent_id in agent_ids:
            action_log = node.step_log.action_logs.get(agent_id)
            pairs: List[List[ChatTurnLog]] = []
            current_pair: List[ChatTurnLog] = []

            if action_log and action_log.chat_turns:
                for chat_turn in action_log.chat_turns:
                    turn_log = ChatTurnLog(
                        time_step=node.time_step,
                        agent_id=agent_id,
                        role=chat_turn.role,
                        content=chat_turn.content,
                        is_state_end=chat_turn.is_state_end,
                        reward=node.step_log.simulation_step_log.rewards.get(
                            agent_id, 0
                        ),
                    )

                    if chat_turn.role == "user":
                        # If a previous pair is open, close it and start a new one
                        if current_pair:
                            pairs.append(current_pair)
                            current_pair = []
                        current_pair = [turn_log]
                    else:
                        # assistant: attach to an open user message if present; otherwise stand alone
                        if (
                            current_pair
                            and len(current_pair) == 1
                            and current_pair[0].role == "user"
                        ):
                            current_pair.append(turn_log)
                            pairs.append(current_pair)
                            current_pair = []
                        else:
                            # No preceding user or already paired; treat as its own unit
                            pairs.append([turn_log])

                if current_pair:
                    # Unpaired trailing user message
                    pairs.append(current_pair)

            per_agent_pairs[agent_id] = pairs

        # Interleave pairs across agents: A1, B1, A2, B2, ...
        index = 0
        while True:
            added_any = False
            for agent_id in agent_ids:
                agent_pairs = per_agent_pairs.get(agent_id, [])
                if index < len(agent_pairs):
                    for tl in agent_pairs[index]:
                        turns.append(tl)
                    added_any = True
            if not added_any:
                break
            index += 1

    return turns


def chat_turns_to_dict(chat_turns: Iterator[ChatTurnLog]) -> Iterator[Dict[str, Any]]:
    """Render all chat turns for a path as structured data for JSON."""
    for chat_turn in chat_turns:
        yield chat_turn.model_dump()


def get_all_agents(root: RolloutTreeRootNode) -> List[str]:
    """list of all agent IDs that appear in the tree."""
    if root.child is None:
        return []

    # Get the first node to extract all agent IDs
    first_node = root.child
    if isinstance(first_node, RolloutTreeBranchNode):
        first_node = first_node.main_child

    if first_node is None:
        return []

    # All agents should be present in the first node
    agents = set(first_node.step_log.action_logs.keys())
    agents.update(first_node.step_log.simulation_step_log.rewards.keys())

    return sorted(list(agents))


def gather_agent_main_rewards(agent_id: str, path: RolloutNodeList) -> List[float]:
    """Gather main rewards for a specific agent in a path."""
    rewards = []
    for node in path.nodes:
        reward = node.step_log.simulation_step_log.rewards[agent_id]
        rewards.append(reward)
    return rewards


def gather_all_rewards(path: RolloutNodeList) -> List[Dict[AgentId, float]]:
    """Gather main rewards from main trajectory in a path."""
    rewards = []
    for node in path.nodes:
        rewards.append(node.step_log.simulation_step_log.rewards.copy())
    return rewards


def gather_simulation_stats(
    path: RolloutNodeList,
    filter: Callable[[SimulationStepLog], bool],
    stat_func: Callable[[SimulationStepLog], Any],
) -> List[Any]:
    """Gather stats from main trajectory in a path."""
    stats = []
    for node in path.nodes:
        sl = node.step_log.simulation_step_log
        if filter(sl):
            stats.append(stat_func(sl))
    return stats


def gather_simulation_infos(path: RolloutNodeList) -> List[Dict[str, Any]]:
    """Gather simulation information from main trajectory in a path."""
    infos = []
    for node in path.nodes:
        infos.append(node.step_log.simulation_step_log.info)
    return infos


def export_chat_logs(path: Path, outdir: Path):
    """Process a rollout tree PKL file and generate a JSONL of chat turns as dicts.
    Each line contains an object with path_id and chat_turns for a single path.
    """
    import json

    root = load_rollout_tree(path)
    mgid = root.id

    main_path, branch_paths = get_rollout_tree_paths(root)
    all_paths = [main_path] + branch_paths

    outdir.mkdir(parents=True, exist_ok=True)
    output_file = outdir / f"mgid:{mgid}_plucked_chats.render.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for path_obj in all_paths:
            chat_turns = gather_all_chat_turns_for_path(path_obj)
            output_obj = {
                "path_id": str(path_obj.id),
                "chat_turns": list(chat_turns_to_dict(iter(chat_turns))),
            }
            f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# HTML exports
# --------------------------------------------------------------------------------------


def html_from_chat_turns(chat_turns: List[ChatTurnLog]) -> str:
    """
    Render chat turns as a single, wrapping sequence of messages in time order.
    Keep badge and message bubble styles, include time on every badge and
    include rewards on assistant badges. Each message is individually
    hide/show by click; when hidden, only the badge remains and "(...)" is
    shown inline (not inside a bubble).
    """
    import html

    # Prepare ordering: sort by (time_step, original_index) to keep stable order within same step
    indexed_turns = list(enumerate(chat_turns))
    indexed_turns.sort(key=lambda t: (t[1].time_step, t[0]))
    assistant_agents = sorted({t.agent_id for t in chat_turns if t.role == "assistant"})
    enable_split_view = len(assistant_agents) == 2

    # CSS styles (simplified layout; no time-step or agent-column backgrounds)
    css = """
    <style>
        :root {
            --font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            --bg: #ffffff;
            --text: #1c0b00;
            --muted-text: #2C3E50;
            --accent-muted: #BDC3C7;
            --accent-muted-2: #D0D7DE;
            --panel-bg: #F8FAFC;
            --reward-color: #3a2e00; /* dark text for reward pill */
            --font-size: 15px;
            --small-font-size: 13px;
            --group-label-font-size: 12px;
            --border-width: 2px;
            --corner-radius: 6px;
            --pill-radius-left: 999px 0 0 999px;
            --pill-radius-right: 0 999px 999px 0;
            --inset-shadow: 0 1px 0 rgba(0,0,0,0.03) inset;
        }
        body {
            font-family: var(--font-family);
            margin: 16px;
            background-color: var(--bg);
            color: var(--text);
            font-size: var(--font-size);
            line-height: 1.6;
        }
        .messages-flow { display: block; }
        .split-wrapper { display: flex; gap: 4px; align-items: flex-start; position: relative; }
        .split-col { flex:1 1 0; min-width:0; }
        /* In split view keep same inline density as linear view */
        .split-col .chat-turn { display: inline; }
        .split-wrapper.resizing { user-select: none; }
    .split-resizer { width:4px; cursor: col-resize; flex:0 0 auto; align-self: stretch; position: relative; background: linear-gradient(90deg, rgba(224,230,235,0), var(--accent-muted-2) 30%, var(--accent-muted-2) 70%, rgba(224,230,235,0)); border-radius:2px; transition: background .15s ease, width .15s ease; }
    .split-resizer:hover { background: linear-gradient(90deg, rgba(224,230,235,0), var(--accent-muted) 35%, var(--accent-muted) 65%, rgba(224,230,235,0)); }
    .split-resizer.dragging { background: linear-gradient(90deg, rgba(224,230,235,0), var(--accent-muted) 25%, var(--accent-muted) 75%, rgba(224,230,235,0)); }
        /* tighten spacing */
        .split-col .group-divider { margin:4px 0 2px 0; }
        .toolbar {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 0;
            font-size: var(--small-font-size);
            max-height: 0;
            overflow: hidden;
            opacity: 0;
            pointer-events: none;
            transition: max-height 0.2s ease, opacity 0.2s ease;
        }
        .toolbar-wrap { position: sticky; top: 0; z-index: 10; background: var(--bg); }
        .toolbar-hotzone { height: 6px; }
        .toolbar-wrap:hover .toolbar { max-height: 200px; opacity: 1; pointer-events: auto; margin-bottom: 12px; }
        .toolbar input[type="number"] {
            width: 72px;
            padding: 2px 6px;
            border: 1px solid var(--accent-muted);
            border-radius: var(--corner-radius);
            background: var(--bg);
        }
        .toolbar button {
            padding: 4px 8px;
            border: 1px solid var(--accent-muted);
            background: var(--panel-bg);
            border-radius: var(--corner-radius);
            cursor: pointer;
        }
        .chat-turn {
            display: inline; /* inline like text */
            background: transparent;
            position: relative;
            cursor: pointer;
        }
        /* No agent-specific background distinctions */
        .turn-content {
            white-space: normal;
            color: var(--text);
            font-size: var(--font-size);
            display: inline; /* inline flow */
        }
        .chat-turn .agent-badge { margin-right: 0; vertical-align: baseline; }
        .agent-badge {
            display: inline;
            position: relative;
            border: var(--border-width) solid var(--accent-muted); /* slightly thicker */
            border-radius: var(--pill-radius-left); /* round left and bottom-right */
            font-size: var(--font-size);
            color: var(--muted-text);
            background: var(--panel-bg);
            box-shadow: var(--inset-shadow);
            line-height: 1.2;
            border-right: 0;
        }
        .agent-badge::after {
            content: none;
        }
        /* removed external separator; emoji is rendered inside message bubble */
        .agent-name { font-weight: 700; }
        .emoji-bw { filter: grayscale(100%); opacity: 0.95; font-size: var(--font-size); vertical-align: baseline; margin: 0; position: relative; top: -1px; line-height: 1; display: inline-block; }
        .ts-badge {
            position: relative;
            display: inline;
            border: var(--border-width) solid var(--accent-muted-2); /* slightly thicker */
            border-radius: var(--corner-radius); /* not a pill */
            font-size: var(--font-size);
            font-weight: 700;
            color: var(--muted-text);
            background: #F4F8FB; /* subtle tint */
            padding: 1px 6px; /* slight padding for visibility */
            margin-right: 8px; /* small gap from following content */
            pointer-events: auto; /* allow events so we can ignore them in JS */
        }
        /* Hide timestep badges when grouping by 1 */
        .hide-ts-badges .ts-badge { display: none; }
        /* Strong hide: completely hide collapsed turns */
        .strong-hide .chat-turn.collapsed { display: none; }
        .ts-badge::before {
            content: "";
            position: relative;
            background: var(--accent-muted-2);
            border-radius: 2px;
        }
        .agent-badge { margin-left: 6px;  }
        .message-box {
            display: inline; /* inline bubble behaving like text */
            font-size: var(--font-size);
            border: var(--border-width) solid var(--accent-muted);
            border-radius: var(--pill-radius-right); /* round left and bottom-right */
            position: relative;
            background: var(--bg);
            vertical-align: baseline;
            line-height: 1.2;
            padding-left: 0;
            border-left: 0;
        }
        .message-box::before { content: none; display: none; margin-right: 0; line-height: 1; }
        .chat-turn.agent-alice.role-assistant .message-box::before { color: #0eb224; }
        .chat-turn.agent-bob.role-assistant .message-box::before { color: #ef8323; }
        .chat-turn.collapsed .message-box::before { display: none; }
        /* Assistant bubble border colors by common agent names */
        .chat-turn.agent-alice.role-assistant .message-box { border-color: #0eb224; }
        .chat-turn.agent-bob.role-assistant .message-box { border-color: #ef8323; }
        /* Tie badge and seam to agent color for a cohesive capsule, assistants only */
    .chat-turn.agent-alice.role-assistant .agent-badge { border-color: #0eb224; background: linear-gradient(90deg, rgba(14,178,36,0.10), #ffffff 55%); }
        .chat-turn.agent-alice.role-assistant .agent-badge::after { border-right-color: #0eb224; }
        .chat-turn.agent-alice.role-assistant .turn-content::before { border-left-color: #0eb224; border-top-color: #0eb224; }
        .chat-turn.agent-alice.role-assistant .message-box { border-color: #0eb224; }

    .chat-turn.agent-bob.role-assistant .agent-badge { border-color: #ef8323; background: linear-gradient(90deg, rgba(239,131,35,0.12), #ffffff 55%); }
        .chat-turn.agent-bob.role-assistant .agent-badge::after { border-right-color: #ef8323; }
        .chat-turn.agent-bob.role-assistant .turn-content::before { border-left-color: #ef8323; border-top-color: #ef8323; }
        .chat-turn.agent-bob.role-assistant .message-box { border-color: #ef8323; }
        /* No colored agent-name; keep neutral */
        .reward {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #fff9c6; /* slightly brighter */
            color: var(--reward-color);
            font-weight: 600;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 11px;
            line-height: 1; /* exact, so flex centers vertically */
            padding: 0 4px; /* thinner capsule */
            min-height: 14px; /* ensure consistent capsule height */
            border-radius: 5px;
            border: 1px solid #e3cb37;
            margin: 0 4px; /* spacing from separators */
            vertical-align: middle;
            box-shadow: 0 1px 0 rgba(0,0,0,0.03) inset;
        }
        .message-placeholder { display: none; color: #7f8c8d; font-style: italic; }
        .chat-turn.collapsed .message-box { color: transparent; font-size: 0; display: inline-block; }
        .chat-turn.collapsed .message-box::after { content: "(...)"; color: #7f8c8d; font-style: italic; font-size: var(--font-size); line-height: 1.2; }
        .chat-turn.collapsed .agent-badge,
        .chat-turn.collapsed .message-box { opacity: 0.3; }
        /* Group divider - clearer and pretty */
        .group-divider {
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
            margin: 8px 0 4px 0;
            position: relative;
        }
        .group-divider::before,
        .group-divider::after {
            content: "";
            flex: 1 1 auto;
            height: 2px;
            background: linear-gradient(90deg, rgba(224,230,235,0), var(--accent-muted-2) 30%, var(--accent-muted-2) 70%, rgba(224,230,235,0));
        }
        .group-divider .group-label {
            display: inline-block;
            border: 1px solid var(--accent-muted);
            border-radius: 999px;
            padding: 2px 10px;
            font-size: var(--group-label-font-size);
            font-weight: 700;
            color: var(--muted-text);
            background: var(--bg);
            box-shadow: var(--inset-shadow);
            position: relative;
            z-index: 1;
        }
        /* Enhance contrast for print / export */
        body.split-mode .group-divider::before,
        body.split-mode .group-divider::after {
            background: linear-gradient(90deg, rgba(224,230,235,0), var(--accent-muted) 25%, var(--accent-muted) 75%, rgba(224,230,235,0));
        }
        .chat-turn .turn-content { position: relative; }
        .chat-turn .turn-content::before {
            content: none;
        }
        .chat-turn .agent-badge {
            position: relative;
        }
        /* removed absolute-positioned emoji to prevent overlap */
    </style>
    """

    # HTML structure
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>Chat Turns</title>",
        css,
        "<script>\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        "  const linearFlow = document.getElementById('flow-linear');\n"
        "  const splitFlow = document.getElementById('flow-split');\n"
        "  let splitViewOn = false;\n"
        "  function activeFlows() { return [splitViewOn && splitFlow ? splitFlow : null, linearFlow].filter(Boolean).filter(f => f.style.display !== 'none'); }\n"
        "  // State for range filtering and strong hide\n"
        "  let currentRangeStart = null;\n"
        "  let currentRangeEnd = null;\n"
        "  let strongHideOn = false;\n"
        "  document.body.addEventListener('click', function(e){\n"
        "    if (e.target.closest('.ts-badge')) { return; }\n"
        "    const turn = e.target.closest('.chat-turn');\n"
        "    if (turn) { e.stopPropagation(); turn.classList.toggle('collapsed'); }\n"
        "  });\n"
        "  function applyRangeFilter() {\n"
        "    for (const flow of activeFlows()) {\n"
        "      const turns = Array.from(flow.querySelectorAll('.chat-turn'));\n"
        "      for (const el of turns) {\n"
        "        const t = parseInt(el.getAttribute('data-time-step') || '0', 10);\n"
        "        const afterStart = (currentRangeStart === null) || (t >= currentRangeStart);\n"
        "        const beforeEnd = (currentRangeEnd === null) || (t <= currentRangeEnd);\n"
        "        el.style.display = (afterStart && beforeEnd) ? '' : 'none';\n"
        "      }\n"
        "      const dividers = Array.from(flow.querySelectorAll('.group-divider'));\n"
        "      for (const d of dividers) {\n"
        "        let anyVisible = false;\n"
        "        let el = d.nextElementSibling;\n"
        "        while (el && !el.classList.contains('group-divider')) {\n"
        "          if (el.classList.contains('chat-turn')) {\n"
        "            const disp = getComputedStyle(el).display;\n"
        "            if (disp !== 'none') { anyVisible = true; break; }\n"
        "          } else if (el.classList.contains('split-wrapper')) {\n"
        "            // Search descendants for any visible chat-turn\n"
        "            const turns = Array.from(el.querySelectorAll('.chat-turn'));\n"
        "            for (const tEl of turns) {\n"
        "              const disp2 = getComputedStyle(tEl).display;\n"
        "              if (disp2 !== 'none') { anyVisible = true; break; }\n"
        "            }\n"
        "            if (anyVisible) break;\n"
        "          }\n"
        "          el = el.nextElementSibling;\n"
        "        }\n"
        "        d.style.display = anyVisible ? '' : 'none';\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  function applyGrouping(n) {\n"
        "    function groupContainer(container, n) {\n"
        "      Array.from(container.querySelectorAll(':scope > .group-divider')).forEach(el => el.remove());\n"
        "      if (!n || n <= 0) { return; }\n"
        "      const turns = Array.from(container.querySelectorAll(':scope > .chat-turn'));\n"
        "      if (turns.length === 0) return;\n"
        "      const items = Array.from(container.children).filter(el => !el.classList.contains('group-divider'));\n"
        "      const frag = document.createDocumentFragment();\n"
        "      let lastGroup = -1;\n"
        "      for (const el of items) {\n"
        "        if (!el.classList.contains('chat-turn')) { frag.appendChild(el); continue; }\n"
        "        const t = parseInt(el.getAttribute('data-time-step') || '0', 10);\n"
        "        const g = Math.floor(t / n);\n"
        "        if (g !== lastGroup) {\n"
        "          const div = document.createElement('div');\n"
        "          div.className = 'group-divider';\n"
        "          const label = document.createElement('span');\n"
        "          label.className = 'group-label';\n"
        "          const roundIndex = g + 1;\n"
        "          label.textContent = `Round ${roundIndex}`;\n"
        "          div.appendChild(label);\n"
        "          frag.appendChild(div);\n"
        "          lastGroup = g;\n"
        "        }\n"
        "        frag.appendChild(el);\n"
        "      }\n"
        "      container.innerHTML = '';\n"
        "      container.appendChild(frag);\n"
        "      container.classList.toggle('hide-ts-badges', n === 1);\n"
        "      container.classList.toggle('strong-hide', strongHideOn);\n"
        "    }\n"
        "    for (const flow of activeFlows()) {\n"
        "      if (flow.id === 'flow-split') {\n"
        "        // Snapshot original turns once to avoid drift on repeated grouping\n"
        "        const getOriginalTurns = () => {\n"
        "          if (!flow.dataset.origData) {\n"
        "            const data = [];\n"
        "            const cols0 = flow.querySelectorAll('.split-col');\n"
        "            cols0.forEach(col => {\n"
        "              const agent = col.getAttribute('data-agent') || '';\n"
        "              col.querySelectorAll(':scope > .chat-turn').forEach(el => {\n"
        "                const t = parseInt(el.getAttribute('data-time-step')||'0',10);\n"
        "                data.push({agent, time:t, html: el.outerHTML});\n"
        "              });\n"
        "            });\n"
        "            flow.dataset.origData = JSON.stringify(data);\n"
        "          }\n"
        "          return JSON.parse(flow.dataset.origData);\n"
        "        };\n"
        "        const original = getOriginalTurns();\n"
        "        const agents = Array.from(new Set(original.map(o => o.agent))).sort();\n"
        "        const groups = new Map();\n"
        "        original.forEach(o => {\n"
        "          const g = n && n > 0 ? Math.floor(o.time / n) : 0;\n"
        "          if (!groups.has(g)) groups.set(g, new Map());\n"
        "          const gm = groups.get(g);\n"
        "          if (!gm.has(o.agent)) gm.set(o.agent, []);\n"
        "          gm.get(o.agent).push(o);\n"
        "        });\n"
        "        flow.innerHTML = '';\n"
        "        const sorted = Array.from(groups.keys()).sort((a,b)=>a-b);\n"
        "        sorted.forEach(g => {\n"
        "          const div = document.createElement('div');\n"
        "          div.className = 'group-divider';\n"
        "          const label = document.createElement('span');\n"
        "          label.className = 'group-label';\n"
        "          label.textContent = `Round ${g+1}`;\n"
        "          div.appendChild(label);\n"
        "          flow.appendChild(div);\n"
        "          const wrapper = document.createElement('div');\n"
        "          wrapper.className = 'split-wrapper';\n"
        "          agents.forEach(agent => {\n"
        "            const colDiv = document.createElement('div');\n"
        "            colDiv.className = 'split-col';\n"
        "            colDiv.setAttribute('data-agent', agent);\n"
        "            (groups.get(g).get(agent) || []).forEach(o => { colDiv.insertAdjacentHTML('beforeend', o.html); });\n"
        "            wrapper.appendChild(colDiv);\n"
        "          });\n"
        "          if (wrapper.children.length === 2) { const res = document.createElement('div'); res.className='split-resizer'; wrapper.insertBefore(res, wrapper.children[1]); }\n"
        "          flow.appendChild(wrapper);\n"
        "        });\n"
        "        flow.classList.toggle('hide-ts-badges', n === 1);\n"
        "        flow.classList.toggle('strong-hide', strongHideOn);\n"
        "        document.body.classList.add('split-mode');\n"
        "      } else {\n"
        "        groupContainer(flow, n);\n"
        "      }\n"
        "    }\n"
        "    applyRangeFilter();\n"
        "    initSplitResizers();\n"
        "  }\n"
        "  function initSplitResizers() {\n"
        "    const wrappers = document.querySelectorAll('#flow-split .split-wrapper');\n"
        "    wrappers.forEach(wrap => {\n"
        "      const resizer = wrap.querySelector('.split-resizer');\n"
        "      if (!resizer || resizer.dataset.bound) return; resizer.dataset.bound='1';\n"
        "      const cols = wrap.querySelectorAll('.split-col'); if (cols.length !== 2) return; const c0=cols[0], c1=cols[1];\n"
        "      c0.style.flex=c1.style.flex='1 1 0'; c0.style.width=c1.style.width='';\n"
        "      requestAnimationFrame(()=>{ const w0=c0.scrollWidth,w1=c1.scrollWidth,total=w0+w1||1; let p0=w0/total,p1=w1/total; const minP=0.25,maxP=0.75; if(p0<minP){p0=minP;p1=1-p0;} else if(p0>maxP){p0=maxP;p1=1-p0;} c0.style.flex='0 0 '+(p0*100).toFixed(2)+'%'; c1.style.flex='0 0 '+(p1*100).toFixed(2)+'%'; });\n"
        "      let dragging=false,startX=0,startP0=0;\n"
        "      const onDown=e=>{ dragging=true; startX=e.clientX; wrap.classList.add('resizing'); resizer.classList.add('dragging'); const rect=wrap.getBoundingClientRect(); const w=rect.width; const c0Rect=c0.getBoundingClientRect(); startP0=c0Rect.width/w; document.body.style.cursor='col-resize'; e.preventDefault(); };\n"
        "      const onMove=e=>{ if(!dragging)return; const rect=wrap.getBoundingClientRect(); const w=rect.width; let delta=(e.clientX-startX)/w; let newP0=startP0+delta; const minP=0.15,maxP=0.85; if(newP0<minP)newP0=minP; if(newP0>maxP)newP0=maxP; c0.style.flex='0 0 '+(newP0*100).toFixed(2)+'%'; c1.style.flex='0 0 '+((1-newP0)*100).toFixed(2)+'%'; };\n"
        "      const onUp=()=>{ if(!dragging)return; dragging=false; wrap.classList.remove('resizing'); resizer.classList.remove('dragging'); document.body.style.cursor=''; };\n"
        "      resizer.addEventListener('mousedown', onDown); window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);\n"
        "      resizer.addEventListener('dblclick', e=>{ if(e.shiftKey){ c0.style.flex=c1.style.flex='1 1 0'; requestAnimationFrame(()=>{ const w0=c0.scrollWidth,w1=c1.scrollWidth,total=w0+w1||1; let p0=w0/total,p1=w1/total; const minP=0.25,maxP=0.75; if(p0<minP){p0=minP;p1=1-p0;} else if(p0>maxP){p0=maxP;p1=1-p0;} c0.style.flex='0 0 '+(p0*100).toFixed(2)+'%'; c1.style.flex='0 0 '+(p1*100).toFixed(2)+'%'; }); } else { c0.style.flex='0 0 50%'; c1.style.flex='0 0 50%'; } });\n"
        "    });\n"
        "  }\n"
        "  initSplitResizers();\n"
        "  const input = document.getElementById('group-size');\n"
        "  const btn = document.getElementById('apply-grouping');\n"
        "  if (btn && input) {\n"
        "    btn.addEventListener('click', () => { const n = parseInt(input.value || '0', 10); applyGrouping(n); });\n"
        "    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { const n = parseInt(input.value || '0', 10); applyGrouping(n); } });\n"
        "  }\n"
        "  if (input) { input.value = '1'; applyGrouping(1); }\n"
        "  const rangeStart = document.getElementById('range-start');\n"
        "  const rangeEnd = document.getElementById('range-end');\n"
        "  const rangeBtn = document.getElementById('apply-range');\n"
        "  if (rangeBtn && rangeStart && rangeEnd) {\n"
        "    const applyRange = () => {\n"
        "      const sv = parseInt(rangeStart.value || '', 10);\n"
        "      const ev = parseInt(rangeEnd.value || '', 10);\n"
        "      currentRangeStart = Number.isFinite(sv) ? sv : null;\n"
        "      currentRangeEnd = Number.isFinite(ev) ? ev : null;\n"
        "      applyRangeFilter();\n"
        "    };\n"
        "    rangeBtn.addEventListener('click', applyRange);\n"
        "    rangeStart.addEventListener('keydown', (e) => { if (e.key === 'Enter') applyRange(); });\n"
        "    rangeEnd.addEventListener('keydown', (e) => { if (e.key === 'Enter') applyRange(); });\n"
        "  }\n"
        "  const strongHideBtn = document.getElementById('toggle-strong-hide');\n"
        "  const strongHideStateEl = document.getElementById('strong-hide-state');\n"
        "  if (strongHideBtn) {\n"
        "    const setLabel = () => { if (strongHideStateEl) { strongHideStateEl.textContent = strongHideOn ? 'On' : 'Off'; } };\n"
        "    strongHideBtn.addEventListener('click', () => { strongHideOn = !strongHideOn; for (const f of activeFlows()) { f.classList.toggle('strong-hide', strongHideOn); } setLabel(); });\n"
        "    if (strongHideOn) { for (const f of activeFlows()) { f.classList.add('strong-hide'); } }\n"
        "    setLabel();\n"
        "  }\n"
        "  const splitBtn = document.getElementById('toggle-split-view');\n"
        "  const splitStateEl = document.getElementById('split-view-state');\n"
        "  if (splitBtn && splitFlow && linearFlow) {\n"
        "    const updateSplit = () => { if (splitStateEl) splitStateEl.textContent = splitViewOn ? 'On' : 'Off'; };\n"
        "    splitBtn.addEventListener('click', () => { splitViewOn = !splitViewOn; linearFlow.style.display = splitViewOn ? 'none' : ''; splitFlow.style.display = splitViewOn ? '' : 'none'; applyGrouping(parseInt(input.value||'1',10)); updateSplit(); });\n"
        "    updateSplit();\n"
        "  }\n"
        "});\n"
        "</script>",
        "</head>",
        "<body>",
        '<div class="toolbar-wrap">',
        '<div class="toolbar-hotzone"></div>',
        '<div class="toolbar">',
        '<label for="group-size">Group every</label>',
        '<input id="group-size" type="number" min="0" step="1" value="1" />',
        "<span>timesteps</span>",
        '<button id="apply-grouping">Apply</button>',
        '<span style="margin-left:8px"></span>',
        '<label for="range-start"><span class="emoji-bw">🔎</span> Range</label>',
        '<input id="range-start" type="number" step="1" />',
        "<span>to</span>",
        '<input id="range-end" type="number" step="1" />',
        '<button id="apply-range"><span class="emoji-bw">▶︎</span> Apply</button>',
        '<button id="toggle-strong-hide"><span class="emoji-bw">🗜️</span> Strong Hide: <span id="strong-hide-state">Off</span></button>',
        (
            '<button id="toggle-split-view"><span class="emoji-bw">🪟</span> Split View: <span id="split-view-state">Off</span></button>'
            if enable_split_view
            else ""
        ),
        "</div>",
        "</div>",
        '<div id="flow-linear" class="messages-flow">',
    ]

    last_time_step = None
    for original_index, turn in indexed_turns:
        # Build classes
        agent_class = f"agent-{re.sub('[^a-z0-9_-]', '-', turn.agent_id.lower())}"
        role_class = f"role-{turn.role}"
        collapsed_class = " collapsed" if turn.role == "user" else ""

        # Badge content
        if turn.role == "assistant":
            name = html.escape(turn.agent_id)
            emoji = '<span class="emoji-bw">  🤖</span>'
            raw_val = turn.reward
            if isinstance(raw_val, (int, float)):
                reward_val = f"{raw_val:.4f}".rstrip("0").rstrip(".")
                if len(reward_val) > 8:
                    reward_val = reward_val[:8] + "…"
            else:
                reward_val = str(raw_val)
            # Format: "🤖 Alice • Reward: 5.5556 • 💬 :"
            badge_inner = (
                f'{emoji} <span class="agent-name">{name}</span>'
                f' <span class="sep"> • </span><span class="reward">Reward: {reward_val}</span>'
                f' <span class="sep"> • </span>💬  '
            )
        else:
            # For user messages, show "User of {Agent ID}" in the badge
            name = "User of " + html.escape(turn.agent_id)
            emoji = '<span class="emoji-bw">⚙️</span>'
            # Format (no reward): "⚙️ User of Alice • "
            badge_inner = f'{emoji} <span class="agent-name">{name}</span> <span class="sep"> • </span>:'

        badge = f'<span class="agent-badge">{badge_inner}</span>'

        # Inline timestep distinction badge at step boundaries (render before first message)
        ts_badge_html = ""
        if last_time_step is None or turn.time_step != last_time_step:
            ts_badge_html = f'<span class="ts-badge">⏱ {turn.time_step}</span>'
            last_time_step = turn.time_step

        escaped_content = html.escape(turn.content)
        collapsed_text = re.sub(r"\s+", " ", escaped_content).strip()

        html_parts.append(
            f'<div class="chat-turn {agent_class} {role_class}{collapsed_class}" data-time-step="{turn.time_step}">'
            f'<div class="turn-content {agent_class} {role_class}">{ts_badge_html}{badge}'
            f'<span class="message-box">{collapsed_text}</span>'
            f'<span class="message-placeholder">(...)</span>'
            f"</div>"
            f"</div>"
        )

    html_parts.append("</div>")  # close linear flow
    if enable_split_view:
        import html as _html_mod

        html_parts.append(
            '<div id="flow-split" class="messages-flow" style="display:none">'
        )
        html_parts.append('<div class="split-wrapper">')
        # Per-agent columns
        per_agent_turns = {
            aid: [t for t in chat_turns if t.agent_id == aid]
            for aid in assistant_agents
        }
        for idx, aid in enumerate(assistant_agents):
            turns_agent = per_agent_turns[aid]
            html_parts.append(
                f'<div class="split-col" data-agent="{_html_mod.escape(aid)}">'
            )
            last_ts_agent = None
            for turn in turns_agent:
                agent_class = (
                    f"agent-{re.sub('[^a-z0-9_-]', '-', turn.agent_id.lower())}"
                )
                role_class = f"role-{turn.role}"
                collapsed_class = " collapsed" if turn.role == "user" else ""
                ts_badge_html = ""
                if last_ts_agent is None or turn.time_step != last_ts_agent:
                    ts_badge_html = f'<span class="ts-badge">⏱ {turn.time_step}</span>'
                    last_ts_agent = turn.time_step
                esc_content = _html_mod.escape(turn.content)
                collapsed_text = re.sub(r"\s+", " ", esc_content).strip()
                if turn.role == "assistant":
                    name = _html_mod.escape(turn.agent_id)
                    emoji = '<span class="emoji-bw">🤖</span>'
                    raw_val = turn.reward
                    if isinstance(raw_val, (int, float)):
                        reward_val = f"{raw_val:.4f}".rstrip("0").rstrip(".")
                        if len(reward_val) > 8:
                            reward_val = reward_val[:8] + "…"
                    else:
                        reward_val = str(raw_val)
                    badge_inner = (
                        f'{emoji} <span class="agent-name">{name}</span>'
                        f' <span class="sep"> • </span><span class="reward"> ⚑ {reward_val}</span>'
                        f' <span class="sep"> • </span>💬 : '
                    )
                else:
                    name = "User of " + _html_mod.escape(turn.agent_id)
                    emoji = '<span class="emoji-bw">⚙️</span>'
                    badge_inner = f'{emoji} <span class="agent-name">{name}</span> <span class="sep"> • </span>:'
                badge = f'<span class="agent-badge">{badge_inner}</span>'
                html_parts.append(
                    f'<div class="chat-turn {agent_class} {role_class}{collapsed_class}" data-time-step="{turn.time_step}">'
                    f'<div class="turn-content {agent_class} {role_class}">{ts_badge_html}{badge}'
                    f'<span class="message-box">{collapsed_text}</span>'
                    f'<span class="message-placeholder">(...)</span>'
                    f"</div></div>"
                )
            html_parts.append("</div>")  # close split col
        html_parts.append("</div>")  # split-wrapper
        html_parts.append("</div>")  # flow-split
    html_parts.extend(["</body>", "</html>"])

    return "\n".join(html_parts)


def export_html_from_rollout_tree(path: Path, outdir: Path, main_only: bool = False):
    """Process a rollout tree file and generate HTML files for each path.
    Creates separate HTML files for the main path and each branch path.
    The main path is saved in the root output directory, while branch paths
    are saved in a 'branches' subdirectory.

    Args:
        path: Path to the rollout tree JSON file
        outdir: Output directory for HTML files
        main_only: If True, only export the main trajectory (default: False)
    """
    root = load_rollout_tree(path)
    mgid = root.id

    main_path, branch_paths = get_rollout_tree_paths(root)

    outdir.mkdir(parents=True, exist_ok=True)

    # Create branches subdirectory if we have branch paths
    if not main_only and branch_paths:
        branches_dir = outdir / f"mgid:{mgid}_branches_html_renders"
        branches_dir.mkdir(parents=True, exist_ok=True)

    # Generate HTML for the main path
    chat_turns = gather_all_chat_turns_for_path(main_path)
    html_content = html_from_chat_turns(chat_turns)
    output_file = outdir / f"mgid:{mgid}_main_html_render.render.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Generate HTML for each branch path
    for path_obj in branch_paths:
        chat_turns = gather_all_chat_turns_for_path(path_obj)

        html_content = html_from_chat_turns(chat_turns)

        path_id: str = path_obj.id
        output_filename = f"{path_id}_html_render.render.html"

        output_file = branches_dir / output_filename

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
