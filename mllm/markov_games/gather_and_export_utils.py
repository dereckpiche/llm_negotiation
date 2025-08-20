from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from mllm.markov_games.rollout_tree import *

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
    """Gather all rollout trees from the iteration folder."""
    rollout_trees = []
    iteration_path = Path(iteration_folder)
    for item in iteration_path.glob("**/*.json"):
        if item.is_file() and re.match(
            r"(rollout_tree\.json|mgid:\d+_rollout_tree\.json)$", item.name
        ):
            rollout_tree = RolloutTreeRootNode.model_validate_json(item.read_text())
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
    """Load a rollout tree from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
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

    # Collect all turns from all agents in all path nodes
    for node in path.nodes:
        for agent_id, action_log in node.step_log.action_logs.items():
            if action_log.chat_turns:
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
                    turns.append(turn_log)
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
    """Process a rollout tree file and a generate a JSONL.
    Each json object in the JSONL should be a list of chat turns for a single path.
    Each json object should be identified by the path id.
    """
    # Load the rollout tree
    root = load_rollout_tree(path)
    mgid = root.id

    # Get all paths
    main_path, branch_paths = get_rollout_tree_paths(root)
    all_paths = [main_path] + branch_paths

    # Create output directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate output filename based on input filename
    output_file = outdir / f"mgid:{mgid}_plucked_chats.jsonl"

    # Export chat logs for each path
    with open(output_file, "w", encoding="utf-8") as f:
        for path_obj in all_paths:
            chat_turns = gather_all_chat_turns_for_path(path_obj)

            # Create output object with path id and chat turns
            output_obj = {
                "path_id": str(path_obj.id),
                "chat_turns": list(chat_turns_to_dict(iter(chat_turns))),
            }

            # Write as JSON line
            f.write(json.dumps(output_obj, indent=2) + "\n")


# --------------------------------------------------------------------------------------
# HTML exports
# --------------------------------------------------------------------------------------


def html_from_chat_turns(chat_turns: List[ChatTurnLog]) -> str:
    """
    Render a list of chat turns as an HTML file.
    Each time step is in a separate div.
    The chat turns of each each agents of the same time step are side by side.
    A visual separator is added between each time step.
    Visual style helps distinguish between roles of each time step.
    """
    import html
    from collections import defaultdict

    # Group chat turns by time step
    turns_by_time_step = defaultdict(list)
    for turn in chat_turns:
        turns_by_time_step[turn.time_step].append(turn)

    # CSS styles - minimal, dense, dedestyle-inspired
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 16px;
            background-color: #ffffff; /* figure.facecolor */
            color: #1c0b00; /* text.color */
            font-size: 14px; /* font.size */
        }
        .time-step {
            margin: 8px 0;
            padding: 4px 0 8px 0;
            border-top: 2px solid #ECF0F1; /* grid.color */
            display: flex;
            flex-wrap: wrap; /* allow footer to wrap to next line */
            align-items: stretch;
            gap: 8px;
        }
        /* Alternating background for time steps (applied via class) */
        .time-step.alt {
            background: #FAFBFC;
        }
        .time-step-index {
            width: 120px;
            flex: 0 0 120px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            color: #2C3E50;
            font-weight: 600;
            gap: 6px;
            padding-top: 2px;
        }
        .index-rewards { display: flex; flex-direction: column; align-items: center; gap: 2px; }
        .index-rewards-row { display: flex; flex-direction: column; gap: 4px; align-items: center; }
        .agents-container-left,
        .agents-container-right {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            flex: 1 1 0;
            min-width: 0;
        }
        .agent-column {
            flex: 1;
            min-width: 260px;
            border: 1px solid #BDC3C7; /* axes.edgecolor */
            border-radius: 4px;
            background: #ffffff; /* axes.facecolor */
        }
        .chat-turn {
            padding: 6px 8px;
            margin: 4px;
            border-radius: 3px;
            border: 1px solid #ECF0F1; /* subtle outer separator */
            background: #ffffff;
        }
        /* Message box that contains the content */
        .message-box {
            display: inline-block;
            padding: 4px 8px;
            margin-left: 6px;
            border: 2px solid #BDC3C7; /* neutral by default */
            border-radius: 14px; /* iMessage-like rounded */
            position: relative; /* for bubble tails */
            vertical-align: top;
            max-width: calc(100% - 140px);
            background: #ffffff; /* keep minimal and readable */
        }
        /* Rectangular corner by role (user: top-left square, assistant: top-right square) */
        .chat-turn.role-user .message-box { border-radius: 0 14px 14px 14px; }
        .chat-turn.role-assistant .message-box { border-radius: 14px 0 14px 14px; }

        /* No pseudo-element tails; rectangular corner is the cue */
        .chat-turn.role-user .message-box::before,
        .chat-turn.role-user .message-box::after,
        .chat-turn.role-assistant .message-box::before,
        .chat-turn.role-assistant .message-box::after { content: none; }
        /* Color only assistant messages per agent (dedestyle palette: green/orange) */
        .agent-alice .chat-turn.role-assistant .message-box { border-color: #0eb224; }
        .agent-bob .chat-turn.role-assistant .message-box { border-color: #ef8323; }
        .turn-meta {
            font-size: 12px; /* compact meta */
            color: #2C3E50; /* readable meta */
            margin-bottom: 4px;
        }
        .turn-content {
            white-space: normal; /* collapse newlines */
            line-height: 1.35; /* dense yet readable */
            color: #1c0b00;
            font-size: 14px;
            display: flex;
            align-items: flex-start;
        }
        /* Role-based alignment inside each agent column */
        .chat-turn.role-user .turn-content { justify-content: flex-start; flex-direction: row; }
        .chat-turn.role-assistant .turn-content { justify-content: flex-end; flex-direction: row; }
        .chat-turn.role-assistant .turn-content::before { content: ""; flex: 1 1 auto; }
        .chat-turn.role-user .message-box { margin-left: 6px; margin-right: 0; }
        .chat-turn.role-assistant .message-box { margin-right: 6px; margin-left: 0; }
        .chat-turn.role-assistant .agent-badge { margin-left: 0; margin-right: 0; }
        .chat-turn.role-user .agent-badge { margin-right: 6px; }
        .agent-name { font-weight: 700; color: #2C3E50; }
        .agent-badge {
            display: inline-block;
            border: 1px solid #BDC3C7;
            border-radius: 999px; /* pill */
            padding: 2px 8px;
            font-size: 12px;
            line-height: 1.4;
            margin-right: 6px;
            color: #2C3E50;
            background: #F8FAFC;
            box-shadow: 0 1px 0 rgba(0,0,0,0.03) inset;
        }
        .emoji-bw { filter: grayscale(100%); opacity: 0.95; font-size: 14px; vertical-align: text-bottom; margin: 0 2px; }
        .inline-sep {
            display: inline-block;
            vertical-align: baseline;
            margin: 0 6px;
            width: 1px;
            height: 0.9em;
            background: #ECF0F1; /* subtle divider using grid color */
        }
        /* no message emoji */
        .state-end-marker {
            border: 1px solid #BDC3C7;
            color: #2C3E50;
            background: transparent;
            padding: 1px 4px;
            border-radius: 2px;
            font-size: 11px;
            font-weight: 600;
        }
        /* Time step footer for rewards */
        .time-step-meta {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 8px;
            padding: 4px 8px 0 8px;
            margin: 6px 0 0 0;
            border-top: 1px dashed #ECF0F1;
            flex-basis: 100%;
        }
        .reward-label { color: #2C3E50; font-size: 12px; font-weight: 700; }
        .reward-pill {
            border: 1px solid #BDC3C7;
            border-radius: 12px;
            padding: 1px 6px;
            font-size: 11px;
            background: #ffffff;
        }
        .reward-value { color: #B8860B; font-weight: 700; }
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
        "</head>",
        "<body>",
    ]

    # Process each time step
    for time_step in sorted(turns_by_time_step.keys()):
        turns = turns_by_time_step[time_step]

        # Group turns by agent for this time step
        turns_by_agent = defaultdict(list)
        for turn in turns:
            turns_by_agent[turn.agent_id].append(turn)

        # Time step container with centered index
        time_step_class = "time-step alt" if (time_step % 2 == 1) else "time-step"
        html_parts.append(f'<div class="{time_step_class}">')
        html_parts.append('<div class="agents-container-left">')
        html_parts_left = []
        html_parts_right = []
        # Prepare middle index to be inserted later (with rewards underneath)
        # Per-time-step rewards
        rewards_by_agent = {}
        for turn in turns:
            if turn.agent_id not in rewards_by_agent:
                rewards_by_agent[turn.agent_id] = turn.reward
        reward_pills = []
        for aid in sorted(rewards_by_agent.keys()):
            raw_val = rewards_by_agent[aid]
            # Format reward: cap long decimals with ellipsis
            formatted = (
                f"{raw_val:.4f}" if isinstance(raw_val, (int, float)) else str(raw_val)
            )
            if isinstance(raw_val, float):
                # Remove trailing zeros and dot
                formatted = formatted.rstrip('0').rstrip('.')
                if len(formatted) > 8:
                    formatted = formatted[:8] + '…'
            reward_pills.append(
                f'<span class="reward-pill">{html.escape(aid)}: <span class="reward-value">{formatted}</span></span>'
            )
        middle_index_html = (
            f'<div class="time-step-index">'
            f'<div>⏱ {time_step}</div>'
            f'<div class="index-rewards">'
            f'  <div class="reward-label">Rewards</div>'
            f'  <div class="index-rewards-row">' + "".join(reward_pills) + '</div>'
            f'</div>'
            f'</div>'
        )

        # Process each agent; split left/right (alice left, bob right; others alternate)
        side_toggle = True
        for agent_id in sorted(turns_by_agent.keys()):
            agent_turns = turns_by_agent[agent_id]

            # Agent-specific class for styling
            agent_class = f"agent-{re.sub('[^a-z0-9_-]', '-', agent_id.lower())}"
            agent_html = []
            agent_html.append(f'<div class="agent-column {agent_class}">')

            # Process each turn for this agent
            for turn in agent_turns:
                # Add role class to enable assistant-only coloring
                role_class = f"role-{turn.role}"
                agent_html.append(f'<div class="chat-turn {role_class}">')

                # Turn metadata inline with content: minimal separators
                # Role-based badges with emojis (user: gear; assistant: robot)
                if turn.role == "assistant":
                    name_badge = (
                        f'<span class="agent-badge agent-name">'
                        f'<span class="emoji-bw">🤖</span>{html.escape(agent_id)}'
                        f'</span>'
                    )
                else:
                    name_badge = f'<span class="agent-badge agent-name"><span class="emoji-bw">⚙️</span>user</span>'

                # Turn content (collapse all whitespace/newlines)
                escaped_content = html.escape(turn.content)
                collapsed = re.sub(r"\s+", " ", escaped_content).strip()
                # Render: role-based order inside the row
                if turn.role == "assistant":
                    # Assistant on the right: bubble then badge
                    agent_html.append(
                        f'<div class="turn-content"><span class="message-box">{collapsed}</span>{name_badge}</div>'
                    )
                else:
                    # User on the left: badge then bubble
                    agent_html.append(
                        f'<div class="turn-content">{name_badge}<span class="message-box">{collapsed}</span></div>'
                    )

                agent_html.append("</div>")  # Close chat-turn

            agent_html.append("</div>")  # Close agent-column

            agent_block = "".join(agent_html)
            agent_id_lower = agent_id.lower()
            if agent_id_lower == "alice":
                html_parts_left.append(agent_block)
            elif agent_id_lower == "bob":
                html_parts_right.append(agent_block)
            else:
                if side_toggle:
                    html_parts_left.append(agent_block)
                else:
                    html_parts_right.append(agent_block)
                side_toggle = not side_toggle

        # Render left, middle index, right
        html_parts.append("".join(html_parts_left))
        html_parts.append("</div>")  # Close agents-container-left
        html_parts.append(middle_index_html)
        html_parts.append('<div class="agents-container-right">')
        html_parts.append("".join(html_parts_right))
        html_parts.append("</div>")  # Close agents-container-right

        html_parts.append("</div>")  # Close time-step

    # Close HTML
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
    output_file = outdir / f"mgid:{mgid}_main_html_render.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Generate HTML for each branch path
    for path_obj in branch_paths:
        chat_turns = gather_all_chat_turns_for_path(path_obj)

        html_content = html_from_chat_turns(chat_turns)

        path_id: str = path_obj.id
        output_filename = f"{path_id}_html_render.html"

        output_file = branches_dir / output_filename

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
