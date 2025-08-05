from __future__ import annotations
from typing import Iterator, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import csv
import re

from mllm.markov_games.rollout_tree import *




# --------------------------------------------------------------------------------------
# Gather data from rollout tree methods
# --------------------------------------------------------------------------------------

def load_rollout_tree(path: Path) -> RolloutTreeRootNode:
    """Load a rollout tree from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return RolloutTreeRootNode.model_validate(data)

@dataclass(frozen=True)
class PathId:
    segments: Tuple[str, ...] = ()
    def extend(self, seg: str) -> "PathId":
        return PathId(self.segments + (seg,))
    def __str__(self) -> str:
        return "/".join(self.segments) if self.segments else "root"

@dataclass
class RolloutNodeList:
    id: PathId
    nodes: List[RolloutTreeNode]

def get_rollout_tree_paths(root: RolloutTreeRootNode) -> Tuple[RolloutNodeList, List[RolloutNodeList]]:
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
                return [current.main_child] + collect_path_nodes(current.main_child.child)
            else:
                return []
    
    def traverse_for_branches(current, main_path_prefix: List[RolloutTreeNode], path_id: PathId):
        """Traverse tree to collect all branch paths."""
        if current is None:
            return
            
        if isinstance(current, RolloutTreeNode):
            # Continue traversing with this node added to the main path prefix
            new_prefix = main_path_prefix + [current]
            traverse_for_branches(current.child, new_prefix, path_id.extend(str(current.time_step)))
            
        elif isinstance(current, RolloutTreeBranchNode):
            
            # Collect all branch paths
            if current.branches:
                for agent_id, branch_node_list in current.branches.items():
                    if branch_node_list:
                        # Start with the main path prefix, then recursively collect all nodes in this branch
                        branch_path_nodes = main_path_prefix.copy()
                        for branch_node in branch_node_list:
                            branch_path_nodes.extend(collect_path_nodes(branch_node))
                        
                        branch_path_id = path_id.extend(f"branch_{agent_id}")
                        branch_paths.append(RolloutNodeList(
                            id=branch_path_id,
                            nodes=branch_path_nodes
                        ))

            # Process the main child and add to prefix
            new_prefix = main_path_prefix
            if current.main_child:
                new_prefix = main_path_prefix + [current.main_child]
            
            # Continue traversing the main path
            if current.main_child:
                traverse_for_branches(current.main_child.child, new_prefix, 
                                    path_id.extend(str(current.main_child.time_step)))
    
    # Collect the main path nodes
    main_path_nodes = collect_path_nodes(root.child)
    
    # Traverse to collect all branch paths
    traverse_for_branches(root.child, [], PathId())
    
    # Create the main path
    main_path = RolloutNodeList(
        id=PathId(),
        nodes=main_path_nodes
    )
    
    return main_path, branch_paths

class ChatTurnLog(BaseModel):
    time_step: int
    agent_id: str
    role: str
    content: str
    is_state_end: bool
    reward: float


def gather_agent_chat_turns_for_path(agent_id: str, path: RolloutNodeList) -> List[ChatTurnLog]:
    """Iterate through all chat turns for a specific agent in a path sorted by time step."""
    turns = []
    for node in path.nodes:
        action_log = node.step_log.action_logs.get(agent_id, [])
        if action_log:
            for chat_turn in action_log.chat_turns or []:
                turns.append(ChatTurnLog(
                    time_step=node.time_step,
                    agent_id=agent_id,
                    role=chat_turn.role,
                    content=chat_turn.content,
                    is_state_end=chat_turn.is_state_end,
                    reward=node.step_log.simulation_step_log.rewards.get(agent_id, 0),
                ))
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
                        reward=node.step_log.simulation_step_log.rewards.get(agent_id, 0),
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
    
    # Get all paths
    main_path, branch_paths = get_rollout_tree_paths(root)
    all_paths = [main_path] + branch_paths
    
    # Create output directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename based on input filename
    output_file = outdir / f"{path.stem}_chat_logs.jsonl"
    
    # Export chat logs for each path
    with open(output_file, 'w', encoding='utf-8') as f:
        for path_obj in all_paths:
            chat_turns = gather_all_chat_turns_for_path(path_obj)
            
            # Create output object with path id and chat turns
            output_obj = {
                'path_id': str(path_obj.id),
                'chat_turns': list(chat_turns_to_dict(iter(chat_turns)))
            }
            
            # Write as JSON line
            f.write(json.dumps(output_obj, indent=2) + '\n')



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
    from collections import defaultdict
    import html
    
    # Group chat turns by time step
    turns_by_time_step = defaultdict(list)
    for turn in chat_turns:
        turns_by_time_step[turn.time_step].append(turn)
    
    # CSS styles
    css = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .time-step { 
            background: white; 
            margin: 20px 0; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .time-step-header { 
            font-weight: bold; 
            color: #333; 
            border-bottom: 2px solid #ddd; 
            padding-bottom: 5px; 
            margin-bottom: 15px; 
        }
        .agents-container { 
            display: flex; 
            gap: 20px; 
            flex-wrap: wrap; 
        }
        .agent-column { 
            flex: 1; 
            min-width: 300px; 
            border: 1px solid #e0e0e0; 
            border-radius: 6px; 
            overflow: hidden;
        }
        .agent-header { 
            background: #f8f9fa; 
            padding: 10px; 
            font-weight: bold; 
            border-bottom: 1px solid #e0e0e0; 
        }
        .chat-turn { 
            padding: 10px; 
            margin: 5px; 
            border-radius: 4px; 
            border-left: 4px solid #ccc;
        }
        .role-user { 
            background-color: #e3f2fd; 
            border-left-color: #1976d2; 
        }
        .role-assistant { 
            background-color: #f3e5f5; 
            border-left-color: #7b1fa2; 
        }
        .turn-meta { 
            font-size: 0.8em; 
            color: #666; 
            margin-bottom: 5px; 
        }
        .turn-content { 
            white-space: pre-wrap; 
            line-height: 1.4; 
        }
        .state-end-marker { 
            background: #ffeb3b; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-size: 0.7em; 
            font-weight: bold; 
        }
        .reward { 
            background: #4caf50; 
            color: white; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-size: 0.7em; 
            margin-left: 5px; 
        }
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
        "<h1>Chat Conversation</h1>"
    ]
    
    # Process each time step
    for time_step in sorted(turns_by_time_step.keys()):
        turns = turns_by_time_step[time_step]
        
        # Group turns by agent for this time step
        turns_by_agent = defaultdict(list)
        for turn in turns:
            turns_by_agent[turn.agent_id].append(turn)
        
        # Time step container
        html_parts.append('<div class="time-step">')
        html_parts.append(f'<div class="time-step-header">Time Step {time_step}</div>')
        html_parts.append('<div class="agents-container">')
        
        # Process each agent
        for agent_id in sorted(turns_by_agent.keys()):
            agent_turns = turns_by_agent[agent_id]
            
            html_parts.append('<div class="agent-column">')
            html_parts.append(f'<div class="agent-header">Agent: {html.escape(agent_id)}</div>')
            
            # Process each turn for this agent
            for turn in agent_turns:
                role_class = f"role-{turn.role}" if turn.role in ["user", "assistant"] else "role-other"
                
                html_parts.append(f'<div class="chat-turn {role_class}">')
                
                # Turn metadata
                meta_parts = [f"Role: {html.escape(turn.role)}"]
                if turn.is_state_end:
                    meta_parts.append('<span class="state-end-marker">STATE END</span>')
                if turn.role == "assistant":
                    meta_parts.append(f'<span class="reward">Reward: {turn.reward}</span>')
                
                html_parts.append(f'<div class="turn-meta">{" ".join(meta_parts)}</div>')
                
                # Turn content
                escaped_content = html.escape(turn.content)
                html_parts.append(f'<div class="turn-content">{escaped_content}</div>')
                
                html_parts.append('</div>')  # Close chat-turn
            
            html_parts.append('</div>')  # Close agent-column
        
        html_parts.append('</div>')  # Close agents-container
        html_parts.append('</div>')  # Close time-step
    
    # Close HTML
    html_parts.extend(["</body>", "</html>"])
    
    return "\n".join(html_parts)


def export_html_from_rollout_tree(path: Path, outdir: Path, main_only: bool = True):
    """Process a rollout tree file and generate HTML files for each path.
    Creates separate HTML files for the main path and each branch path.
    
    Args:
        path: Path to the rollout tree JSON file
        outdir: Output directory for HTML files
        main_only: If True, only export the main trajectory (default: False)
    """
    # Load the rollout tree
    root = load_rollout_tree(path)
    
    # Get all paths
    main_path, branch_paths = get_rollout_tree_paths(root)
    all_paths = [main_path] if main_only else [main_path] + branch_paths
    
    # Create output directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML for each path
    for path_obj in all_paths:
        # Get chat turns for this path
        chat_turns = gather_all_chat_turns_for_path(path_obj)
        
        # Generate HTML content
        html_content = html_from_chat_turns(chat_turns)
        
        # Create safe filename from path ID
        path_id_str = str(path_obj.id).replace("/", "_") if str(path_obj.id) != "root" else "main_path"
        output_filename = f"{path.stem}_{path_id_str}.html"
        output_file = outdir / output_filename
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    path_type = "main trajectory only" if main_only else f"{len(all_paths)} paths (main + branches)"
    print(f"Generated {len(all_paths)} HTML files ({path_type}) in {outdir}")
    
    
