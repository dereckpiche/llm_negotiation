from __future__ import annotations
from pathlib import Path
import json, jsonschema
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Tuple, Literal
from dataclasses import dataclass
AgentId = str

class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    agent_id: AgentId # ID of the agent with which the chat occured
    content: str
    is_state_end: bool = False # indicates whether this chatturn marks the end of a state in the trajectory
    time_step: Optional[int] = None # t

class SimulationStepLog(BaseModel):
    rewards: dict[AgentId, float]
    info: Any = None

class AgentActLog(BaseModel):
    chat_turns: list[ChatTurn] | None
    info: Any = None

class StepLog(BaseModel):
    action_logs:  dict[AgentId, AgentActLog]
    simulation_step_log: SimulationStepLog


BranchType = Literal["unilateral_deviation", "common_deviation"] # might not be necessary
class BranchNodeInfo(BaseModel):
    branch_id: str
    branch_for: AgentId
    branch_type: BranchType

class RolloutTreeNode(BaseModel):
    step_log: StepLog
    time_step: int
    child: RolloutTreeNode | RolloutTreeBranchNode | None = None

class RolloutTreeBranchNode(BaseModel):
    """
    First item of the tuple indicates which agent "called" for an alternative branch.
    """
    main_child: RolloutTreeNode
    branches: dict[AgentId, list[RolloutTreeNode]] | None = None

class RolloutTreeRootNode(BaseModel):
    id: int
    child: RolloutTreeNode | RolloutTreeBranchNode | None = None

class RolloutTreeLeafNode(BaseModel):
    step_log: StepLog
    time_step: int


# Necessary for self-referential stuff in pydantic
RolloutTreeBranchNode.model_rebuild()
RolloutTreeNode.model_rebuild()
