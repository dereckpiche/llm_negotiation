
def get_coagent_id(ids: List[str], agent_id:str) -> str:
    for id in ids:
        if id != agent_id: return id
