def always_matching_automaton(state):
    """
    An automaton that always submits a finalizing proposal,
    proposing it takes exactly half (floor) of each item.
    
    Returns:
        tuple: (is_finalization: bool, output: any)
    """
    # For each item, propose taking half its total quantity.
    i_take = {}
    for item in state["items"]:
        # floor division for half
        i_take[item] = state["quantities"][item] // 2

    return True, {"i_take": i_take}


def random_message_then_random_proposal(state):
    """
    An automaton that, on the first turn, sends a random (non-final) message,
    and on the next turn finalizes with a random proposal of items.
    
    Returns:
        tuple: (is_finalization: bool, output: any)
    """
    import random
    # If we haven't yet made a move this round, send a random message
    if state["turn"] == 0:
        random_message = f"Random message: {random.randint(0, 10000)}"
        return False, random_message
    else:
        # Otherwise, finalize with a random proposal
        i_take = {}
        for item in state["items"]:
            i_take[item] = random.randint(0, state["quantities"][item])
        return True, {"i_take": i_take}
