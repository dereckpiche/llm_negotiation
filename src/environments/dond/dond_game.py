import random
from collections import deque

from utils.common_imports import *


class DondEnv:
    def __init__(
        self,
        game_id,
        random_seed,
        agents=["alice", "bob"],
        max_messages=None,
        min_messages=None,
        max_chars_per_message=None,
        rounds_per_game=1,
        random_setup_func=None,
        random_setup_kwargs=None,
        points_attribution_method=None,
        points_attributions_kwargs=None,
        role_assignator_func=None,
        role_assignator_func_kwargs=None,
        finalization_visibility=False,
        other_values_visibility=False,
        mode="basic",
        roundwise_utilities=[],
        group_id=0,
    ):
        """
        Initializes the DoND game.

        Args:
            agents (list): List of agent names.
            mode (str): The mode of the game, either 'coop' or 'basic'.
            max_messages (int): Maximum number of conversation (non-finalization) messages per agent
                                allowed before finalization is forced.
            min_messages (int): Minimum number of conversation messages required before a agent can finalize.
            max_chars_per_message (int): Maximum number of characters allowed per message.
            rounds_per_game (int): The number of rounds per game.
            random_setup_func (str or callable): The function to use for random setup.
            random_setup_kwargs (dict): Keyword arguments for the random setup function.
            role_assignator_func (str or callable): The function to use for role assignment.
            role_assignator_func_kwargs (dict): Keyword arguments for the role assignment function.
            finalization_visibility (bool): Visibility of finalization.
            other_values_visibility (bool): Visibility of other agent's values.
            random_seed (int, optional): The base seed that will be used (and incremented) for random generation.
        """

        # A game can be uniquely identified by the combination of match_id and group_id
        self.match_id = game_id

        # Minibatch / group id for which roundwise utilities are same
        self.group_id = group_id

        # TODO: Random seed should be tied to the sytem random seed.
        if random_seed is None:
            self.random_seed = random.randint(1, 10**9)
        else:
            self.random_seed = random_seed

        self.agents = agents
        self.roles = ["starting_negotiator", "responding_negotiator"]
        self.mode = mode
        self.max_messages = max_messages
        self.min_messages = min_messages
        self.max_chars_per_message = max_chars_per_message
        self.random_setup_func = (
            globals()[random_setup_func]
            if isinstance(random_setup_func, str)
            else random_setup_func
        )
        self.random_setup_kwargs = random_setup_kwargs
        if random_setup_kwargs is not None:
            self.random_setup_kwargs["random_seed"] = random_seed
        else:
            self.random_setup_kwargs = {"random_seed": random_seed}

        # In this game, players negotiate to divide items between them.
        # Each item has a quantity (like 6 books or 4 apples) that must be completely
        # distributed between players. For example, if there are 10 cookies,
        # one player might take 7 and the other 3. Each player values the items
        # differently, so what's valuable to one player might not be to the other.

        self.finalization_visibility = finalization_visibility
        self.rounds_per_game = rounds_per_game
        self.role_assignator_func = (
            globals()[role_assignator_func]
            if isinstance(role_assignator_func, str)
            else role_assignator_func
        )
        self.role_assignator_func_kwargs = role_assignator_func_kwargs or {}
        self.role_assignator_func_kwargs["seed"] = self.random_seed
        self.role_assignator_func_kwargs["first_agent"] = (
            agents[0] if self.group_id % 2 == 0 else agents[1]
        )  # TODO: make this flexible -- this is a hack, will cause problems later
        self.other_values_visibility = other_values_visibility

        # Store the points_attribution_method
        self.points_attribution_method = (
            globals()[points_attribution_method]
            if isinstance(points_attribution_method, str)
            else points_attribution_method
        )
        self.points_attributions_kwargs = points_attributions_kwargs or {}

        self.game_moves = {agent: 0 for agent in agents}
        self.round_moves = {agent: 0 for agent in agents}
        self.round_messages = {agent: 0 for agent in agents}

        # Roundwise utilities for the corresponding minibatch / group.
        self.roundwise_utilities = roundwise_utilities

        self.reset()

    def set_new_setup(self):
        """
        Sets up a new game configuration using a local (and updated) RNG.
        The random_seed is incremented to ensure that each setup is different.
        """
        if self.roundwise_utilities:
            # Reason for using `min`: round_nb should be greater than rounds_per_game - 1
            # for the game to end but round_nb is incremented only if new_round() is called
            # which also calls set_new_setup(). This gives index out of range.
            # self.items, self.quantities, role_values = self.roundwise_utilities[min(self.round_nb, self.rounds_per_game-1)]
            self.items, self.quantities, role_values = self.roundwise_utilities[
                self.round_nb
            ]

        else:
            self.random_seed += 1
            self.random_setup_kwargs[
                "random_seed"
            ] = self.random_seed  # Ensure the new seed is used

            kwargs = self.random_setup_kwargs
            self.items, self.quantities, role_values = self.random_setup_func(**kwargs)

        self.role_values = {
            self.roles[0]: role_values[0],
            self.roles[1]: role_values[1],
        }

    def step(self, actions):
        """
        Advances the game by one step.

        Args:
            actions (dict): A dictionary where keys are agent identifiers and values are actions
                           in the form of (is_finalization, processed_response, raw_response).

        Returns:
            observations (dict): A dictionary where keys are agent identifiers and values are observations.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """
        # Process the action for the current agent
        current_agent = self.get_current_agent()

        if current_agent in actions:
            action = actions[current_agent]
            is_finalization, processed_response, raw_response = action

            # Count this move for the current agent (finalization or conversation).
            self.game_moves[current_agent] += 1
            self.round_moves[current_agent] += 1

            # Only conversation messages (non-finalization) increment the message counter.
            if not is_finalization:
                self.round_messages[current_agent] += 1
                self.message_turn += 1

            # Update state flags.
            self.last_raw_response = raw_response
            self.last_processed_response = processed_response
            self.is_new_round = self.message_turn == 1
            self.is_new_game = self.round_nb == 0 and self.message_turn == 1
            self.game_over = False
            round_over = False

            # Check the minimum message requirement on a finalization attempt.
            if (
                is_finalization
                and self.round_messages[current_agent] < self.min_messages
            ):
                # Treat the finalization as a conversation message
                self.round_messages[current_agent] += 1
                self.message_turn += 1
                self.last_raw_response = raw_response
                self.last_processed_response = processed_response
                is_finalization = False

            if self.has_finalized:
                # We are in the second finalization phase.
                if not is_finalization:
                    self.points = {role: 0 for role in self.roles}
                    self.agreement_reached = False
                else:
                    self.finalize(processed_response)
                    # Use the custom points attribution method which now returns (points, valid_agreement)
                    (
                        self.points,
                        self.agreement_reached,
                    ) = self.points_attribution_method(
                        self.get_state(), **self.points_attributions_kwargs
                    )
                round_over = True
                self.round_nb += 1

            else:
                # If a agent sends a finalization, record it.
                if is_finalization:
                    self.has_finalized = True
                    self.finalize(processed_response)
                # Check if any agent has exceeded their personal maximum message limit.
                elif any(
                    count > self.max_messages for count in self.round_messages.values()
                ):
                    round_over = True
                    self.round_nb += 1

            self.role_deque.rotate(-1)

            if self.round_nb > self.rounds_per_game - 1:
                self.game_over = True

            if round_over:
                self.new_round()

        else:
            raise ValueError(f"agent {current_agent} did not provide an action.")

        # Get the updated state to return as observation
        state = self.get_state()
        # Create observation for only the current agent who needs to act.
        current_actor = self.get_current_agent()
        observations = {current_actor: state}
        done = self.game_over
        info = self.get_info()

        return observations, done, info

    def get_log_info(self):
        """
        Get additional information about the environment. This information is used to log the game.

        Returns:
            log_info (dict): Information about the environment required to log the game.
        """
        return {
            "mode": self.mode,
            "agents": self.agents,
            "finalization_visibility": self.finalization_visibility,
            "other_values_visibility": self.other_values_visibility,
            "round_agent_roles": self.round_agent_roles,
            "round_quantities": self.round_quantities,
            "round_values": self.round_values,
            "round_finalizations": self.round_finalizations,
            "round_agreements_reached": self.round_agreements_reached,
            "round_points": self.round_points,
            "game_state": self.get_state(),
            "match_id": self.match_id,
            "group_id": self.group_id,
        }

    def render(self):
        """Render the current state of the environment."""
        print(f"Current state: {self.get_state()}")

    def close(self):
        """Perform any necessary cleanup."""
        pass

    def finalize(self, finalization):
        """
        Records the finalization from the current agent.

        Args:
            finalization (dict or str): Items taken by each player for each category,
                                        or "reject" to explicitly reject the previous offer.
        """
        current_role = self.current_turn()
        current_agent = self.get_current_agent()

        # Handle explicit rejection
        if finalization == "reject_flag":
            # Mark the agreement as explicitly rejected
            self.agreement_reached = False
            self.role_props[current_role] = {"reject_flag": True}
            return

        # Regular finalization (dict of items)
        # Ensure every item is present in the finalization, defaulting to 0 if missing
        for item in self.items:
            finalization.setdefault(item, 0)
        self.role_props[current_role] = finalization

    def get_state(self):
        """
        Retrieves the current state of the game.

        scores:
            dict: The current state of the game.
        """
        state = {
            "mode": self.mode,
            "role_values": self.role_values,
            "role_props": self.role_props,
            "agent_to_role": self.agent_to_role,
            "is_new_round": self.is_new_round,
            "is_new_game": self.is_new_game,
            "game_over": self.game_over,
            "items": self.items,
            "message_count": self.message_turn,
            "max_messages": self.max_messages,
            "min_messages": self.min_messages,
            "current_agent": self.get_current_agent(),
            "round_number": self.round_nb,
            "nb_rounds": self.rounds_per_game,
            "quantities": self.quantities,
            "has_finalized": self.has_finalized,
            "last_raw_response": getattr(self, "last_raw_response", None),
            "last_processed_response": getattr(self, "last_processed_response", None),
            "last_message": getattr(
                self, "last_raw_response", None
            ),  # For backward compatibility
            "agents": self.agents,
            "finalization_visibility": self.finalization_visibility,
            "other_values_visibility": self.other_values_visibility,
            # rounds history
            "round_agent_roles": self.round_agent_roles,
            "round_quantities": self.round_quantities,
            "round_values": self.round_values,
            "round_finalizations": self.round_finalizations,
            "round_agreements_reached": self.round_agreements_reached,
            "round_points": self.round_points,
            # New tracking information added:
            "game_moves": self.game_moves,
            "round_moves": self.round_moves,
            "round_messages": self.round_messages,
            "messages_remaining": {
                agent: self.max_messages - self.round_messages.get(agent, 0)
                for agent in self.agents
            },
        }
        return state

    def get_info(self):
        return {
            "mode": self.mode,
            "agents": self.agents,
            "finalization_visibility": self.finalization_visibility,
            "other_values_visibility": self.other_values_visibility,
            "round_agent_roles": self.round_agent_roles,
            "round_quantities": self.round_quantities,
            "round_values": self.round_values,
            "round_finalizations": self.round_finalizations,
            "round_agreements_reached": self.round_agreements_reached,
            "round_points": self.round_points,
        }

    def archive_agent_states(self):
        """
        Archives the states of the agents for the current round.
        """
        # Ensure points are initialized for all roles
        if not all(role in self.points for role in self.roles):
            self.points = {role: 0 for role in self.roles}

        self.round_agent_roles.append(self.agent_to_role.copy())
        self.round_quantities.append(self.quantities)
        self.round_values.append({role: self.role_values[role] for role in self.roles})
        self.round_finalizations.append(
            {role: self.role_props[role] for role in self.roles}
        )
        self.round_agreements_reached.append(self.agreement_reached)
        self.round_points.append({role: self.points[role] for role in self.roles})

    def new_round(self):
        """
        Ends the current round and prepares for the next round.
        """
        self.archive_agent_states()
        self.has_finalized = False
        self.role_props = {role: {} for role in self.roles}
        self.points = {role: 0 for role in self.roles}  # Ensure points are reset
        self.agreement_reached = False
        self.last_raw_response = None
        self.last_processed_response = None

        # Reset the conversation message counter for the new round.
        self.message_turn = 0

        # Reset per-round move tracking for every agent.
        self.round_moves = {agent: 0 for agent in self.agents}
        self.round_messages = {agent: 0 for agent in self.agents}

        # Only set new utility values if the game is not over.
        # This prevents index out-of-range errors when accessing roundwise utilities,
        # and avoids unnecessary RNG state changes in the default case.
        # Note: new_round() must still be called when self.game_over=True correctly attribute final points,
        # but set_new_setup() should be skipped to avoid unnecessary / invalid state updates.
        if not self.game_over:
            self.set_new_setup()

        self.assign_roles()
        self.role_deque = deque(self.roles)

    def reset(self, checkpoint=None):
        """
        Resets the game to its initial state or to a checkpoint if provided.

        Args:
            checkpoint (dict, optional): A dictionary containing the checkpoint state.

        Returns:
            observation (dict): A dictionary where keys are agent identifiers and values are observations.
        """
        if checkpoint:
            self.load_checkpoint(checkpoint)
        else:
            self.has_finalized = False
            self.role_props = {role: {} for role in self.roles}
            self.points = {
                role: 0 for role in self.roles
            }  # Ensure points are initialized
            self.agreement_reached = False
            self.last_raw_response = None
            self.last_processed_response = None
            self.round_nb = 0
            # Remove the old turn counter and use message_turn for conversation messages.
            self.message_turn = 0
            self.is_new_round = True
            self.is_new_game = True
            self.game_over = False
            self.role_deque = deque(self.roles)
            self.agent_to_role = None
            self.round_agent_roles = []
            self.round_quantities = []
            self.round_values = []
            self.round_finalizations = []
            self.round_agreements_reached = []
            self.round_points = []
            self.set_new_setup()
            self.assign_roles()
            # Initialize move tracking dictionaries for a fresh game.
            self.game_moves = {agent: 0 for agent in self.agents}
            self.round_moves = {agent: 0 for agent in self.agents}
            self.round_messages = {agent: 0 for agent in self.agents}

        # Get the initial state to return as observation
        state = self.get_state()
        # Create a dictionary of observations for each agent
        current_actor = self.get_current_agent()
        initial_observations = {current_actor: state}
        return initial_observations

    def get_current_agent(self):
        """
        Get the current agent (the one who has to play next)
        """
        if (
            not hasattr(self, "role_to_agent")
            or not hasattr(self, "role_deque")
            or not self.role_deque
        ):
            return None
        return self.role_to_agent[self.role_deque[0]]

    def current_turn(self):
        """
        Determines the current role's turn.

        scores:
            str: The name of the current role.
        """
        return self.role_deque[0]

    def assign_roles(self):
        """
        Assigns roles to agents for the current round using the role_assignator_func.
        """
        self.agent_to_role = self.role_assignator_func(
            self.get_state(), **self.role_assignator_func_kwargs
        )

        # Create agent_to_role mapping
        self.role_to_agent = {role: agent for agent, role in self.agent_to_role.items()}

    def load_checkpoint(self, checkpoint):
        """
        Loads the game state from a checkpoint.

        Args:
            checkpoint (dict): A dictionary containing the checkpoint state.
        """
        self.__dict__.update(checkpoint)


def dond_random_setup(items, min_quant, max_quant, min_val, max_val, random_seed=None):
    """
    Generates items, even-numbered quantities and distinct random values for each category for both agents.

    Args:
        items (list): List of items.
        min_quant (int): Minimum quantity per item.
        max_quant (int): Maximum quantity per item.
        min_val (int): Minimum value per item.
        max_val (int): Maximum value per item.
        random_seed (int, optional): Seed for random generation.

    scores:
        tuple: (items, quantities, (val_starting_negotiator, val_responding_negotiator))
            - quantities (dict): A dictionary mapping each item to an even quantity.
            - val_starting_negotiator (dict): Mapping for the starting negotiator with distinct values per item.
            - val_responding_negotiator (dict): Mapping for the responding negotiator with distinct values per item.
    """
    import numpy as np

    rng = np.random.default_rng(random_seed)

    # Determine the possible even numbers in the given range.
    start = min_quant if min_quant % 2 == 0 else min_quant + 1
    end = max_quant if max_quant % 2 == 0 else max_quant - 1
    if start > end:
        raise ValueError("No even numbers available in the given quantity range.")
    even_numbers = np.arange(start, end + 1, 2)

    # Generate quantities: for each item, randomly choose an even number.
    quantities = {item: int(rng.choice(even_numbers)) for item in items}

    # Make sure there are enough distinct values available for each agent's assignment.
    available_values = np.arange(min_val, max_val + 1)
    if len(available_values) < len(items):
        raise ValueError(
            "Range of values is not sufficient to assign unique values for all items."
        )

    # For each agent, randomly assign a distinct value to each item.
    val_starting_negotiator = dict(
        zip(items, rng.choice(available_values, size=len(items), replace=False))
    )
    val_responding_negotiator = dict(
        zip(items, rng.choice(available_values, size=len(items), replace=False))
    )

    return items, quantities, (val_starting_negotiator, val_responding_negotiator)


def independent_random_vals(
    items, min_quant, max_quant, min_val, max_val, random_seed=None
):
    rng = np.random.default_rng(random_seed)
    quantities = {item: int(rng.integers(min_quant, max_quant + 1)) for item in items}
    val_starting_negotiator = {
        item: int(rng.integers(min_val, max_val + 1)) for item in items
    }
    val_responding_negotiator = {
        item: int(rng.integers(min_val, max_val + 1)) for item in items
    }
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)


def fixed_manual(
    items,
    quantities,
    val_starting_negotiator,
    val_responding_negotiator,
    random_seed=None,
):
    quantities = {item: q for item, q in zip(items, quantities)}
    val_starting_negotiator = {
        item: v for item, v in zip(items, val_starting_negotiator)
    }
    val_responding_negotiator = {
        item: v for item, v in zip(items, val_responding_negotiator)
    }
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)


def random_quant_fixed_vals(
    items,
    min_quant,
    max_quant,
    val_starting_negotiator,
    val_responding_negotiator,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)
    quantities = {item: int(rng.integers(min_quant, max_quant + 1)) for item in items}
    val_starting_negotiator = {
        item: v for item, v in zip(items, val_starting_negotiator)
    }
    val_responding_negotiator = {
        item: v for item, v in zip(items, val_responding_negotiator)
    }
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)


def bicameral_vals_assignator(
    items,
    min_quant,
    max_quant,
    low_val_mean,
    low_val_std,
    high_val_mean,
    high_val_std,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)
    quantities = {item: int(rng.integers(min_quant, max_quant + 1)) for item in items}

    bernoullis = np.random.binomial(1, 0.5, len(items))
    random_left = np.random.normal(low_val_mean, low_val_std, len(items))
    random_right = np.random.normal(high_val_mean, high_val_std, len(items))

    vals_0 = np.ceil(
        np.abs(bernoullis * random_left + (1 - bernoullis) * random_right) + 0.001
    )
    vals_1 = np.ceil(
        np.abs((1 - bernoullis) * random_left + (bernoullis) * random_right) + 0.001
    )
    val_starting_negotiator = {item: v for item, v in zip(items, vals_0)}
    val_responding_negotiator = {item: v for item, v in zip(items, vals_1)}
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)


def alternating_role_assignator(state, **kwargs):
    """
    Alternates roles between agent_0 and agent_1 at each round.
    At the first round, agent_0 is assigned to the role "starting_negotiator".

    Args:
        state (dict): The current state of the game.
        kwargs (dict): Additional keyword arguments (not used here).

    scores:
        dict: A mapping of agents to roles.
    """
    round_number = state["round_number"]
    agents = state["agents"]
    roles = ["starting_negotiator", "responding_negotiator"]
    seed = kwargs["seed"]
    first_agent = kwargs["first_agent"]

    if first_agent == None:
        rng = np.random.default_rng(seed=seed)
        random_number = rng.random()
        if random_number > 0.5:
            first_agent = agents[0]
            second_agent = agents[1]
        else:
            first_agent = agents[1]
            second_agent = agents[0]
    else:
        for agent in agents:
            if agent != first_agent:
                second_agent = agent

    if round_number % 2 == 0:
        # Even rounds: agent_0 is "starting_negotiator"
        agent_to_role = {first_agent: roles[0], second_agent: roles[1]}
    else:
        # Odd rounds: agent_1 is "starting_negotiator"
        agent_to_role = {first_agent: roles[1], second_agent: roles[0]}

    return agent_to_role


def fixed_role_assignator(state, **kwargs):
    """
    Always assigns agent_0 to the role "starting_negotiator".

    Args:
        state (dict): The current state of the game.
        kwargs (dict): Additional keyword arguments (not used here).

    scores:
        dict: A mapping of agents to roles.
    """
    agents = state["agents"]
    roles = ["starting_negotiator", "responding_negotiator"]

    # Always assign agent_0 to "starting_negotiator"
    agent_to_role = {agents[0]: roles[0], agents[1]: roles[1]}

    return agent_to_role


def regular_set_points(state, **kwargs):
    """
    Sets the points for each role based on their finalizations.
    This is the default/regular points attribution method.

    Args:
        state (dict): The current state of the game.
        kwargs (dict): Additional keyword arguments (not used here).

    Returns:
        tuple: (dict, bool) - A mapping of roles to points and a boolean indicating if the agreement is valid.
    """
    roles = state["agent_to_role"].values()
    items = state["items"]
    role_values = state["role_values"]
    role_props = state["role_props"]
    mode = state["mode"]
    quantities = state["quantities"]

    # Check if any role has explicitly rejected (with reject_flag flag)
    for role in roles:
        if role_props.get(role, {}).get("reject_flag", False):
            # Agreement rejected, everyone gets 0 points
            return {role: 0 for role in roles}, False

    # Verify if finalizations match the total quantities
    valid_agreement = True
    for item in items:
        total = sum(role_props[role].get(item, 0) for role in roles)
        if total != quantities[item]:
            valid_agreement = False
            break

    # Calculate utility for each role
    utilities = {
        role: sum(
            role_values[role][item] * role_props[role].get(item, 0) for item in items
        )
        for role in roles
    }

    # Assign points based on game mode
    if mode == "coop":
        total = sum(utilities.values())
        return {role: total for role in roles}, valid_agreement
    elif mode == "basic":
        return {role: utilities[role] for role in roles}, valid_agreement
    else:
        # Default to individual utilities if mode is not recognized
        return {role: utilities[role] for role in roles}, valid_agreement


def negotiation_payoff(state, use_max_divisor=True):
    """
    Implements the payoff formula r_a = ∑ (p_a * q_a * v_a) / max(q, p_a + p_o) from https://arxiv.org/pdf/2406.14662

    Where:
    - r_a is the reward for agent a
    - p_a is the proposal (quantity taken) by agent a
    - v_a is the value agent a places on each item
    - q_a is the total quantity of the item
    - p_o is the proposal (quantity taken) by the opponent agent
    - q is the total quantity of the item

    Args:
        state (dict): The current state of the game.
        kwargs (dict): Additional keyword arguments.
            - min_divisor (int, optional): The minimum divisor value (default is 5)
            - use_max_divisor (bool, optional): If True, use max(min_divisor, p_a + p_o),
              otherwise use the total quantity (default is False)

    Returns:
        tuple: (dict, bool) - A mapping of roles to points and a boolean indicating if the agreement is valid.
    """
    roles = list(state["agent_to_role"].values())
    items = state["items"]
    role_values = state["role_values"]
    role_props = state["role_props"]
    quantities = state["quantities"]

    # Check if any role has explicitly rejected (with reject_flag flag)
    for role in roles:
        if role_props.get(role, {}).get("reject_flag", False):
            # Agreement rejected, everyone gets 0 points
            return {role: 0 for role in roles}, False

    # Verify if finalizations match the total quantities
    valid_agreement = True
    for item in items:
        total = sum(role_props[role].get(item, 0) for role in roles)
        if total != quantities[item]:
            valid_agreement = False
            break

    points = {}
    for i, role in enumerate(roles):
        opponent_role = roles[1 - i]

        total_points = 0

        for item in items:
            p_a = role_props[role].get(item, 0)
            v_a = role_values[role].get(item, 0)
            q_a = quantities.get(item, 0)

            p_o = role_props[opponent_role].get(item, 0)

            if use_max_divisor:
                divisor = max(q_a, p_a + p_o)
            else:
                divisor = p_a + p_o

            item_points = (p_a * q_a * v_a) / divisor if divisor > 0 else 0

            total_points += item_points

        points[role] = total_points

    return points, valid_agreement
