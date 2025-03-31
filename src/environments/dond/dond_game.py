import random
from utils.common_imports import *
from collections import deque


class DondEnv:
    def __init__(
        self,
        agents,
        mode="coop",
        max_messages=None,
        min_messages=None,
        max_chars_per_message=None,
        rounds_per_game=1,
        random_setup_func=None,
        random_setup_kwargs=None,
        role_assignator_func=None,
        role_assignator_func_kwargs=None,
        finalization_visibility=False,
        other_values_visibility=False,
        random_seed=None
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

        self.agents = agents
        self.roles = ["starting_negotiator", "responding_negotiator"]
        self.mode = mode
        self.max_messages = max_messages
        self.min_messages = min_messages
        self.max_chars_per_message = max_chars_per_message
        self.random_setup_func = (
            globals()[random_setup_func] if isinstance(random_setup_func, str) else random_setup_func
        )
        self.random_setup_kwargs = random_setup_kwargs
        if random_setup_kwargs is not None:
            self.random_setup_kwargs["random_seed"] = random_seed
        else:
            self.random_setup_kwargs = {"random_seed": random_seed}

        self.finalization_visibility = finalization_visibility
        self.rounds_per_game = rounds_per_game
        self.role_assignator_func = (
            globals()[role_assignator_func] if isinstance(role_assignator_func, str) else role_assignator_func
        )
        self.role_assignator_func_kwargs = role_assignator_func_kwargs or {}
        self.other_values_visibility = other_values_visibility

        if random_seed is None:
            self.random_seed = random.randint(1, 10**9)
        else:
            self.random_seed = random_seed

        self.game_moves = {agent: 0 for agent in agents}
        self.round_moves = {agent: 0 for agent in agents}
        self.round_messages = {agent: 0 for agent in agents}

        self.reset()

    def set_new_setup(self):
        """
        Sets up a new game configuration using a local (and updated) RNG.
        The random_seed is incremented to ensure that each setup is different.
        """
        self.random_seed += 1
        self.random_setup_kwargs["random_seed"] = self.random_seed  # Ensure the new seed is used

        kwargs = self.random_setup_kwargs
        self.items, self.quantities, role_values = self.random_setup_func(**kwargs)
        self.role_values = {
            self.roles[0]: role_values[0],
            self.roles[1]: role_values[1]
        }


    def step(self, actions):
        """
        Advances the game by one step.

        Args:
            actions (dict): A dictionary where keys are agent identifiers and values are actions
                           in the form of (is_finalization, output).

        Returns:
            observations (dict): A dictionary where keys are agent identifiers and values are observations.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """
        # Process the action for the current agent
        current_agent = self.get_current_agent()

        if current_agent in actions:

            action = actions[current_agent]
            is_finalization, output = action

            # Count this move for the current agent (finalization or conversation).
            self.game_moves[current_agent] += 1
            self.round_moves[current_agent] += 1

            # Only conversation messages (non-finalization) increment the message counter.
            if not is_finalization:
                self.round_messages[current_agent] += 1
                self.message_turn += 1

            # Update state flags.
            self.last_message = output
            self.is_new_round = (self.message_turn == 1)
            self.is_new_game = (self.round_nb == 0 and self.message_turn == 1)
            self.game_over = False
            round_over = False

            # Check the minimum message requirement on a finalization attempt.
            if is_finalization and self.round_messages[current_agent] < self.min_messages:
                # Treat the finalization as a conversation message
                self.round_messages[current_agent] += 1
                self.message_turn += 1
                self.last_message = output
                is_finalization = False

            if self.has_finalized:
                # We are in the second finalization phase.
                if not is_finalization:
                    self.points = {agent: 0 for agent in self.agents}
                    self.agreement_reached = False
                else:
                    self.finalize(output)
                    if self.verify_finalizations_match():
                        self.set_points()
                        self.agreement_reached = True
                    else:
                        self.points = {agent: 0 for agent in self.agents}
                        self.agreement_reached = False
                round_over = True

            else:
                # If a agent sends a finalization, record it.
                if is_finalization:
                    self.has_finalized = True
                    self.finalize(output)
                # Check if any agent has exceeded their personal maximum message limit.
                elif any(count > self.max_messages for count in self.round_messages.values()):
                    round_over = True

            self.role_deque.rotate(-1)
            if round_over:
                self.new_round()
            if self.round_nb > self.rounds_per_game - 1:
                self.game_over = True

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
        }

    def render(self):
        """Render the current state of the environment."""
        print(f"Current state: {self.get_state()}")

    def close(self):
        """Perform any necessary cleanup."""
        pass

    def verify_finalizations_match(self):
        """
        Verifies if the finalizations from both agents match the total quantities.

        scores:
            bool: True if the finalizations match, False otherwise.
        """
        for item in self.items:
            total = sum(self.role_props[role][item] for role in self.roles)
            if total != self.quantities[item]:
                return False
        return True

    def set_points(self):
        """
        Sets the points for each role based on their finalizations.
        """
        utilities = {
            role: sum(self.role_values[role][item] * self.role_props[role][item] for item in self.items)
            for role in self.roles
        }

        if self.mode == "coop":
            total = sum(utilities.values())
            self.points = {role: total for role in self.roles}

        elif self.mode == "basic":
            self.points = {role: utilities[role] for role in self.roles}

    def finalize(self, finalization: list):
        """
        Records the finalization from the current agent.

        Args:
            finalization (list): The list of finalized quantities for each item.
        """
        current_role = self.current_turn()
        finalization_dict = finalization["i_take"]
        # Ensure every item is present in the finalization, defaulting to 0 if missing
        for item in self.items:
            finalization_dict.setdefault(item, 0)
        self.role_props[current_role] = finalization_dict

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
            "last_message": self.last_message,
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
            "agents" : self.agents,
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
        self.round_finalizations.append({role: self.role_props[role] for role in self.roles})
        self.round_agreements_reached.append(self.agreement_reached)
        self.round_points.append({role: self.points[role] for role in self.roles})

    def new_round(self):
        """
        Ends the current round and prepares for the next round.
        """
        self.archive_agent_states()
        self.round_nb += 1
        self.has_finalized = False
        self.role_props = {role: {} for role in self.roles}
        self.points = {role: 0 for role in self.roles}  # Ensure points are reset
        self.agreement_reached = False
        self.last_message = None
        # Reset the conversation message counter for the new round.
        self.message_turn = 0
        # Reset per-round move tracking for every agent.
        self.round_moves = {agent: 0 for agent in self.agents}
        self.round_messages = {agent: 0 for agent in self.agents}
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
            self.points = {role: 0 for role in self.roles}  # Ensure points are initialized
            self.agreement_reached = False
            self.last_message = None
            self.round_nb = 0
            # Remove the old turn counter and use message_turn for conversation messages.
            self.message_turn = 0
            self.is_new_round = True
            self.is_new_game = True
            self.game_over = False
            self.last_message = None
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
        if not hasattr(self, 'role_to_agent') or not hasattr(self, 'role_deque') or not self.role_deque:
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
        self.agent_to_role = self.role_assignator_func(self.get_state(), **self.role_assignator_func_kwargs)

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
        raise ValueError("Range of values is not sufficient to assign unique values for all items.")

    # For each agent, randomly assign a distinct value to each item.
    val_starting_negotiator = dict(zip(items, rng.choice(available_values, size=len(items), replace=False)))
    val_responding_negotiator = dict(zip(items, rng.choice(available_values, size=len(items), replace=False)))

    return items, quantities, (val_starting_negotiator, val_responding_negotiator)

def independent_random_vals(items, min_quant, max_quant, min_val, max_val, random_seed=None):
    rng = np.random.default_rng(random_seed)
    quantities = {item: int(rng.integers(min_quant, max_quant + 1)) for item in items}
    val_starting_negotiator = {item: int(rng.integers(min_val, max_val + 1)) for item in items}
    val_responding_negotiator = {item: int(rng.integers(min_val, max_val + 1)) for item in items}
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)

def fixed_manual(items, quantities, val_starting_negotiator, val_responding_negotiator, random_seed=None):
    quantities = {item: q for item, q in zip(items, quantities)}
    val_starting_negotiator = {item: v for item, v in zip(items, val_starting_negotiator)}
    val_responding_negotiator = {item: v for item, v in zip(items, val_responding_negotiator)}
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)

def random_quant_fixed_vals(items, min_quant, max_quant, val_starting_negotiator, val_responding_negotiator, random_seed=None):
    rng = np.random.default_rng(random_seed)
    quantities = {item: int(rng.integers(min_quant, max_quant + 1)) for item in items}
    val_starting_negotiator = {item: v for item, v in zip(items, val_starting_negotiator)}
    val_responding_negotiator = {item: v for item, v in zip(items, val_responding_negotiator)}
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)

def bicameral_vals_assignator(items, min_quant, max_quant, low_val_mean, low_val_std, high_val_mean, high_val_std, random_seed=None):
    rng = np.random.default_rng(random_seed)
    quantities = {item: int(rng.integers(min_quant, max_quant + 1)) for item in items}

    bernoullis = np.random.binomial(1, 0.5, len(items))
    random_left = np.random.normal(low_val_mean, low_val_std, len(items))
    random_right = np.random.normal(high_val_mean, high_val_std, len(items))

    vals_0 = np.ceil( np.abs(bernoullis * random_left + (1 - bernoullis) * random_right) + 0.001)
    vals_1 = np.ceil( np.abs((1 - bernoullis) * random_left + (bernoullis) * random_right) + 0.001)
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

    if round_number % 2 == 0:
        # Even rounds: agent_0 is "starting_negotiator"
        agent_to_role = {agents[0]: roles[0], agents[1]: roles[1]}
    else:
        # Odd rounds: agent_1 is "starting_negotiator"
        agent_to_role = {agents[0]: roles[1], agents[1]: roles[0]}

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