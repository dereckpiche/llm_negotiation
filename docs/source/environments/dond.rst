=================
Deal or No Deal
=================

The Deal or No Deal (DoND) environment provides a multi-agent negotiation interface where players trade 
items with different values. This document describes the API for interacting with the DoND environment
and its associated agent handler.

Overview
--------

Deal or No Deal is a negotiation game where two agents must agree on how to divide a set of items, 
each of which has different values to each agent. The agents engage in a back-and-forth dialogue to 
determine an allocation of the items, with each trying to maximize their own total value.

Our implementation follows the Multi-Agent Negotiation Environment standard, allowing it to be used 
with LLM agents through a text-based interface.

Game Rules
----------

### Basic Structure

The core mechanics of Deal or No Deal are:

1. Two agents negotiate over a set of items (e.g., books, balls, hats)
2. Each item has:
   - A specific quantity (how many of each item is available)
   - A value for each agent (which may differ between agents)
3. Agents take turns sending messages to negotiate how to split the items
4. Once an agreement is reached, agents finalize the deal
5. Points are awarded based on the value of items each agent receives

### Detailed Gameplay

#### Setup Phase

The game begins with:
- A set of items (e.g., "book", "hat", "ball")
- Each item has a quantity (e.g., 6 books, 2 hats, 4 balls)
- Each agent has private values for each item (e.g., books might be worth 5 points to one agent but only 2 points to the other)
- Agents are assigned roles (starting negotiator and responding negotiator)

#### Negotiation Phase

1. Agents take turns sending free-form text messages to each other
2. Messages can include offers, counter-offers, questions, or strategic communication
3. There is a maximum number of messages permitted (preventing endless negotiations)
4. Either agent can propose to finalize an agreement at any time

For example:
- Agent 1: "I propose I get all the books and you get all the hats and balls."
- Agent 2: "That doesn't work for me. How about you get 3 books and I get 3 books, all the hats, and all the balls?"
- Agent 1: "Let me counter-offer: I get 4 books and 2 balls, you get 2 books, all hats, and 2 balls."

#### Finalization Phase

1. When an agent wants to finalize a deal, they must specify the exact allocation:
   - How many of each item they receive
   - How many of each item the other agent receives
2. The other agent must then either agree (by submitting the same allocation) or reject the finalization
3. If both agents submit matching finalizations, the deal is executed
4. If finalizations don't match, no agreement is reached, and both agents receive 0 points

#### Scoring

1. Each agent's score is calculated based on the value of items they receive
2. The formula is: Sum(quantity_of_item_i × value_of_item_i_to_agent)
3. If no agreement is reached, both agents receive 0 points

### Example Game

Let's walk through a simple example:

**Setup:**
- Items: Books (4), Hats (2), Balls (6)
- Agent 1 values: Books=5, Hats=1, Balls=2
- Agent 2 values: Books=3, Hats=6, Balls=1

**Negotiation (simplified):**
1. Agent 1: "I would like all the books and balls. You can have the hats."
2. Agent 2: "That doesn't work for me. Books are valuable. I propose I get all the hats and 2 books, you get 2 books and all the balls."
3. Agent 1: "How about I get 3 books and all the balls, and you get 1 book and all the hats?"
4. Agent 2: "I accept your proposal."

**Finalization:**
- Agent 1 submits: Agent 1 gets (Books: 3, Hats: 0, Balls: 6), Agent 2 gets (Books: 1, Hats: 2, Balls: 0)
- Agent 2 submits the same allocation, confirming agreement

**Scoring:**
- Agent 1 score: (3 books × 5) + (0 hats × 1) + (6 balls × 2) = 15 + 0 + 12 = 27 points
- Agent 2 score: (1 book × 3) + (2 hats × 6) + (0 balls × 1) = 3 + 12 + 0 = 15 points

### Game Variations

The DoND environment supports several variations through configuration parameters:

#### Different Value Distributions

The environment offers multiple ways to assign values to items:

1. **Standard Random Setup (dond_random_setup)**:
   - Items have even-numbered quantities
   - Each agent receives distinct random values for each item
   - Values are drawn from a uniform distribution

2. **Independent Random Values (independent_random_vals)**:
   - Item quantities can be any number in the specified range
   - Values for each agent are drawn independently
   - Creates more varied negotiation scenarios

3. **Bicameral Value Distribution (bicameral_vals_assignator)**:
   - Creates a "high value" and "low value" distribution for each item
   - Each agent values approximately half the items highly and half lowly
   - Values are drawn from normal distributions with different means
   - Creates scenarios with clear trade opportunities

#### Visibility Options

1. **Finalization Visibility**:
   - When enabled, both agents can see each other's finalization proposals
   - When disabled, finalization proposals remain private until both are submitted

2. **Other Values Visibility**:
   - When enabled, agents can see each other's value functions
   - When disabled, agents only know their own values
   - Creates information asymmetry and richer negotiation dynamics

#### Game Modes

1. **Cooperative Mode ("coop")**:
   - Agents are encouraged to find mutually beneficial solutions
   - Success is measured by the sum of both agents' scores

2. **Competitive Mode ("comp")**:
   - Agents aim to maximize their individual scores
   - Creates more adversarial negotiations

#### Round Structure

1. **Single Round**:
   - One negotiation session between the same agents
   - Simple evaluation of negotiation skills

2. **Multiple Rounds**:
   - Agents negotiate multiple times with different item setups
   - Allows for learning and adaptation over time
   - Roles can be swapped between rounds

DondEnv
------------

The ``DondEnv`` class provides an interface to the Deal or No Deal environment that follows the Multi-Agent 
Negotiation Environment standard.

.. code-block:: python

    class DondEnv:
        """
        Multi-Agent Negotiation Environment for Deal or No Deal.
        """
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
            """Initialize the Deal or No Deal environment.
            
            Args:
                agents: List of agent IDs participating in the game
                mode: Game mode ("coop" or "comp")
                max_messages: Maximum number of messages per agent per round
                min_messages: Minimum number of messages per agent per round
                max_chars_per_message: Maximum characters per message
                rounds_per_game: Number of negotiation rounds to play
                random_setup_func: Function to generate item quantities and values
                random_setup_kwargs: Arguments for the random setup function
                role_assignator_func: Function to assign roles to agents
                role_assignator_func_kwargs: Arguments for the role assignator
                finalization_visibility: Whether agents can see each other's finalizations
                other_values_visibility: Whether agents can see each other's values
                random_seed: Seed for reproducibility
            """
            # ...
            
        def reset(self):
            """Reset the environment to an initial state and return the initial observation.
            
            Returns:
                observation (dict): A dictionary where keys are agent identifiers and values are observations.
            """
            # ...
            
        def step(self, actions):
            """Take a step in the environment using the provided actions.

            Args:
                actions (dict): A dictionary where keys are agent identifiers and values are actions.
                    Actions can be messages or finalization proposals.

            Returns:
                observations (dict): A dictionary where keys are agent identifiers and values are observations.
                done (bool): Whether the episode has ended.
                info (dict): Additional information about the environment.
            """
            # ...
            
        def get_state(self):
            """Retrieve the current state of the game.
            
            Returns:
                state (dict): The current state of the game, including items, quantities, values, etc.
            """
            # ...

Key Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DondEnv`` class implements several key features:

1. **Multi-Agent Support**: The environment tracks two agents and manages their alternating messages.

2. **Turn-Based Dialogue**: The environment enforces turn structure and limits on message count.

3. **Finalization Processing**: The environment validates and processes finalization proposals.

4. **Random Setup**: The environment supports multiple methods of generating negotiation scenarios.

5. **Round Management**: The environment can handle multiple rounds with different setups.

Observation Structure
~~~~~~~~~~~~~~~~~~~~

Each agent receives an observation (state) dictionary with rich information about the game:

.. code-block:: python

    {
        "mode": str,                 # Game mode ("coop" or "comp")
        "role_values": dict,         # Value mappings for each role
        "role_props": dict,          # Properties for each role
        "agent_to_role": dict,       # Mapping from agent IDs to roles
        "is_new_round": bool,        # Whether this is the start of a new round
        "is_new_game": bool,         # Whether this is the start of a new game
        "game_over": bool,           # Whether the game is over
        "items": list,               # List of item names
        "quantities": dict,          # Quantities of each item
        "has_finalized": bool,       # Whether finalization has been proposed
        "last_message": dict,        # The last message sent
        "messages_remaining": dict,  # Number of messages each agent can still send
        # And various history tracking fields
    }

Action Structure
~~~~~~~~~~~~~~~

Actions can be:

1. **Text Messages**: Free-form text for negotiation.
2. **Finalization Proposals**: Structured data specifying the exact allocation of items.

Example finalization format:

.. code-block:: python

    {
        "type": "finalize",
        "allocation": {
            "agent1": {"book": 3, "hat": 0, "ball": 6},
            "agent2": {"book": 1, "hat": 2, "ball": 0}
        }
    }

Value Setup Functions
--------------------

The DoND environment provides several functions for setting up item values:

.. code-block:: python

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
        
        Returns:
            tuple: (items, quantities, (val_starting_negotiator, val_responding_negotiator))
        """
        # ...
    
    def independent_random_vals(items, min_quant, max_quant, min_val, max_val, random_seed=None):
        """
        Generates random quantities and independent random values for both agents.
        
        Args:
            Similar to dond_random_setup
        
        Returns:
            tuple: (items, quantities, (val_starting_negotiator, val_responding_negotiator))
        """
        # ...
    
    def bicameral_vals_assignator(items, min_quant, max_quant, low_val_mean, low_val_std, high_val_mean, high_val_std, random_seed=None):
        """
        Generates values with a bicameral distribution - each agent values half the items highly.
        
        Args:
            items (list): List of items.
            min_quant, max_quant: Range for quantities
            low_val_mean, low_val_std: Mean and standard deviation for the "low value" distribution
            high_val_mean, high_val_std: Mean and standard deviation for the "high value" distribution
            random_seed: Seed for reproducibility
        
        Returns:
            tuple: (items, quantities, (val_starting_negotiator, val_responding_negotiator))
        """
        # ...

Running DoND Games
----------------------

To run Deal or No Deal games with LLM agents, you can use the following structure:

.. code-block:: python

    from src.environments.dond.dond_game import DondEnv
    from src.environments.dond.dond_agent import DondAgent
    from src.run_matches import run_batched_matches

    # Create environment
    env = DondEnv(
        agents=["agent1", "agent2"],
        mode="coop",
        max_messages=10,
        rounds_per_game=1,
        random_setup_func="dond_random_setup",
        random_setup_kwargs={
            "items": ["book", "hat", "ball"],
            "min_quant": 2,
            "max_quant": 8,
            "min_val": 1,
            "max_val": 10
        },
        finalization_visibility=False
    )
    
    # Create agent handlers (implementation details would vary)
    agent_handlers = {
        "agent1": DondAgent(agent_id="agent1"),
        "agent2": DondAgent(agent_id="agent2")
    }

    # Define policy mapping
    policy_mapping = {
        "llm_policy": my_llm_policy_function
    }

    # Run the game
    game_results = run_batched_matches(
        envs=[env],
        agent_handlers_per_env=[agent_handlers],
        policy_mapping=policy_mapping,
        max_parallel_matches=1
    )

Limitations and Considerations
-----------------------------

1. **Negotiation Complexity**: The open-ended nature of negotiations can be challenging for some LLM agents.

2. **Parsing Challenges**: Extracting structured finalization proposals from free-form text requires robust parsing.

3. **Optimization Opportunities**: Different agents may employ different negotiation strategies to optimize outcomes.

4. **Fairness Evaluation**: The environment allows research into questions of fair division and Pareto optimality.

5. **Strategic Deception**: Agents might strategically misrepresent their true values, adding complexity to negotiations.

Advanced Usage
------------

For advanced usage, you can:

1. **Custom Value Functions**: Create more complex distributions of item values for specific research questions.

2. **Novel Negotiation Scenarios**: Design item sets and values to test specific negotiation skills.

3. **Curriculum Learning**: Create progressively more difficult negotiation scenarios.

4. **Communication Analysis**: Analyze the language and strategies used in successful negotiations.

5. **Multi-Round Dynamics**: Study how agents adapt their strategies over multiple rounds.