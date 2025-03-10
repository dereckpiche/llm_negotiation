=================
Diplomacy
=================

The Diplomacy environment provides a multi-agent negotiation interface for the classic board game Diplomacy, 
based on DeepMind's implementation. This document describes the API for interacting with the Diplomacy environment
and its associated agent handler.

Overview
--------

Diplomacy is a strategic board game set in Europe before World War I, where players control one of seven European powers 
and negotiate with each other to gain control of supply centers. The game is played in turns, with each turn consisting 
of movement phases, retreat phases, and build phases.

Our implementation adapts DeepMind's Diplomacy code to the Multi-Agent Negotiation Environment standard, allowing it 
to be used with LLM agents through a text-based interface.

Game Rules
----------

### Game Board and Powers

Diplomacy is played on a map of Europe divided into provinces. The game features seven Great Powers that players can control:

- England (blue)
- France (light blue)
- Germany (black)
- Italy (green)
- Austria-Hungary (red)
- Russia (white)
- Turkey (yellow)

Each power begins with three supply centers (except Russia, which starts with four) and an equal number of units.

### Units and Movement

There are two types of units in Diplomacy:
- **Armies (A)**: Can move to adjacent land provinces or be convoyed across water by fleets
- **Fleets (F)**: Can move to adjacent coastal provinces and sea regions

During movement phases, each unit can execute one of these orders:
- **Hold**: The unit remains in its current province (e.g., "A PAR H")
  - Format: [Unit Type] [Province] H
  - Example: "A PAR H" means "Army in Paris holds its position"

- **Move**: The unit attempts to move to an adjacent province (e.g., "A PAR - BUR")
  - Format: [Unit Type] [Current Province] - [Destination Province]
  - Example: "A PAR - BUR" means "Army in Paris moves to Burgundy"
  - Example: "F BRE - ENG" means "Fleet in Brest moves to the English Channel"

- **Support**: The unit supports another unit's move or hold (e.g., "A PAR S A MAR - BUR")
  - Format for supporting a move: [Unit Type] [Province] S [Unit Type] [Province] - [Destination]
  - Format for supporting a hold: [Unit Type] [Province] S [Unit Type] [Province]
  - Example: "A PAR S A MAR - BUR" means "Army in Paris supports the Army in Marseille's move to Burgundy"
  - Example: "F LON S F NTH" means "Fleet in London supports the Fleet in North Sea holding its position"

- **Convoy**: A fleet can convoy an army across water (e.g., "F ENG C A LON - BRE")
  - Format: [Fleet] [Sea Province] C [Army] [Coastal Province] - [Coastal Province]
  - Example: "F ENG C A LON - BRE" means "Fleet in English Channel convoys the Army in London to Brest"

All orders are executed simultaneously, and conflicts are resolved based on strength (number of supporting units).

### Common Province Abbreviations

Diplomacy uses three-letter abbreviations for provinces. Some common ones include:
- **PAR**: Paris
- **LON**: London
- **BER**: Berlin
- **MUN**: Munich
- **BUR**: Burgundy
- **MAR**: Marseilles
- **BRE**: Brest
- **ENG**: English Channel
- **NTH**: North Sea
- **VIE**: Vienna
- **ROM**: Rome
- **VEN**: Venice
- **MOW**: Moscow
- **CON**: Constantinople

### Example: Movement and Conflicts

For example, if France orders "A PAR - BUR" and Germany orders "A MUN - BUR", neither move succeeds as they have equal strength. However, if France also orders "A MAR S A PAR - BUR", then the French army from Paris would successfully move to Burgundy with strength of 2 against Germany's strength of 1.

### Turn Structure

A game year consists of five phases:
1. **Spring Movement**: All powers submit orders for their units
2. **Spring Retreat**: Units dislodged in the movement phase must retreat or be disbanded
3. **Fall Movement**: Another round of movement orders
4. **Fall Retreat**: Retreat orders for dislodged units
5. **Winter Adjustment**: Powers gain or lose units based on the number of supply centers they control

### Supply Centers and Building

Supply centers (marked on the map) are key to victory. When a power occupies a supply center during a Fall turn, they gain control of it. During the Winter Adjustment phase:
- If you control more supply centers than you have units, you can build new units in your home supply centers
- If you control fewer supply centers than you have units, you must remove excess units

### Example: Building and Removing Units

If France controls 5 supply centers but only has 4 units, during the Winter phase they can build one new unit in an unoccupied home supply center (Paris, Marseilles, or Brest). Conversely, if France controls only 3 supply centers but has 4 units, they must remove one unit of their choice.

### Negotiation

A critical component of Diplomacy is the negotiation between players. Before submitting orders, players can communicate freely to form alliances, coordinate attacks, or mislead opponents. These negotiations are not binding, and betrayal is a common strategy.

### Example: Alliance and Betrayal

England and France might agree to an alliance against Germany, with England promising to support France's move into Belgium. However, England could secretly order their fleet to move into Belgium themselves or support a German move instead.

### Victory Conditions

The game ends when one power controls 18 or more supply centers (majority of the 34 total centers), or when players agree to a draw. In tournament settings, games may also end after a predetermined number of game years.

DiplomacyEnv
------------

The ``DiplomacyEnv`` class provides an interface to the Diplomacy game environment that follows the Multi-Agent 
Negotiation Environment standard.

.. code-block:: python

    class DiplomacyEnv:
        """
        Multi-Agent Negotiation Environment for Diplomacy, adapting Deepmind's implementation
        to the MarlEnvironment standard.
        """
        def __init__(self, 
                    initial_state: Optional[DiplomacyState] = None,
                    max_turns: int = 100,
                    points_per_supply_centre: bool = True,
                    forced_draw_probability: float = 0.0,
                    min_years_forced_draw: int = 35):
            """Initialize the Diplomacy environment.
            
            Args:
                initial_state: Initial DiplomacyState (optional)
                max_turns: Maximum number of turns in the game
                points_per_supply_centre: Whether to award points per supply center in case of a draw
                forced_draw_probability: Probability of forcing a draw after min_years_forced_draw
                min_years_forced_draw: Minimum years before considering a forced draw
            """
            # ...
            
        def reset(self):
            """Reset the environment to an initial state and return the initial observation.
            
            Returns:
                observation (dict): A dictionary where keys are agent identifiers and values are observations.
                Each observation contains:
                - board_state: Current state of the board
                - current_season: Current season in the game
                - player_index: Index of the player's power
                - possible_actions: List of possible actions in DeepMind's format
                - human_readable_actions: List of human-readable action descriptions
                - supply_centers: List of supply centers owned by the player
                - units: List of units owned by the player
                - year: Current year in the game
            """
            # ...
            
        def step(self, actions):
            """Take a step in the environment using the provided actions.

            Args:
                actions (dict): A dictionary where keys are agent identifiers and values are actions.
                    Actions can be:
                    - List of integer actions in DeepMind's format
                    - List of string actions in text format (e.g., "A MUN - BER")

            Returns:
                observations (dict): A dictionary where keys are agent identifiers and values are observations.
                    Each observation has the same structure as in reset().
                done (bool): Whether the episode has ended.
                info (dict): Additional information about the environment, including:
                    - turn: Current turn number
                    - returns: Game returns if the game is done, otherwise None
                    - waiting_for: List of agents that still need to provide actions (if not all actions are provided)
            """
            # ...
            
        def get_log_info(self):
            """Get additional information about the environment for logging.
            
            Returns:
                log_info (dict): Information about the environment required to log the game, including:
                    - power_names: List of power names
                    - game_history: History of the game
                    - current_turn: Current turn number
                    - current_season: Current season name
                    - supply_centers: Dictionary mapping power names to supply center counts
            """
            # ...
            
        def render(self):
            """Render the current state of the environment.
            
            Displays a visualization of the current game state.
            """
            # ...
            
        def close(self):
            """Perform any necessary cleanup."""
            # ...


Key Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DiplomacyEnv`` class implements several key features:

1. **Multi-Agent Support**: The environment tracks multiple agents (powers) and manages their interactions.

2. **Turn-Based Gameplay**: The environment enforces the turn structure of Diplomacy, including different phases.

3. **Action Processing**: The environment can handle actions in both text format and DeepMind's integer format.

4. **Observation Generation**: The environment generates detailed observations for each agent, including board state, supply centers, and possible actions.

5. **Game Termination**: The environment tracks game termination conditions, including supply center victory and maximum turn limits.

Observation Structure
~~~~~~~~~~~~~~~~~~~~

Each agent receives an observation dictionary with the following structure:

.. code-block:: python

    {
        "board_state": np.ndarray,  # Board state representation
        "current_season": int,      # Season index (0-4)
        "player_index": int,        # Index of the player's power (0-6)
        "possible_actions": [int],  # List of possible actions in DeepMind's format
        "human_readable_actions": [str],  # List of human-readable action descriptions
        "supply_centers": [str],    # List of supply centers owned by the player
        "units": [dict],            # List of units owned by the player
        "year": int                 # Current year in the game
    }

Action Structure
~~~~~~~~~~~~~~~

Actions can be provided in two formats:

1. **Text Format**: String actions like ``"A MUN - BER"`` or ``"F NTH C A LON - BEL"``.

2. **Integer Format**: Lists of integers corresponding to DeepMind's action representation.

The environment will convert text actions to the internal format as needed.

DiplomacyAgent
--------------

The ``DiplomacyAgent`` class implements the agent handler interface for Diplomacy, processing observations from the environment and generating actions through an LLM.

.. code-block:: python

    class DiplomacyAgent:
        """
        Agent handler for Diplomacy, implementing the AgentState interface
        for the multi-agent negotiation standard.
        """
        
        def __init__(self, 
                    power_name: str,
                    use_text_interface: bool = True,
                    system_prompt: Optional[str] = None):
            """Initialize the Diplomacy agent handler.
            
            Args:
                power_name: Name of the power this agent controls
                use_text_interface: Whether to use text-based interface (vs. structured)
                system_prompt: Optional system prompt to use for the LLM
            """
            # ...
            
        def step(self, observation_from_env, policy_output=None):
            """Update the agent state based on the observation and action.
            
            Args:
                observation_from_env: The observation from the environment, with structure:
                    - board_state: Current state of the board
                    - current_season: Current season in the game
                    - player_index: Index of the player's power
                    - possible_actions: List of possible actions
                    - human_readable_actions: List of human-readable action descriptions
                    - supply_centers: List of supply centers owned by the player
                    - units: List of units owned by the player
                    - year: Current year in the game
                
                policy_output: The output of the policy (LLM response), or None for initial prompt
                
            Returns:
                policy_id (str): The policy identifier ("llm_policy")
                policy_input (dict): The input to the policy, with structure:
                    - messages: List of conversation messages in the format:
                        [{"role": "system", "content": "..."}, 
                         {"role": "user", "content": "..."}]
                action: The official action to be sent to the environment, or None if not ready
                done (bool): Whether the LLM action is ready to be sent to the environment
                info (dict): Additional information about the agent:
                    - valid_action: Whether the extracted action is valid
            """
            # ...
            
        def get_log_info(self):
            """Get information about the agent required to log a trajectory.
            
            Returns:
                log_info (dict): Information about the agent required to log a trajectory:
                    - power_name: Name of the power this agent controls
                    - conversation_history: List of conversation messages
                    - current_action: The current action, if any
            """
            # ...
            
        def render(self):
            """Render the current state of the agent.
            
            Displays the agent's current state, including conversation history.
            """
            # ...
            
        def close(self):
            """Perform any necessary cleanup."""
            # ...


Key Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DiplomacyAgent`` class implements several key features:

1. **LLM Interaction**: The agent generates prompts for an LLM and processes the LLM's responses to extract actions.

2. **Conversation Management**: The agent maintains a conversation history for coherent interactions with the LLM.

3. **Action Validation**: The agent validates extracted actions against the set of possible actions provided by the environment.

4. **Error Handling**: The agent generates clarification prompts when invalid actions are detected.

5. **Text-Based Interface**: The agent formats game state information into human-readable text for the LLM.

Prompt Structure
~~~~~~~~~~~~~~~

The agent generates prompts that include:

1. **System Prompt**: Instructions and context for the LLM, explaining its role as a Diplomacy player.

2. **Game State Description**: A text description of the current game state, including:
   - Current year and season
   - Supply centers owned
   - Units controlled
   - Possible actions

3. **Action Request**: Instructions on how to format actions.

Example system prompt:

.. code-block:: text

    You are playing the role of FRANCE in a game of Diplomacy. 
    Your goal is to control as many supply centers as possible. 
    You can negotiate with other players and form alliances, but remember that 
    these alliances are not binding. When you need to submit orders for your units,
    write them in the correct format, with each order on a new line.

Example game state description:

.. code-block:: text

    Year: 1901, Season: SPRING_MOVES
    You are playing as FRANCE.
    You currently control 3 supply centers: PAR, MAR, BRE.
    Your units are: A PAR, A MAR, F BRE.

    Please provide orders for your units. Here are your possible actions:
    A PAR - BUR
    A PAR - GAS
    A PAR - PIC
    A PAR H
    ...

    Submit your orders, one per line, in the format like: "A MUN - BER" or "F NTH C A LON - BEL"

Running Diplomacy Games
----------------------

To run Diplomacy games with LLM agents, you can use the ``run_batched_matches`` function with the ``DiplomacyEnv`` and ``DiplomacyAgent`` classes:

.. code-block:: python

    from src.environments.diplomacy.diplomacy_env import DiplomacyEnv
    from src.environments.diplomacy.diplomacy_agent import DiplomacyAgent
    from src.run_matches import run_batched_matches

    # Create environment and agent handlers
    env = DiplomacyEnv(max_turns=30)
    
    agent_handlers = {
        "AUSTRIA": DiplomacyAgent(power_name="AUSTRIA"),
        "ENGLAND": DiplomacyAgent(power_name="ENGLAND"),
        "FRANCE": DiplomacyAgent(power_name="FRANCE"),
        "GERMANY": DiplomacyAgent(power_name="GERMANY"),
        "ITALY": DiplomacyAgent(power_name="ITALY"),
        "RUSSIA": DiplomacyAgent(power_name="RUSSIA"),
        "TURKEY": DiplomacyAgent(power_name="TURKEY")
    }

    # Define policy mapping (mapping from policy IDs to actual policy functions)
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

    # Process results
    for result in game_results:
        print(f"Game finished. Winner: {result['winner']}")
        print(f"Supply centers: {result['supply_centers']}")

This setup allows you to run Diplomacy games with LLM agents using the Multi-Agent Negotiation Environment standard.

Limitations and Considerations
-----------------------------

1. **Performance**: Processing observations and actions for seven powers using LLMs can be computationally intensive.

2. **Action Parsing**: Extracting valid actions from LLM outputs may require sophisticated parsing and error handling.

3. **Game Complexity**: Diplomacy is a complex game with many rules and edge cases, which may be challenging for LLMs to fully grasp.

4. **Turn Duration**: Real Diplomacy games include negotiation phases of variable duration, which are not fully captured in this implementation.

5. **Text Formatting**: The quality of LLM interactions depends heavily on the formatting and clarity of text prompts.

Advanced Usage
------------

For advanced usage, you can customize:

1. **System Prompts**: Modify agent behavior by providing custom system prompts.

2. **Observation Processing**: Extend the observation processing to include additional information.

3. **Action Parsing**: Implement more sophisticated action parsing for complex orders.

4. **Visualization**: Add custom visualization methods to the environment's render function.

5. **Logging**: Extend the logging capabilities to capture additional information about the game state. 