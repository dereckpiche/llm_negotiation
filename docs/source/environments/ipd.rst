=================
Iterated Prisoner's Dilemma
=================

The Iterated Prisoner's Dilemma environment provides a classic game theory setting for studying cooperation 
and competition between agents. This document describes the API for interacting with the IPD environment
and its associated agent handler.

Overview
--------

The Prisoner's Dilemma is a fundamental problem in game theory that demonstrates why two rational individuals might not 
cooperate, even when it appears in their best interest to do so. In the iterated version, the same two players 
repeatedly face the same dilemma, allowing for the development of trust or retaliation based on previous interactions.

Our implementation follows the Multi-Agent Negotiation Environment standard, allowing it to be used with 
LLM agents through a text-based interface.

Game Rules
----------

### Basic Premise

The scenario behind the Prisoner's Dilemma is as follows:

Two criminals are arrested and imprisoned. Each prisoner is in solitary confinement with no means of communicating with 
the other. The prosecutors lack sufficient evidence to convict the pair on the principal charge, but they have enough 
to convict both on a lesser charge. Simultaneously, the prosecutors offer each prisoner a bargain:

- If both prisoners betray each other, each serves 2 years in prison (the "punishment" payoff)
- If one betrays the other while the other remains silent, the betrayer goes free (the "temptation" payoff) while the 
  silent accomplice serves 3 years (the "sucker" payoff)
- If both remain silent, each serves only 1 year in prison (the "reward" payoff)

### Game Mechanics

In our implementation, the choices are simplified to:
- **C**: Cooperate (remain silent)
- **D**: Defect (betray the other prisoner)

Each round, both players simultaneously choose either C or D, and receive points based on the combination of their choices:

- Both choose C: Both receive the "reward" payoff (3 points by default)
- Both choose D: Both receive the "punishment" payoff (1 point by default)
- One chooses C, one chooses D: The defector receives the "temptation" payoff (5 points by default), while the cooperator 
  receives the "sucker" payoff (0 points by default)

### Example: Single Round

Let's see how a single round plays out:

1. Alice and Bob simultaneously make their choices
2. If Alice chooses C and Bob chooses C:
   - Alice receives 3 points
   - Bob receives 3 points
3. If Alice chooses C and Bob chooses D:
   - Alice receives 0 points
   - Bob receives 5 points
4. If Alice chooses D and Bob chooses C:
   - Alice receives 5 points
   - Bob receives 0 points
5. If Alice chooses D and Bob chooses D:
   - Alice receives 1 point
   - Bob receives 1 point

### Iterated Game Structure

The iterated version repeats this basic game for a fixed number of rounds. The key features are:

1. Players know the total number of rounds in advance
2. After each round, players learn what choice the other player made
3. Players maintain a cumulative score across all rounds
4. Players can adjust their strategy based on the history of previous interactions

### Game Variations

The IPD environment supports several variations through configuration parameters:

#### Different Payoff Matrices

The standard payoff values can be modified to create different incentive structures:
- **Traditional PD**: reward=3, punishment=1, temptation=5, sucker=0
- **Weak Temptation**: reward=3, punishment=1, temptation=4, sucker=0 (reduces the incentive to defect)
- **Harsh Punishment**: reward=3, punishment=0, temptation=5, sucker=0 (increases the cost of mutual defection)
- **Generous**: reward=4, punishment=2, temptation=5, sucker=1 (cushions the blow of being betrayed)

#### Game Length Variations

The number of rounds can significantly impact strategy:
- **Short Games** (5-10 rounds): Incentivizes more defection, especially near the end
- **Medium Games** (20-50 rounds): Allows for the development of tit-for-tat and forgiveness strategies
- **Long Games** (100+ rounds): Favors steady cooperation with occasional "probing" defections

### Common Strategies

While not enforced by the environment, several well-known strategies can emerge:
- **Always Cooperate**: Always choose C
- **Always Defect**: Always choose D
- **Tit for Tat**: Start with C, then copy what the opponent did in the previous round
- **Forgiving Tit for Tat**: Like Tit for Tat, but occasionally cooperate even after being defected against
- **Grudger**: Cooperate until the opponent defects once, then always defect
- **Random**: Choose randomly between C and D

IPDEnv
------

The ``IPDEnv`` class provides an interface to the Iterated Prisoner's Dilemma environment that follows the 
Multi-Agent Negotiation Environment standard.

.. code-block:: python

    class IPDEnv:
        """
        Iterated Prisoner's Dilemma environment following the MarlEnvironment standard.
        
        In each round of the game, two agents simultaneously choose to either cooperate (C) or defect (D).
        The payoffs are as follows:
        - If both cooperate: Both receive the "reward" (usually 3 points)
        - If both defect: Both receive the "punishment" (usually 1 point)
        - If one cooperates and one defects: The defector receives the "temptation" (usually 5 points)
          and the cooperator receives the "sucker" payoff (usually 0 points)
        
        The game is played for a specified number of rounds.
        """
        
        def __init__(
            self,
            rounds_per_game: int = 10,
            reward: float = 3.0,           # Both cooperate
            punishment: float = 1.0,       # Both defect
            temptation: float = 5.0,       # Defector's reward when other cooperates
            sucker: float = 0.0,           # Cooperator's reward when other defects
            random_seed: Optional[int] = None,
        ):
            """
            Initialize the Iterated Prisoner's Dilemma environment.
            
            Args:
                rounds_per_game: Number of rounds to play
                reward: Payoff when both agents cooperate
                punishment: Payoff when both agents defect
                temptation: Payoff for defecting when other agent cooperates
                sucker: Payoff for cooperating when other agent defects
                seed: Random seed for reproducibility
            """
            # ...
            
        def reset(self) -> Dict[str, Dict[str, Any]]:
            """
            Reset the environment to an initial state and return the initial observation.
            
            Returns:
                observation (dict): A dictionary where keys are agent identifiers and values are observations.
            """
            # ...
        
        def step(self, actions: Dict[str, str]) -> Tuple[Dict[str, Dict[str, Any]], bool, Dict[str, Any]]:
            """
            Take a step in the environment using the provided actions.
            
            Args:
                actions (dict): A dictionary where keys are agent identifiers and values are actions ('C' or 'D').
            
            Returns:
                observations (dict): A dictionary where keys are agent identifiers and values are observations.
                done (bool): Whether the episode has ended.
                info (dict): Additional information about the environment.
            """
            # ...

Key Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``IPDEnv`` class implements several key features:

1. **Two-Agent Support**: The environment tracks two agents ("alice" and "bob") and manages their interactions.

2. **Round-Based Play**: The environment enforces turn structure and tracks game history.

3. **Payoff Matrix**: The environment calculates rewards based on the standard prisoner's dilemma payoff matrix.

4. **Observation Generation**: The environment generates detailed observations for each agent, including action history and rewards.

5. **Game Termination**: The environment tracks game termination after the specified number of rounds.

Observation Structure
~~~~~~~~~~~~~~~~~~~~

Each agent receives an observation dictionary with the following structure:

.. code-block:: python

    {
        "current_round": int,                # Current round number (0-indexed)
        "rounds_per_game": int,              # Total number of rounds in the game
        "history": List[Dict],               # Complete game history so far
        "last_round_actions": Dict[str, str], # Actions from the previous round (if any)
        "last_round_reward": float,          # Reward received in the previous round (if any)
        "total_reward": float,               # Cumulative reward so far
        "payoff_matrix": Dict[str, float],   # The game's payoff matrix values
    }

Action Structure
~~~~~~~~~~~~~~~

Actions are simple strings:

1. ``"C"`` for Cooperate
2. ``"D"`` for Defect

IPDAgent
--------------

The ``IPDAgent`` class implements the agent handler interface for the Iterated Prisoner's Dilemma, processing observations from the environment and generating actions through an LLM.

.. code-block:: python

    class IPDAgent:
        """
        Agent handler for Iterated Prisoner's Dilemma, implementing the AgentState interface 
        for the multi-agent negotiation standard.
        """
        
        def __init__(
            self,
            agent_id: str,
            policy_id: str = "llm_policy",
            system_prompt: Optional[str] = None,
            max_errors: int = 3,
            opponent_id: Optional[str] = None,
        ):
            """
            Initialize the IPD agent handler.
            
            Args:
                agent_id: Identifier for this agent ("alice" or "bob")
                policy_id: Identifier for the policy this agent uses
                system_prompt: Optional custom system prompt for the LLM
                max_errors: Maximum number of parsing errors before defaulting to cooperate
                opponent_id: Optional identifier of the opponent (inferred if not provided)
            """
            # ...
            
        def step(self, observation_from_env: Dict[str, Any], policy_output: str = None) -> Tuple[str, Dict[str, Any], str, bool, Dict[str, Any]]:
            """
            Update the agent state based on the observation and process the policy output.
            
            Args:
                observation_from_env: The observation from the environment
                policy_output: The output from the policy (LLM response)
                
            Returns:
                policy_id: The policy identifier
                policy_input: The input to the policy
                action: The action to be sent to the environment
                done: Whether the action is ready to be sent to the environment
                info: Additional information about the agent
            """
            # ...

Key Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``IPDAgent`` class implements several key features:

1. **LLM Interaction**: The agent generates prompts for an LLM and processes the LLM's responses.

2. **Action Extraction**: The agent parses the LLM's output to extract valid actions (C or D).

3. **Error Handling**: The agent provides helpful error messages when parsing fails and defaults to cooperation after multiple failures.

4. **History Tracking**: The agent maintains and provides the complete game history in its prompts.

5. **Strategy Explanation**: The agent can extract and log the reasoning behind an LLM's decisions.

Prompt Structure
~~~~~~~~~~~~~~~

The agent generates prompts that include:

1. **System Prompt**: Instructions and context for the LLM, explaining its role and the rules of the Prisoner's Dilemma.

2. **Game State Description**: A text description of the current game state, including:
   - Current round number
   - History of previous rounds (if any)
   - Cumulative score

3. **Action Request**: Instructions on how to format the response, requiring an explicit action tag.

Example system prompt:

.. code-block:: text

    You are playing as Alice in an Iterated Prisoner's Dilemma game against Bob.
    In each round, you must choose to either Cooperate (C) or Defect (D).
    
    The payoffs are:
    - If both players Cooperate: You each get 3 points
    - If both players Defect: You each get 1 point
    - If you Cooperate and Bob Defects: You get 0 points, Bob gets 5 points
    - If you Defect and Bob Cooperates: You get 5 points, Bob gets 0 points
    
    Your goal is to maximize your total points across all rounds.
    The game will last for exactly 10 rounds, and both players know this.

Example game state prompt:

.. code-block:: text

    Current round: 3/10
    
    History:
    Round 1: You chose C, Bob chose C. You earned 3 points.
    Round 2: You chose C, Bob chose D. You earned 0 points.
    
    Your total score so far: 3 points
    
    What is your choice for round 3?
    Please respond with <action>C</action> to cooperate or <action>D</action> to defect,
    and explain your reasoning.

Running IPD Games
----------------------

To run Iterated Prisoner's Dilemma games with LLM agents, you can use the following code structure:

.. code-block:: python

    from mllm.environments.ipd.ipd_game import IPDEnv
    from mllm.environments.ipd.ipd_agent import IPDAgent
    from mllm.run_matches import run_batched_matches

    # Create environment
    env = IPDEnv(
        rounds_per_game=10,
        reward=3.0,
        punishment=1.0,
        temptation=5.0,
        sucker=0.0
    )
    
    # Create agent handlers
    agent_handlers = {
        "alice": IPDAgent(agent_id="alice"),
        "bob": IPDAgent(agent_id="bob")
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

    # Process results
    for result in game_results:
        print(f"Game finished. Scores: {result['total_rewards']}")

Statistics and Analysis
----------------------

The IPD environment includes utility functions for analyzing game outcomes:

1. **Cooperation Rates**: Percentage of rounds where each agent cooperated.
2. **Mutual Cooperation/Defection**: Percentage of rounds where both agents made the same choice.
3. **Score Distribution**: Analysis of how points were accumulated over the game.

These statistics can be calculated using the ``gather_ipd_statistics`` function:

.. code-block:: python

    from mllm.environments.ipd.ipd_statistics_funcs import gather_ipd_statistics
    
    stats = gather_ipd_statistics(match_info, env_info)
    print(f"Cooperation rates: {stats['cooperation_rate']}")
    print(f"Mutual cooperation rate: {stats['mutual_cooperation_rate']}")
    print(f"Mutual defection rate: {stats['mutual_defection_rate']}")

Limitations and Considerations
-----------------------------

1. **Determinism**: The environment is deterministic, with randomness only in initialization if a seed is provided.

2. **Limited Player Count**: The IPD environment only supports exactly two players.

3. **Perfect Information**: Both players have perfect information about the game history.

4. **Simultaneous Actions**: Both players act simultaneously, which requires adaptations for some LLM interfaces.

5. **Fixed Game Length**: The total number of rounds is fixed and known to both players from the start.

Advanced Usage
------------

For advanced usage, you can customize:

1. **Payoff Matrix**: Modify reward values to create different incentive structures.

2. **System Prompts**: Customize the LLM's understanding of the game and potential strategies.

3. **Error Handling**: Adjust how the agent responds to invalid LLM outputs.

4. **Analysis**: Create custom statistics gathering for specific research questions.

5. **Integration**: Connect the IPD environment to other negotiation frameworks or tournament systems.