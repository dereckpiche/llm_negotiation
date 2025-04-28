import json
import re

import regex as re

from utils.common_imports import *

class DondAgent:
    def __init__(
        self,
        agent_name,
        allow_reasoning,
        max_errors,
        policy_id,
        enable_strategize=False,
        max_strategize_chars=500,
        value_function_id=None,
        max_reasoning_chars=None,
        intro_prompt=None,
        goal_prompt=None,
        strategize_prompt=None,
        first_round_prompt=None,
        new_round_prompt=None,
        agent_with_first_move_prompt=None,
        agent_with_second_move_prompt=None,
        received_message_prompt=None,
        other_agent_finalized_prompt=None,
        message_mechanics_prompt=None,
        finalization_mechanics_prompt=None,
        dond_version_specificities=None,
        reasoning_mechanics_prompt=None,
        time_to_finalize_prompt=None,
        time_to_send_message_prompt=None,
        message_parser=None,
        finalization_parser=None,
        finalization_parser_kwargs=None,
        strategize_parser=None,
        thinking_parser=None,
    ):
        """
        Initializes the DondAgent.

        Args:
            agent_name (str): The name of the agent.
            allow_reasoning (bool): Whether reasoning is allowed.
            max_errors (int): Maximum number of retries allowed.
            policy_id (str): The model adapter id to use.
            value_function_id (str): The value function id to use.
            max_reasoning_chars (int): Maximum reasoning characters allowed.
            intro_prompt (str): Prompt for the game introduction.
            goal_prompt (str): Prompt for the agent's goal.
            first_round_prompt (str): Prompt for the first round.
            new_round_prompt (str): Prompt for the new round.
            agent_with_first_move_prompt (str): Prompt when the agent is assigned the first move.
            agent_with_second_move_prompt (str): Prompt when the agent is assigned the second move.
            received_message_prompt (str): Prompt when a message is received from the other agent.
            other_agent_finalized_prompt (str): Prompt to indicate that the other agent has finalized.
            message_mechanics_prompt (str, optional): Instructions for message mechanics.
            finalization_mechanics_prompt (str, optional): Instructions for finalization mechanics.
            dond_version_specificities (str, optional): DOND-specific game instructions.
            reasoning_mechanics_prompt (str, optional): Instructions for reasoning mechanics.
            time_to_finalize_prompt (str, optional): Prompt for time to finalize.
            time_to_send_message_prompt (str, optional): Prompt for time to send message.
            message_parser (callable, optional): Function to parse message responses.
            finalization_parser (callable, optional): Function to parse finalization responses.
            strategize_parser (callable, optional): Function to parse strategize responses.
            thinking_parser (callable, optional): Function to parse thinking (reasoning) responses.
            attribution_map (dict, optional): Mapping of agent names to their attribution keys for finalization.
                Format: {
                    "agent_name": {
                        "i_take_key": "key_for_self",
                        "other_takes_key": "key_for_other"
                    },
                    ...
                }
        """
        self.agent_name = agent_name
        self.allow_reasoning = allow_reasoning
        self.max_errors = max_errors
        self.policy_id = policy_id
        self.value_function_id = value_function_id
        self.max_reasoning_chars = max_reasoning_chars
        self.enable_strategize = enable_strategize
        self.max_strategize_chars = max_strategize_chars

        self.intro_prompt = intro_prompt
        self.goal_prompt = goal_prompt
        self.strategize_prompt = strategize_prompt
        self.first_round_prompt = first_round_prompt
        self.new_round_prompt = new_round_prompt
        self.agent_with_first_move_prompt = agent_with_first_move_prompt
        self.agent_with_second_move_prompt = agent_with_second_move_prompt
        self.received_message_prompt = received_message_prompt
        self.other_agent_finalized_prompt = other_agent_finalized_prompt

        self.needs_to_strategize = self.enable_strategize
        self.has_been_introduced = False
        
        # Set the new mechanics prompts.
        self.message_mechanics_prompt = message_mechanics_prompt
        self.finalization_mechanics_prompt = finalization_mechanics_prompt
        self.dond_version_specificities = (
            dond_version_specificities  # New prompt for version specificities
        )
        self.reasoning_mechanics_prompt = reasoning_mechanics_prompt
        self.time_to_finalize_prompt = time_to_finalize_prompt
        self.time_to_send_message_prompt = time_to_send_message_prompt

        # Store the parser functions
        self.message_parser = globals()[message_parser]
        self.finalization_parser = globals()[finalization_parser]
        self.finalization_parser_kwargs = finalization_parser_kwargs
        self.strategize_parser = globals()[strategize_parser] if strategize_parser else regular_strategize_parser
        self.thinking_parser = globals()[thinking_parser] if thinking_parser else regular_thinking_parser

        self.game_id = None  # ID of the agent in the game
        self.reset()

    def step(self, observation_from_env, policy_output=None):
        """Update the agent state based on the observation and action.
        The action is the output of the LLM.

        Args:
            observation_from_env (dict): The observation of the environment.
            policy_output (str, optional): The output of the policy (LLM response).

        Returns:
            policy_id (str): The policy identifier.
            policy_input (dict): The input to the policy.
            action : The official action to be sent to the environment.
            done (bool): Whether the LLM action is ready to be sent to the environment.
            info (dict): Additional information about the agent.
        """
        state = observation_from_env
        is_error = False
        error_message = None

        # If we don't have policy output, we always need to generate the user message and return policy input
        if policy_output is None:
            # Set the user message in chat history based on the current state
            user_message = self.get_user_message(state, is_error, error_message)
            self.add_to_chat_history(user_message)
            return self.policy_id, self.chat_history, None, False, self.get_log_info()

        # If we have policy output, we need to process it and determine the action
        else:
            # Process the LLM output
            (
                is_error,
                error_message,
                is_finalization,
                processed_response,
                raw_response,
            ) = self.process_response(policy_output, state)

            # Add the model response to chat history
            model_response = {
                "role": "assistant",
                "content": policy_output,
                "is_error": is_error,
                "is_finalization": is_finalization,
                "round_nb": state["round_number"],
            }
            self.add_to_chat_history(model_response)

            # Handle strategize 
            if self.needs_to_strategize:
                self.needs_to_strategize = False
                user_message = self.get_user_message(state, is_error, error_message)
                self.add_to_chat_history(user_message)
                return (
                    self.policy_id,
                    self.chat_history,
                    None,
                    False,
                    self.get_log_info(),
                )

            # Handle errors
            if is_error:
                self.retries += 1
                self.error_message = error_message

                # Too many mistakes were made: force dummy message
                if self.retries > self.max_errors:
                    self.retries = 0
                    if self.needs_to_strategize: self.needs_to_strategize = False
                    # If the policy output indicates a finalization move via the <finalize> tag,
                    # we keep the fallback as "-------". Otherwise (it was time to send a message),
                    # we generate a fallback message truncated to the maximum allowed characters.
                    fallback = "[ERROR]"
                    if policy_output and "<finalize>" not in policy_output:
                        max_chars = state.get("max_chars_per_message", 300)
                        fallback = policy_output[:max_chars]
                    action = (False, fallback, policy_output)
                    return None, None, action, True, self.get_log_info()

                # Set error user message
                user_message = self.get_user_message(state, is_error, error_message)
                self.add_to_chat_history(user_message)

                # Return with a request for policy output again (done = False)
                return (
                    self.policy_id,
                    self.chat_history,
                    None,
                    False,
                    self.get_log_info(),
                )

            # Reset retries on successful processing
            self.retries = 0

            # Create the action to be sent to the environment
            action = (is_finalization, processed_response, raw_response)

            # Action is ready to be sent to the environment
            return None, None, action, True, self.get_log_info()

    def get_log_info(self):
        """Get information about the agent required to log a trajectory.

        Returns:
            log_info (dict): Information about the agent required to log a trajectory.
        """
        return {
            "agent_name": self.agent_name,
            "chat_history": self.chat_history,
            "augmented_chat_history": self.augmented_chat_history,
            "strategize_content": self.strategize_content,
        }

    def render(self):
        """Render the current state of the agent."""
        # Implementation can be expanded if needed
        pass

    def close(self):
        """Perform any necessary cleanup."""
        # Implementation can be expanded if needed
        pass

    ###########################################################################
    # Helper methods below this point
    ###########################################################################

    def get_user_message(self, state, is_error, error_message):
        """
        Constructs a user message based on the current game state.

        Args:
            state (dict): The current state of the game.
        """
        user_message = ""

        if is_error:
            user_message = error_message
            usr_prompt = {
                "role": "user",
                "content": user_message,
                "is_error": True,
                "round_nb": state["round_number"],
            }
            return usr_prompt

        # Use the new move information to decide on the prompts.
        # If the current agent has not yet made any move in the game, prepend the introductory prompts.
        if not self.has_been_introduced:
            user_message += self.format_prompt(self.intro_prompt, state)
            if self.message_mechanics_prompt:
                user_message += "\n\n" + self.format_prompt(
                    self.message_mechanics_prompt, state
                )
            if self.finalization_mechanics_prompt:
                user_message += "\n\n" + self.format_prompt(
                    self.finalization_mechanics_prompt, state
                )
            if self.dond_version_specificities:
                user_message += "\n\n" + self.format_prompt(
                    self.dond_version_specificities, state
                )
            if self.allow_reasoning and self.reasoning_mechanics_prompt:
                user_message += "\n\n" + self.format_prompt(
                    self.reasoning_mechanics_prompt, state
                )
            if self.goal_prompt:
                user_message += "\n\n" + self.format_prompt(self.goal_prompt, state)
            self.has_been_introduced = True

        if self.needs_to_strategize:
            user_message += "\n\n" + self.format_prompt(
                self.strategize_prompt, state
            )
            usr_prompt = {
                "role": "user",
                "content": user_message,
                "is_error": False,
                "round_nb": state["round_number"],
            }
            return usr_prompt

        # If the current agent has not yet made any move in this round, add round instructions.
        if state["round_moves"].get(self.agent_name, 0) == 0:
            if state["round_number"] == 0 and self.first_round_prompt:
                user_message += "\n\n" + self.format_prompt(
                    self.first_round_prompt, state
                )
            elif self.new_round_prompt:
                user_message += self.format_prompt(self.new_round_prompt, state)

        # Then add the appropriate message based on the finalization state and move count.
        if state["has_finalized"]:
            user_message += self.format_prompt(self.other_agent_finalized_prompt, state)
        elif state["last_raw_response"] is None:
            current_round_moves = state["round_moves"].get(self.agent_name, 0)
            if current_round_moves == 0:
                user_message += self.agent_with_first_move_prompt
            elif current_round_moves == 1 and self.agent_with_second_move_prompt:
                user_message += self.agent_with_second_move_prompt
            else:
                user_message += self.format_prompt(self.received_message_prompt, state)
        else:
            user_message += self.format_prompt(self.received_message_prompt, state)

        # Append timing prompts based on the number of remaining messages.
        # Here we use the precomputed remaining_msgs from the state.
        min_msgs = state.get("min_messages", None)
        max_msgs = state.get("max_messages", None)
        agent_messages = state.get("round_messages", {}).get(self.agent_name, 0)
        if agent_messages == max_msgs and self.time_to_finalize_prompt:
            user_message += "\n\n" + self.format_prompt(
                self.time_to_finalize_prompt, state
            )
        elif agent_messages < min_msgs and self.time_to_send_message_prompt:
            user_message += "\n\n" + self.format_prompt(
                self.time_to_send_message_prompt, state
            )

        usr_prompt = {
            "role": "user",
            "content": user_message,
            "is_error": False,
            "round_nb": state["round_number"],
        }
        return usr_prompt

    def process_response(self, response, state):
        """
        Validates and extracts content from the response of the LLM agent.

        Args:
            response (str): The response from the LLM agent.
            state (dict): The current state of the game.

        Returns:
            tuple: (is_error, error_message, is_finalization, processed_response, raw_content)
            where raw_content is the complete original response for display purposes
        """
        # Basic validation for content outside valid tags
        errors = []
        total_length = len(response)
        valid_tags = ["strategize", "think", "message", "finalize"]
        total_tag_content_length = 0
        for tag in valid_tags:
            pattern = rf"<{tag}>.*?</{tag}>"
            matches = re.findall(pattern, response, flags=re.S)
            for match in matches:
                total_tag_content_length += len(match)
        outside_length = total_length - total_tag_content_length
        if outside_length > 5:
            errors.append("Excessive content outside of valid tags.")

        # Check strategize content if enabled
        contains_strategize, strategize_errors, processed_strategize = False, [], None
        if self.enable_strategize:
            contains_strategize, strategize_errors, processed_strategize = self.strategize_parser(
                response, state, self.has_strategized, self.max_strategize_chars
            )
            if contains_strategize and not strategize_errors:
                self.strategize_content = processed_strategize
                self.has_strategized = True

        # Check reasoning content
        contains_thinking, thinking_errors, processed_thinking = False, [], None
        if self.allow_reasoning:
            contains_thinking, thinking_errors, processed_thinking = self.thinking_parser(
                response, state, self.max_reasoning_chars
            )

        # Check if the response exceeds the maximum allowed characters per message.
        max_chars = state.get("max_chars_per_message", None)
        if max_chars is not None and len(response) > max_chars:
            errors.append(
                "Response exceeds the maximum allowed characters per message."
            )

        # Use the message parser to check for and extract message content
        contains_message, message_errors, processed_message = self.message_parser(
            response, state
        )

        # Use the finalization parser to check for and extract finalization content
        (
            contains_finalization,
            finalization_errors,
            processed_finalization,
        ) = self.finalization_parser(response, state, **self.finalization_parser_kwargs)

        # Combine errors from all validations
        if strategize_errors:
            errors.extend(strategize_errors)
        if thinking_errors:
            errors.extend(thinking_errors)
        if message_errors:
            errors.extend(message_errors)
        if finalization_errors:
            errors.extend(finalization_errors)

        # Check for conflicting actions
        if contains_strategize and (contains_message or contains_finalization):
            # If it's the first move of the first round, strategize and message together are allowed
            if not (state["round_number"] == 0 and state["round_moves"].get(self.agent_name, 0) == 0 and contains_message):
                errors.append("You cannot send both a strategize and another action in one response.")
        
        if contains_message and contains_finalization:
            errors.append(
                "You cannot send both a message and a finalization in one response."
            )

        # Only require message/finalization if there's no strategize
        if not contains_strategize and not contains_message and not contains_finalization:
            errors.append(
                "You must send either a message or a finalization. You have sent nothing."
            )

        # Return results
        if errors:
            if len(errors) == 1:
                error_message = errors[0]
            else:
                error_message = "Errors:\n" + "\n".join(
                    f"{i+1}) {err}" for i, err in enumerate(errors)
                )
            return True, error_message, False, None, response

        # If this is only a strategize message (no message/finalization), we return a dummy message
        if contains_strategize and not contains_message and not contains_finalization:
            dummy_message = "[Strategizing...]"
            return False, "", False, dummy_message, response

        # If no errors, return the appropriate processed content
        if contains_finalization:
            return False, "", True, processed_finalization, response
        else:
            return False, "", False, processed_message, response

    def format_prompt(self, prompt, state):
        """
        Replaces placeholders in a prompt with actual values from the game state.
        """
        if prompt:
            if state.get("has_finalized"):
                other_agent_finalization = state.get("last_raw_response", "")
            else:
                other_agent_finalization = ""

            # Get the values for the current agent based on their role.
            values = state["role_values"][
                state["agent_to_role"][state["current_agent"]]
            ]

            if state.get("round_points") != []:
                last_round_points = state["round_points"][-1][
                    state["agent_to_role"][state["current_agent"]]
                ]
            else:
                last_round_points = 0

            # Get the coagent's last round points if available
            if state.get("round_points") != [] and len(state["round_points"][-1]) > 1:
                # Find the coagent's role
                coagent_role = None
                for role in state["round_points"][-1]:
                    if role != state["agent_to_role"][state["current_agent"]]:
                        coagent_role = role
                        break

                if coagent_role:
                    coagent_last_round_points = state["round_points"][-1][coagent_role]
                else:
                    coagent_last_round_points = 0
            else:
                coagent_last_round_points = 0

            # Retrieve message-related values from the state.
            remaining_msgs = state["messages_remaining"][self.agent_name]
            max_msgs = state.get("max_messages", 0)
            min_msgs = state.get("min_messages", 0)
            current_sent = state["round_messages"].get(self.agent_name, 0)

            # Format finalize samples using actual item names and agent's attribution keys
            items = state.get("items", [])

            # Get the appropriate keys for this agent from attribution_map or use defaults
            i_take_key = "i_take"
            other_takes_key = "other_takes"

            if self.finalization_parser_kwargs:
                attribution_map = self.finalization_parser_kwargs.get(
                    "attribution_map", {}
                )
                i_take_key = attribution_map.get(self.agent_name, {}).get(
                    "i_take_key", "i_take"
                )
                other_takes_key = attribution_map.get(self.agent_name, {}).get(
                    "other_takes_key", "other_takes"
                )
            # import pdb; pdb.set_trace()

            finalize_sample = "{"
            finalize_sample += f'  "{i_take_key}": {{'
            item_entries = []
            for item in items:
                item_entries.append(f'    "{item}": x')
            finalize_sample += ",".join(item_entries)
            finalize_sample += "  },"

            finalize_sample += f'  "{other_takes_key}": {{'
            item_entries = []
            for item in items:
                item_entries.append(f'    "{item}": y')
            finalize_sample += ",".join(item_entries)
            finalize_sample += "  }}"

            last_round_coagent_values = {}
            # -------------------------------------------
            # New logic: Compute last round breakdown details for new_round_prompt_with_values.
            #
            # If archived round info exists, then check if an agreement was reached.
            # If no agreement was reached, fill with "0 points, since no agreement was reached".
            # If agreement was reached, compute detailed breakdown per item.
            if (
                state.get("round_agent_roles")
                and state.get("round_agreements_reached")
                and len(state["round_agent_roles"]) > 0
                and len(state["round_agreements_reached"]) > 0
            ):
                last_agreement = state["round_agreements_reached"][-1]
                last_arch_roles = state["round_agent_roles"][
                    -1
                ]  # mapping: agent -> role for that round
                # Determine current agent's role in the last round
                my_role = last_arch_roles.get(self.agent_name, None)
                # Determine the other agent's name and role
                other_agent_name, other_role = None, None
                for p, role in last_arch_roles.items():
                    if p != self.agent_name:
                        other_agent_name = p
                        other_role = role
                        break

                # Get the last round values
                last_round_values = state["round_values"][
                    -1
                ]  # mapping: role -> values dict
                # Extract coagent values outside the agreement check
                last_round_coagent_values = last_round_values.get(other_role, {})

                if not last_agreement:
                    last_round_points_computed = (
                        "0 points, since no agreement was reached"
                    )
                    coagent_last_round_points_computed = (
                        "0 points, since no agreement was reached"
                    )
                else:
                    # Retrieve last round's finalizations
                    last_round_finalizations = state["round_finalizations"][
                        -1
                    ]  # mapping: role -> finalization dict
                    # Compute detailed breakdown for current agent:
                    total_my = 0
                    details_my = []
                    for item in items:
                        my_qty = last_round_finalizations.get(my_role, {}).get(item, 0)
                        my_val = last_round_values.get(my_role, {}).get(item, 0)
                        product_my = my_qty * my_val
                        total_my += product_my
                        details_my.append(
                            f"{my_val} per {item} x {my_qty} = {product_my}"
                        )
                    last_round_points_computed = (
                        "; ".join(details_my) + f" | Total: {total_my} points"
                    )
                    # Compute detailed breakdown for the coagent:
                    total_other = 0
                    details_other = []
                    for item in items:
                        other_qty = last_round_finalizations.get(other_role, {}).get(
                            item, 0
                        )
                        other_val = last_round_values.get(other_role, {}).get(item, 0)
                        product_other = other_qty * other_val
                        total_other += product_other
                        details_other.append(
                            f"{other_val} per {item} x {other_qty} = {product_other}"
                        )
                    coagent_last_round_points_computed = (
                        "; ".join(details_other) + f" | Total: {total_other} points"
                    )

            else:
                last_round_points_computed = "0 points, since no agreement was reached"
                coagent_last_round_points_computed = (
                    "0 points, since no agreement was reached"
                )

            # -------------------------------------------

            # After computing last_round_points_computed and coagent_last_round_points_computed, add cumulative points calculations based on historical rounds.
            rounds_roles = state.get("round_agent_roles", [])
            rounds_points = state.get("round_points", [])
            cumulative_your_points = 0
            cumulative_coagent_points = 0
            for mapping, rp in zip(rounds_roles, rounds_points):
                if self.agent_name in mapping:
                    your_role = mapping[self.agent_name]
                    cumulative_your_points += rp.get(your_role, 0)
                    # Determine the coagent's name from the round mapping
                    coagent_names = [p for p in mapping if p != self.agent_name]
                    if coagent_names:
                        cp = coagent_names[0]
                        cp_role = mapping[cp]
                        cumulative_coagent_points += rp.get(cp_role, 0)

            # Get the coagent's name for prompt replacement
            coagent_name = ""
            if state.get("agent_to_role"):
                # Find the coagent's name from the current state
                for agent_name, role in state["agent_to_role"].items():
                    if agent_name != self.agent_name:
                        coagent_name = agent_name
                        break

            return (
                prompt.replace(
                    "{rounds_per_game}", str(state.get("rounds_per_game", ""))
                )
                .replace("{last_round_points}", str(last_round_points))
                .replace("{agent_name}", self.agent_name)
                .replace("{coagent_name}", coagent_name)
                .replace("{last_round_points_computed}", last_round_points_computed)
                .replace("{coagent_last_round_points}", str(coagent_last_round_points))
                .replace(
                    "{coagent_last_round_points_computed_other}",
                    coagent_last_round_points_computed,
                )
                .replace("{last_round_coagent_values}", str(last_round_coagent_values))
                .replace("{current_round}", str(state.get("current_round", "")))
                .replace("{nb_rounds}", str(state["round_number"] + 1))
                .replace("{quantities}", str(state.get("quantities", "")))
                .replace("{values}", str(values))
                .replace("{items}", str(items))
                .replace("{max_reasoning_chars}", str(self.max_reasoning_chars))
                .replace("{max_strategize_chars}", str(self.max_strategize_chars))
                .replace("{max_messages}", str(max_msgs))
                .replace("{min_messages}", str(min_msgs))
                .replace(
                    "{max_chars_per_message}",
                    str(state.get("max_chars_per_message", "")),
                )
                .replace("{max_errors}", str(self.max_errors))
                .replace("{last_message}", str(state.get("last_raw_response", "")))
                .replace("{last_raw_response}", str(state.get("last_raw_response", "")))
                .replace(
                    "{last_processed_response}",
                    str(state.get("last_processed_response", "")),
                )
                .replace("{other_agent_finalization}", str(other_agent_finalization))
                .replace(
                    "{remaining_messages}",
                    f"Minimum Messages: {min_msgs}, Maximum Messages: {max_msgs}, Current Number Sent: {current_sent}",
                )
                .replace("{finalize_sample}", finalize_sample)
                .replace(
                    "{finalize_sample_i_take}",
                    f'"{i_take_key}": '
                    + "{"
                    + ", ".join([f'"{item}": x' for item in items])
                    + "}",
                )
                .replace(
                    "{finalize_sample_other}",
                    f'"{other_takes_key}": '
                    + "{"
                    + ", ".join([f'"{item}": y' for item in items])
                    + "}",
                )
                .replace("{cumulative_round_points_your}", str(cumulative_your_points))
                .replace(
                    "{cumulative_round_points_coagent}", str(cumulative_coagent_points)
                )
                .replace("{strategize_content}", str(self.strategize_content or ""))
            )
        return ""

    def get_chat_history(self):
        return self.chat_history

    def add_to_chat_history(self, element: dict):
        self.chat_history.append(element)

    def new_round(self):
        """
        Resets round attributes.
        """
        self.retries = 0
        self.error_message = None

    def reset(self, checkpoint=None):
        """
        Resets the message history of the LLM agent or to a checkpoint if provided.

        Args:
            checkpoint (dict, optional): A dictionary containing the checkpoint state.
        """
        if checkpoint:
            self.load_checkpoint(checkpoint)
        else:
            self.retries = 0
            self.error_message = None
            self.chat_history = []
            self.augmented_chat_history = []
            self.has_strategized = False
            self.strategize_content = None
        return self.chat_history  # Return initial observation

    def load_checkpoint(self, checkpoint):
        """
        Loads the agent state from a checkpoint.

        Args:
            checkpoint (dict): A dictionary containing the checkpoint state.
        """
        self.__dict__.update(checkpoint)

    def export(self, path, state_history):
        game_stats = self.gather_statistics_func(
            state_history, self.conversation_history, **self.gather_statistics_func_args
        )
        self.set_chat_scores_func(
            self.conversation_history, **self.set_chat_scores_func_args
        )

        with open(path, "w") as f:
            json.dump(game_stats, f)
            json.dump(self.conversation_history, f)


def regular_message_parser(response, state):
    """
    Default message parser for DondAgent.

    Args:
        response (str): The full response from the agent.
        state (dict): The current state of the game.

    Returns:
        tuple: (contains_message, errors, processed_message)
    """
    errors = []

    # Find all message tags in the response
    message_tags = re.findall(r"<message>(.*?)</message>", response, flags=re.S)
    num_message_tags = len(message_tags)

    # Check if there is exactly one message tag
    if num_message_tags == 0:
        return False, [], None

    if num_message_tags > 1:
        errors.append(
            "Multiple <message> blocks detected. Please send only one message block."
        )
        return True, errors, None

    # Extract message content
    message_content = message_tags[0].strip()

    # Check if this response is a message and would exceed per-agent allowed messages.
    max_msgs = state.get("max_messages", None)
    agent_name = state.get("current_agent", "")
    agent_messages = state.get("round_messages", {}).get(agent_name, 0)

    if max_msgs is not None:
        if agent_messages == max_msgs:
            errors.append(
                "You must finalize because you reached the maximum number of messages!"
            )

    # Check that the message is not sent when the other agent has finalized
    if state.get("has_finalized", False):
        errors.append(
            "You must finalize your move because the other agent has finalized. Do not send a conversation message."
        )

    return True, errors, message_content


def regular_finalization_parser(response, state, attribution_map=None):
    """
    Simplified finalization parser for DondAgent.

    Args:
        response (str): The full response from the agent.
        state (dict): The current state of the game.
        attribution_map (dict, optional): Mapping of agent names to their attribution keys.

    Returns:
        tuple: (contains_finalization, errors, processed_finalization)
    """
    errors = []

    # Find all finalize tags in the response
    finalize_tags = re.findall(r"<finalize>(.*?)</finalize>", response, flags=re.S)

    # Check if there is a finalize tag
    if not finalize_tags:
        return False, [], None

    # Extract finalization content from the first tag
    finalization_content = finalize_tags[0].strip()

    # Get current agent name and other agent names
    current_agent = state.get("current_agent", "")
    agents = state.get("agents", [])
    other_agent = next((agent for agent in agents if agent != current_agent), "")

    try:
        # Check 1: Proper formatting
        finalization_json = json.loads(finalization_content)
        if not isinstance(finalization_json, dict):
            errors.append("The finalization is not properly formatted.")
            return True, errors, None

        # Format 1: Agent name keys format - {current_agent: {...}, other_agent: {...}}
        if current_agent in finalization_json and other_agent in finalization_json:
            my_items = finalization_json.get(current_agent, {})
            other_items = finalization_json.get(other_agent, {})

            if not isinstance(my_items, dict) or not isinstance(other_items, dict):
                errors.append("The finalization is not properly formatted.")
                return True, errors, None

            # Check 2: Consistency with quantities
            expected_items = set(state.get("items", []))
            expected_item_quantities = state.get("quantities", {})

            for item in expected_items:
                if item in my_items and item in other_items:
                    if not (
                        isinstance(my_items.get(item), int)
                        and isinstance(other_items.get(item), int)
                    ):
                        errors.append("The finalization is not properly formatted.")
                        return True, errors, None

                    if my_items.get(item, 0) + other_items.get(
                        item, 0
                    ) != expected_item_quantities.get(item, 0):
                        errors.append(
                            "The quantities don't sum up to the total quantities available."
                        )
                        return True, errors, None

            # Return what the current agent takes for themselves
            return True, errors, my_items

        # Format 2: i_take/other_takes format - backward compatibility
        i_take_key = "i_take"
        other_takes_key = "other_takes"

        if attribution_map and current_agent in attribution_map:
            agent_keys = attribution_map[current_agent]
            i_take_key = agent_keys.get("i_take_key", "i_take")
            other_takes_key = agent_keys.get("other_takes_key", "other_takes")

        i_take = finalization_json.get(i_take_key, {})
        other_takes = finalization_json.get(other_takes_key, {})

        if not isinstance(i_take, dict) or not isinstance(other_takes, dict):
            errors.append("The finalization is not properly formatted.")
            return True, errors, None

        # Check 2: Consistency with quantities
        expected_items = set(state.get("items", []))
        expected_item_quantities = state.get("quantities", {})

        for item in expected_items:
            if item in i_take and item in other_takes:
                if not (
                    isinstance(i_take.get(item), int)
                    and isinstance(other_takes.get(item), int)
                ):
                    errors.append("The finalization is not properly formatted.")
                    return True, errors, None

                if i_take.get(item, 0) + other_takes.get(
                    item, 0
                ) != expected_item_quantities.get(item, 0):
                    errors.append(
                        "The quantities don't sum up to the total quantities available."
                    )
                    return True, errors, None

        # Return what the current agent takes for themselves
        return True, errors, i_take

    except json.JSONDecodeError:
        errors.append("The finalization is not properly formatted.")
        return True, errors, None


def accept_reject_finalization_parser(response, state, attribution_map=None):
    """
    A finalization parser that uses regular_parser for the first finalization,
    but expects 'accept' or 'reject' for subsequent finalizations.

    Args:
        response (str): The full response from the agent.
        state (dict): The current state of the game.
        attribution_map (dict, optional): Mapping of agent names to their attribution keys.

    Returns:
        tuple: (contains_finalization, errors, processed_finalization)
        processed_finalization is a dict of items for first finalization and accept,
        or simply "reject_flag" string for rejection.
    """
    errors = []

    # Find all finalize tags in the response
    finalize_tags = re.findall(r"<finalize>(.*?)</finalize>", response, flags=re.S)

    # Check if there is a finalize tag
    if not finalize_tags:
        return False, [], None

    # Extract finalization content from the first tag
    finalization_content = finalize_tags[0].strip()

    # Check if this is the first finalization or a response to a previous finalization
    agent_name = state.get("current_agent", "")
    is_first_finalization = True

    # Check if there's a previous finalization from the other agent
    if state.get("has_finalized", False):
        is_first_finalization = False

    # If it's the first finalization, use the regular parser
    if is_first_finalization:
        return regular_finalization_parser(response, state, attribution_map)

    # For subsequent finalizations, expect 'accept' or 'reject'
    if finalization_content.lower() not in ["accept", "reject"]:
        errors.append(
            "For responding to a finalization, you must either <finalize>accept</finalize> or <finalize>reject</finalize>."
        )
        return True, errors, None

    # Return based on the response
    if finalization_content.lower() == "accept":
        # For accept, we need to calculate what the current agent gets
        # based on what the other agent took
        other_agent_finalization = state.get("last_processed_response", {})
        quantities = state.get("quantities", {})
        items = state.get("items", [])

        # Create our finalization by subtracting what the other agent took from total quantities
        my_finalization = {}

        for item in items:
            # Get what the other agent took for this item
            other_took = other_agent_finalization.get(item, 0)
            # Get total quantity for this item
            total_qty = quantities.get(item, 0)
            # Calculate what I get
            my_finalization[item] = total_qty - other_took

        return True, errors, my_finalization
    else:  # reject
        # Simply return the "reject_flag" string to match the finalize method in DondEnv
        return True, errors, "reject_flag"


def regular_strategize_parser(response, state, has_strategized, max_strategize_chars):
    """
    Default strategize parser for DondAgent.

    Args:
        response (str): The full response from the agent.
        state (dict): The current state of the game.
        has_strategized (bool): Whether the agent has already provided a strategize block.
        max_strategize_chars (int): Maximum characters allowed in strategize block.

    Returns:
        tuple: (contains_strategize, errors, processed_strategize)
    """
    errors = []

    # Find all strategize tags in the response
    strategize_blocks = re.findall(r"<strategize>(.*?)</strategize>", response, flags=re.S)
    num_strategize_blocks = len(strategize_blocks)

    # Check if there is a strategize tag
    if num_strategize_blocks == 0:
        return False, [], None

    # Check if agent has already strategized
    if has_strategized:
        errors.append("You have already provided a strategize block. You cannot provide another one.")
        return True, errors, None

    # Check if there is exactly one strategize tag
    if num_strategize_blocks > 1:
        errors.append("Multiple <strategize> blocks detected. Please include only one strategize block.")
        return True, errors, None

    # Extract strategize content
    strategize_content = strategize_blocks[0].strip()

    # Check if strategize content exceeds max allowed chars
    if max_strategize_chars is not None and len(strategize_content) > max_strategize_chars:
        errors.append(
            f"The strategize section exceeds the maximum allowed characters "
            f"({len(strategize_content)} > {max_strategize_chars})."
        )
        return True, errors, None

    # Check that strategize is provided at the start of the game
    agent_name = state.get("current_agent", "")
    if state["round_number"] != 0 or state["round_moves"].get(agent_name, 0) != 0:
        errors.append("Strategize blocks can only be provided at the start of the game.")
        return True, errors, None

    return True, errors, strategize_content


def regular_thinking_parser(response, state, max_reasoning_chars):
    """
    Default thinking (reasoning) parser for DondAgent.

    Args:
        response (str): The full response from the agent.
        state (dict): The current state of the game.
        max_reasoning_chars (int): Maximum characters allowed in thinking blocks.

    Returns:
        tuple: (contains_thinking, errors, processed_thinking)
    """
    errors = []

    # Find all think tags in the response
    think_blocks = re.findall(r"<think>(.*?)</think>", response, flags=re.S)
    num_think_blocks = len(think_blocks)

    # Check if there is a think tag
    if num_think_blocks == 0:
        return False, [], None

    # Calculate total thinking characters
    total_thinking_chars = sum(len(block.strip()) for block in think_blocks)

    # Check if thinking content exceeds max allowed chars
    if max_reasoning_chars is not None and total_thinking_chars > max_reasoning_chars:
        errors.append(
            f"The reasoning section exceeds the maximum allowed reasoning characters "
            f"({total_thinking_chars} > {max_reasoning_chars})."
        )
        return True, errors, None

    # Combine all thinking blocks into one if needed
    processed_thinking = "\n\n".join(block.strip() for block in think_blocks)
    
    return True, errors, processed_thinking
