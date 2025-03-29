import json
import regex as re
import re
import json
from utils.common_imports import *


class DondAgent:
    def __init__(
        self,
        agent_name,
        allow_reasoning,
        max_errors,
        policy_id,
        value_function_id=None,
        max_reasoning_chars=None,
        intro_prompt=None,
        goal_prompt=None,
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
        time_to_send_message_prompt=None
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
        """
        self.agent_name = agent_name
        self.allow_reasoning = allow_reasoning
        self.max_errors = max_errors
        self.policy_id = policy_id
        self.value_function_id = value_function_id
        self.max_reasoning_chars = max_reasoning_chars

        self.intro_prompt = intro_prompt
        self.goal_prompt = goal_prompt
        self.first_round_prompt = first_round_prompt
        self.new_round_prompt = new_round_prompt
        self.agent_with_first_move_prompt = agent_with_first_move_prompt
        self.agent_with_second_move_prompt = agent_with_second_move_prompt
        self.received_message_prompt = received_message_prompt
        self.other_agent_finalized_prompt = other_agent_finalized_prompt

        # Set the new mechanics prompts.
        self.message_mechanics_prompt = message_mechanics_prompt
        self.finalization_mechanics_prompt = finalization_mechanics_prompt
        self.dond_version_specificities = dond_version_specificities  # New prompt for version specificities
        self.reasoning_mechanics_prompt = reasoning_mechanics_prompt
        self.time_to_finalize_prompt = time_to_finalize_prompt
        self.time_to_send_message_prompt = time_to_send_message_prompt

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
            is_error, error_message, is_finalization, processed_response = self.process_response(
                policy_output, state
            )

            # Add the model response to chat history
            model_response = {
                "role": "assistant",
                "content": policy_output,
                "is_error": is_error,
                "is_finalization": is_finalization,
                "round_nb": state["round_number"],
            }
            self.add_to_chat_history(model_response)

            # Handle errors
            if is_error:
                self.retries += 1
                self.error_message = error_message

                # Too many mistakes were made: force dummy message
                if self.retries > self.max_errors:
                    self.retries = 0
                    # If the policy output indicates a finalization move via the <finalize> tag,
                    # we keep the fallback as "-------". Otherwise (it was time to send a message),
                    # we generate a fallback message truncated to the maximum allowed characters.
                    fallback = "[ERROR]"
                    if policy_output and "<finalize>" not in policy_output:
                        max_chars = state.get("max_chars_per_message", 300)
                        fallback = policy_output[:max_chars]
                    action = (False, fallback)
                    return None, None, action, True, self.get_log_info()

                # Set error user message
                user_message = self.get_user_message(state, is_error, error_message)
                self.add_to_chat_history(user_message)

                # Return with a request for policy output again (done = False)
                return self.policy_id, self.chat_history, None, False, self.get_log_info()

            # Reset retries on successful processing
            self.retries = 0


            # Create the action to be sent to the environment
            action = (is_finalization, processed_response)

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
            "augmented_chat_history": self.augmented_chat_history
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
        if state["game_moves"].get(self.agent_name) == 0:
            user_message += self.format_prompt(self.intro_prompt, state)
            if self.message_mechanics_prompt:
                user_message += "\n\n" + self.format_prompt(self.message_mechanics_prompt, state)
            if self.finalization_mechanics_prompt:
                user_message += "\n\n" + self.format_prompt(self.finalization_mechanics_prompt, state)
            if self.dond_version_specificities:
                user_message += "\n\n" + self.format_prompt(self.dond_version_specificities, state)
            if self.allow_reasoning and self.reasoning_mechanics_prompt:
                user_message += "\n\n" + self.format_prompt(self.reasoning_mechanics_prompt, state)
            if self.goal_prompt:
                user_message += "\n\n" + self.format_prompt(self.goal_prompt, state)

        # If the current agent has not yet made any move in this round, add round instructions.
        if state["round_moves"].get(self.agent_name, 0) == 0:
            if state["round_number"] == 0 and self.first_round_prompt:
                user_message += "\n\n" + self.format_prompt(self.first_round_prompt, state)
            elif self.new_round_prompt:
                user_message += self.format_prompt(self.new_round_prompt, state)

        # Then add the appropriate message based on the finalization state and move count.
        if state["has_finalized"]:
            user_message += self.format_prompt(self.other_agent_finalized_prompt, state)
        elif state["last_message"] is None:
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
            user_message += "\n\n" + self.format_prompt(self.time_to_finalize_prompt, state)
        elif agent_messages < min_msgs and self.time_to_send_message_prompt:
            user_message += "\n\n" + self.format_prompt(self.time_to_send_message_prompt, state)

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
            tuple: (is_error, error_message, is_finalization, processed_response)
        """
        errors = []

        # Count the occurrences of valid tag blocks.
        message_tags = re.findall(r"<message>.*?</message>", response, flags=re.S)
        num_message_tags = len(message_tags)
        finalize_tags = re.findall(r"<finalize>.*?</finalize>", response, flags=re.S)
        num_finalize_tags = len(finalize_tags)

        # 2) Check that exactly one of <message>...</message> or <finalize>...</finalize> is present.
        has_message = num_message_tags == 1
        has_finalization = num_finalize_tags == 1

        # New check: If the co-agent has finalized and our agent sends a message instead of a finalization, raise an error.
        if state.get("has_finalized", False) and has_message:
            errors.append("You must finalize your move because the other agent has finalized. Do not send a conversation message.")

        if num_message_tags > 1:
            errors.append("Multiple <message> blocks detected. Please send only one message block.")
        if num_finalize_tags > 1:
            errors.append("Multiple <finalize> blocks detected. Please send only one finalization block.")

        if has_message and has_finalization:
            errors.append("You cannot send both a message and a finalization in one response.")

        if not has_message and not has_finalization:
            errors.append("You must send either a message or a finalization. You have sent nothing.")

        # 2.5) Check if this response is a message and would exceed per-agent allowed messages.
        max_msgs = state.get("max_messages", None)
        agent_messages = state.get("round_messages", {}).get(self.agent_name, 0)
        if max_msgs is not None and has_message:
            if agent_messages == max_msgs:
                errors.append("You must finalize because you reached the maximum number of messages!")

        # NEW: Check that the minimum number of messages has been sent before finalizing.
        min_msgs = state.get("min_messages", None)
        if min_msgs is not None and has_finalization:
            if agent_messages < min_msgs:
                errors.append(
                    f"You must send at least {min_msgs} message(s) before finalizing. You have sent {agent_messages}."
                )

        # 3) Check for excessive content outside valid tags (<think>, <message>, <finalize>).
        total_length = len(response)
        valid_tags = ["think", "message", "finalize"]
        total_tag_content_length = 0
        for tag in valid_tags:
            pattern = rf"<{tag}>.*?</{tag}>"
            matches = re.findall(pattern, response, flags=re.S)
            for match in matches:
                total_tag_content_length += len(match)
        outside_length = total_length - total_tag_content_length
        if outside_length > 5:
            errors.append("Excessive content outside of valid tags.")

        # 3.6) Check if the content inside <think> tags exceeds the allowed reasoning characters.
        think_blocks = re.findall(r"<think>(.*?)</think>", response, flags=re.S)
        if think_blocks:
            total_thinking_chars = sum(len(block.strip()) for block in think_blocks)
            if self.max_reasoning_chars is not None and total_thinking_chars > self.max_reasoning_chars:
                errors.append(
                    f"The reasoning section exceeds the maximum allowed reasoning characters "
                    f"({total_thinking_chars} > {self.max_reasoning_chars})."
                )

        # 3.5) Check if the response exceeds the maximum allowed characters per message.
        max_chars = state.get("max_chars_per_message", None)
        if max_chars is not None and len(response) > max_chars:
            errors.append("Response exceeds the maximum allowed characters per message.")

        # 4) Process finalization if present.
        if has_finalization:
            finalization_content = response.split("<finalize>", 1)[1].split("</finalize>", 1)[0].strip()
            try:
                finalization_json = json.loads(finalization_content)
                if not isinstance(finalization_json, dict):
                    errors.append("The content within <finalize> is not a valid dictionary.")
                    i_take = None
                    other_agent_gets = None
                else:
                    i_take = finalization_json.get("i_take", {})
                    other_agent_gets = finalization_json.get("other_agent_gets", {})

                if not isinstance(i_take, dict) or not isinstance(other_agent_gets, dict):
                    errors.append('"i_take" and "other_agent_gets" must be dictionaries.')
                else:
                    expected_items = set(state.get("items", []))
                    expected_item_quantities = state.get("quantities", {})

                    # Validate that the keys exactly match the expected items.
                    if set(i_take.keys()) != expected_items:

                        missing = expected_items - set(i_take.keys())
                        extra = set(i_take.keys()) - expected_items
                        error_str = "Invalid keys in 'i_take':"

                        if missing:
                            error_str += f" Missing keys: {', '.join(missing)}."
                        if extra:
                            error_str += f" Unexpected keys: {', '.join(extra)}."

                        errors.append(error_str)

                    if set(other_agent_gets.keys()) != expected_items:

                        missing = expected_items - set(other_agent_gets.keys())
                        extra = set(other_agent_gets.keys()) - expected_items
                        error_str = "Invalid keys in 'other_agent_gets':"

                        if missing:
                            error_str += f" Missing keys: {', '.join(missing)}."
                        if extra:
                            error_str += f" Unexpected keys: {', '.join(extra)}."
                        errors.append(error_str)

                    i_take_valid_items = set(i_take.keys()) & expected_items
                    other_agent_gets_valid_items = set(other_agent_gets.keys()) & expected_items

                    # Verify that every value for each key is an integer and sums to total quantities.
                    for item in expected_items:
                        if item in i_take_valid_items and item in other_agent_gets_valid_items:
                            is_i_take_int = isinstance(i_take.get(item), int)
                            is_other_agent_gets_int = isinstance(other_agent_gets.get(item), int)

                            if not is_i_take_int:
                                errors.append(f'Value of "{item}" in "i_take" must be an integer.')

                            if not is_other_agent_gets_int:
                                errors.append(f'Value of "{item}" in "other_agent_gets" must be an integer.')

                            if (
                                is_i_take_int
                                and is_other_agent_gets_int
                                and i_take.get(item, 0) + other_agent_gets.get(item, 0)
                                != expected_item_quantities.get(item, 0)
                            ):
                                errors.append(f'Total {item} divided should sum to {expected_item_quantities.get(item, 0)}.')

            except json.JSONDecodeError:
                errors.append("The content within <finalize> is not valid JSON.")

        # 5) Return results: if errors exist, return an error message.
        if errors:
            if len(errors) == 1:
                error_message = errors[0]
            else:
                error_message = "Errors:\n" + "\n".join(f"{i+1}) {err}" for i, err in enumerate(errors))
            return True, error_message, False, None

        if has_finalization:
            return False, "", True, {"i_take": i_take, "other_agent_gets": other_agent_gets}
        if has_message:
            # Extract using our earlier found list.
            message_content = message_tags[0].split("<message>", 1)[1].split("</message>", 1)[0].strip()
            return False, "", False, message_content

        return True, "Unknown error: Invalid response format.", False, None

    def format_prompt(self, prompt, state):
        """
        Replaces placeholders in a prompt with actual values from the game state.
        """
        if prompt:
            if state.get("has_finalized"):
                other_agent_finalization = state.get("last_message", "")
            else:
                other_agent_finalization = ""

            # Get the values for the current agent based on their role.
            values = state["role_values"][state["agent_to_role"][state["current_agent"]]]

            if state.get("round_points") != []:
                last_round_points = state['round_points'][-1][state["agent_to_role"][state["current_agent"]]]
            else:
                last_round_points = 0

            # Retrieve message-related values from the state.
            remaining_msgs = state['messages_remaining'][self.agent_name]
            max_msgs = state.get("max_messages", 0)
            min_msgs = state.get("min_messages", 0)
            current_sent = state["round_messages"].get(self.agent_name, 0)

            # Format finalize samples using actual item names
            items = state.get("items", [])
            finalize_sample_i_take = ", ".join([f'"{item}": x' for item in items])
            finalize_sample_other = ", ".join([f'"{item}": y' for item in items])

            # -------------------------------------------
            # New logic: Compute last round breakdown details for new_round_prompt_with_values.
            #
            # If archived round info exists, then check if an agreement was reached.
            # If no agreement was reached, fill with "0 points, since no agreement was reached".
            # If agreement was reached, compute detailed breakdown per item.
            if (state.get("round_agent_roles") and state.get("round_agreements_reached") and
                len(state["round_agent_roles"]) > 0 and len(state["round_agreements_reached"]) > 0):
                last_agreement = state["round_agreements_reached"][-1]
                last_arch_roles = state["round_agent_roles"][-1]  # mapping: agent -> role for that round
                # Determine current agent's role in the last round
                my_role = last_arch_roles.get(self.agent_name, None)
                # Determine the other agent's name and role
                other_agent_name, other_role = None, None
                for p, role in last_arch_roles.items():
                    if p != self.agent_name:
                        other_agent_name = p
                        other_role = role
                        break
                if not last_agreement:
                    last_round_points_computed = "0 points, since no agreement was reached"
                    coagent_last_round_points_computed = "0 points, since no agreement was reached"
                else:
                    # Retrieve last round's finalizations and values
                    last_round_finalizations = state["round_finalizations"][-1]  # mapping: role -> finalization dict
                    last_round_values = state["round_values"][-1]  # mapping: role -> values dict
                    # Compute detailed breakdown for current agent:
                    total_my = 0
                    details_my = []
                    for item in items:
                        my_qty = last_round_finalizations.get(my_role, {}).get(item, 0)
                        my_val = last_round_values.get(my_role, {}).get(item, 0)
                        product_my = my_qty * my_val
                        total_my += product_my
                        details_my.append(f"{my_val} per {item} x {my_qty} = {product_my}")
                    last_round_points_computed = "; ".join(details_my) + f" | Total: {total_my} points"
                    # Compute detailed breakdown for the coagent:
                    total_other = 0
                    details_other = []
                    for item in items:
                        other_qty = last_round_finalizations.get(other_role, {}).get(item, 0)
                        other_val = last_round_values.get(other_role, {}).get(item, 0)
                        product_other = other_qty * other_val
                        total_other += product_other
                        details_other.append(f"{other_val} per {item} x {other_qty} = {product_other}")
                    coagent_last_round_points_computed = "; ".join(details_other) + f" | Total: {total_other} points"
            else:
                last_round_points_computed = "0 points, since no agreement was reached"
                coagent_last_round_points_computed = "0 points, since no agreement was reached"
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

            return prompt.replace("{rounds_per_game}", str(state.get("rounds_per_game", ""))) \
                        .replace("{last_round_points}", str(last_round_points)) \
                        .replace("{last_round_points_computed}", last_round_points_computed) \
                        .replace("{coagent_last_round_points_computed_other}", coagent_last_round_points_computed) \
                        .replace("{current_round}", str(state.get("current_round", ""))) \
                        .replace("{nb_rounds}", str(state["round_number"] + 1)) \
                        .replace("{quantities}", str(state.get("quantities", ""))) \
                        .replace("{values}", str(values)) \
                        .replace("{items}", str(items)) \
                        .replace("{max_reasoning_chars}", str(self.max_reasoning_chars)) \
                        .replace("{max_messages}", str(max_msgs)) \
                        .replace("{min_messages}", str(min_msgs)) \
                        .replace("{max_chars_per_message}", str(state.get("max_chars_per_message", ""))) \
                        .replace("{max_errors}", str(self.max_errors)) \
                        .replace("{last_message}", str(state.get("last_message", ""))) \
                        .replace("{other_agent_finalization}", str(other_agent_finalization)) \
                        .replace("{remaining_messages}", \
                                 f"Minimum Messages: {min_msgs}, Maximum Messages: {max_msgs}, Current Number Sent: {current_sent}") \
                        .replace("{finalize_sample_i_take}", finalize_sample_i_take) \
                        .replace("{finalize_sample_other}", finalize_sample_other) \
                        .replace("{cumulative_round_points_your}", str(cumulative_your_points)) \
                        .replace("{cumulative_round_points_coagent}", str(cumulative_coagent_points))
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
        return self.chat_history  # Return initial observation

    def load_checkpoint(self, checkpoint):
        """
        Loads the agent state from a checkpoint.

        Args:
            checkpoint (dict): A dictionary containing the checkpoint state.
        """
        self.__dict__.update(checkpoint)

    def export(self, path, state_history):
        game_stats = self.gather_statistics_func(state_history, self.conversation_history, **self.gather_statistics_func_args)
        self.set_chat_scores_func(self.conversation_history, **self.set_chat_scores_func_args)

        with open(path, "w") as f:
            json.dump(game_stats, f)
            json.dump(self.conversation_history, f)





