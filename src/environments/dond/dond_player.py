import json
import regex as re
import re
import json
from utils.common_imports import *


class DondPlayerHandler:
    def __init__(
        self,
        player_name,
        allow_reasoning,
        max_errors,
        mod_adpt_id,
        max_reasoning_chars,
        intro_prompt,
        goal_prompt,
        first_round_prompt,
        new_round_prompt,
        player_with_first_move_prompt,
        received_message_prompt,
        other_player_finalized_prompt,
        message_mechanics_prompt=None,
        dond_version_specificities=None,
        reasoning_mechanics_prompt=None
    ):
        """
        Initializes the DondPlayerHandler.

        Args:
            player_name (str): The name of the player.
            allow_reasoning (bool): Whether reasoning is allowed.
            max_errors (int): Maximum number of retries allowed.
            mod_adpt_id (str): The model adapter id to use.
            max_reasoning_chars (int): Maximum reasoning characters allowed.
            intro_prompt (str): Prompt for the game introduction.
            goal_prompt (str): Prompt for the player's goal.
            first_round_prompt (str): Prompt for the first round.
            new_round_prompt (str): Prompt for the new round.
            player_with_first_move_prompt (str): Prompt when the player is assigned the first move.
            received_message_prompt (str): Prompt when a message is received from the other player.
            other_player_finalized_prompt (str): Prompt to indicate that the other player has finalized.
            message_mechanics_prompt (str, optional): Instructions for message mechanics.
            dond_version_specificities (str, optional): DOND-specific game instructions.
            reasoning_mechanics_prompt (str, optional): Instructions for reasoning mechanics.
        """
        self.player_name = player_name
        self.allow_reasoning = allow_reasoning
        self.max_errors = max_errors
        self.mod_adpt_id = mod_adpt_id
        self.max_reasoning_chars = max_reasoning_chars

        self.intro_prompt = intro_prompt
        self.goal_prompt = goal_prompt
        self.first_round_prompt = first_round_prompt
        self.new_round_prompt = new_round_prompt
        self.player_with_first_move_prompt = player_with_first_move_prompt
        self.received_message_prompt = received_message_prompt
        self.other_player_finalized_prompt = other_player_finalized_prompt

        # Set the new mechanics prompts.
        self.message_mechanics_prompt = message_mechanics_prompt
        self.dond_version_specificities = dond_version_specificities  # New prompt for version specificities
        self.reasoning_mechanics_prompt = reasoning_mechanics_prompt

        self.game_id = None  # ID of the player in the game
        self.reset()

    def set_usr_message(self, state):
        """
        Constructs a user message based on the current game state.

        Args:
            state (dict): The current state of the game.
        """
        user_message = ""

        if self.error_message:
            user_message = self.error_message
            usr_prompt = {
                "role": "user",
                "content": user_message,
                "is_error": True,
                "round_nb": state["round_number"],
            }
            self.add_to_chat_history(usr_prompt)
            self.error_message = None
            return

        # Use the new move information to decide on the prompts.
        # If the current player has not yet made any move in the game, prepend the introductory prompts.
        if state["game_moves"].get(self.player_name, 0) == 0:
            user_message += self.format_prompt(self.intro_prompt, state)
            if self.message_mechanics_prompt:
                user_message += "\n\n" + self.format_prompt(self.message_mechanics_prompt, state)
            if self.dond_version_specificities:
                user_message += "\n\n" + self.format_prompt(self.dond_version_specificities, state)
            if self.allow_reasoning and self.reasoning_mechanics_prompt:
                user_message += "\n\n" + self.format_prompt(self.reasoning_mechanics_prompt, state)
            if self.goal_prompt:
                user_message += "\n\n" + self.format_prompt(self.goal_prompt, state)

        # If the current player has not yet made any move in this round, add round instructions.
        if state["round_moves"].get(self.player_name, 0) == 0:
            if state["round_number"] == 0 and self.first_round_prompt:
                user_message += "\n\n" + self.format_prompt(self.first_round_prompt, state)
            elif self.new_round_prompt:
                user_message += self.format_prompt(self.new_round_prompt, state)

        # Then add the appropriate message based on the finalization state.
        if state["has_finalized"]:
            user_message += self.format_prompt(self.other_player_finalized_prompt, state)
        elif state["last_message"] is None:
            user_message += self.player_with_first_move_prompt
        else:
            user_message += self.format_prompt(self.received_message_prompt, state)

        usr_prompt = {
            "role": "user",
            "content": user_message,
            "is_error": False,
            "round_nb": state["round_number"],
        }
        self.add_to_chat_history(usr_prompt)

    def process_response(self, response, state):
        """
        Validates and extracts content from the response of the LLM player.

        Args:
            response (str): The response from the LLM player.
            state (dict): The current state of the game.

        scores:
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

        if num_message_tags > 1:
            errors.append("Multiple <message> blocks detected. Please send only one message block.")
        if num_finalize_tags > 1:
            errors.append("Multiple <finalize> blocks detected. Please send only one finalization block.")

        if has_message and has_finalization:
            errors.append("You cannot send both a message and a finalization in one response.")

        # 2.5) Check if this response is a message and would exceed per-player allowed messages.
        max_msgs = state.get("max_messages", None)
        player_messages = state.get("round_messages", {}).get(self.player_name, 0)
        if max_msgs is not None and has_message:
            if player_messages == max_msgs:
                errors.append("You must finalize because you reached the maximum number of messages!")

        # NEW: Check that the minimum number of messages has been sent before finalizing.
        min_msgs = state.get("min_messages", None)
        if min_msgs is not None and has_finalization:
            if player_messages < min_msgs:
                errors.append(
                    f"You must send at least {min_msgs} message(s) before finalizing. You have sent {player_messages}."
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
                    other_player_gets = None
                else:
                    i_take = finalization_json.get("i_take", {})
                    other_player_gets = finalization_json.get("other_player_gets", {})
                if not isinstance(i_take, dict) or not isinstance(other_player_gets, dict):
                    errors.append('"i_take" and "other_player_gets" must be dictionaries.')
                else:
                    # Validate that the keys exactly match the expected items.
                    expected_items = set(state.get("items", []))
                    if set(i_take.keys()) != expected_items:
                        missing = expected_items - set(i_take.keys())
                        extra = set(i_take.keys()) - expected_items
                        error_str = "Invalid keys in 'i_take':"
                        if missing:
                            error_str += f" Missing keys: {', '.join(missing)}."
                        if extra:
                            error_str += f" Unexpected keys: {', '.join(extra)}."
                        errors.append(error_str)
                    if set(other_player_gets.keys()) != expected_items:
                        missing = expected_items - set(other_player_gets.keys())
                        extra = set(other_player_gets.keys()) - expected_items
                        error_str = "Invalid keys in 'other_player_gets':"
                        if missing:
                            error_str += f" Missing keys: {', '.join(missing)}."
                        if extra:
                            error_str += f" Unexpected keys: {', '.join(extra)}."
                        errors.append(error_str)
                    # Verify that every value for each key is an integer.
                    for item in expected_items:
                        if not isinstance(i_take.get(item), int):
                            errors.append(f'Value of "{item}" in "i_take" must be an integer.')
                        if not isinstance(other_player_gets.get(item), int):
                            errors.append(f'Value of "{item}" in "other_player_gets" must be an integer.')
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
            return False, "", True, {"i_take": i_take, "other_player_gets": other_player_gets}
        if has_message:
            # Extract using our earlier found list.
            message_content = message_tags[0].split("<message>", 1)[1].split("</message>", 1)[0].strip()
            return False, "", False, message_content

        return True, "Unknown error: Invalid response format.", False, None

    def initialize_prompts(self, prompts):
        """
        Initializes and formats prompts from the configuration.

        Args:
            prompts (dict): Dictionary containing various prompts.

        scores:
            dict: Formatted prompts with placeholders replaced.
        """
        formatted_prompts = {}
        for key, prompt in prompts.items():
            if isinstance(prompt, list):
                formatted_prompts[key] = self.format_prompt_list(prompt)
            else:
                formatted_prompts[key] = self.format_prompt(prompt)
        return formatted_prompts

    def format_prompt(self, prompt, state):
        """
        Replaces placeholders in a prompt with actual values from the game state.
        """
        if prompt:
            if state.get("has_finalized"):
                other_player_finalization = state.get("last_message", "")
            else:
                other_player_finalization = ""

            # Get the values for the current player based on their role.
            values = state["role_values"][state["player_to_role"][state["current_player"]]]

            if state.get("round_points") != []:
                last_round_points = state['round_points'][-1][state["player_to_role"][state["current_player"]]]
            else:
                last_round_points = 0

            # Retrieve message-related values from the state.
            remaining_msgs = state['messages_remaining'][self.player_name]
            max_msgs = state.get("max_messages", 0)
            min_msgs = state.get("min_messages", 0)
            current_sent = state["round_messages"].get(self.player_name, 0)

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
            if (state.get("round_player_roles") and state.get("round_agreements_reached") and
                len(state["round_player_roles"]) > 0 and len(state["round_agreements_reached"]) > 0):
                last_agreement = state["round_agreements_reached"][-1]
                last_arch_roles = state["round_player_roles"][-1]  # mapping: player -> role for that round
                # Determine current player's role in the last round
                my_role = last_arch_roles.get(self.player_name, None)
                # Determine the other player's name and role
                other_player_name, other_role = None, None
                for p, role in last_arch_roles.items():
                    if p != self.player_name:
                        other_player_name = p
                        other_role = role
                        break
                if not last_agreement:
                    last_round_points_computed = "0 points, since no agreement was reached"
                    coplayer_last_round_points_computed = "0 points, since no agreement was reached"
                else:
                    # Retrieve last round's finalizations and values
                    last_round_finalizations = state["round_finalizations"][-1]  # mapping: role -> finalization dict
                    last_round_values = state["round_values"][-1]  # mapping: role -> values dict
                    # Compute detailed breakdown for current player:
                    total_my = 0
                    details_my = []
                    for item in items:
                        my_qty = last_round_finalizations.get(my_role, {}).get(item, 0)
                        my_val = last_round_values.get(my_role, {}).get(item, 0)
                        product_my = my_qty * my_val
                        total_my += product_my
                        details_my.append(f"{my_val} per {item} x {my_qty} = {product_my}")
                    last_round_points_computed = "; ".join(details_my) + f" | Total: {total_my} points"
                    # Compute detailed breakdown for the coplayer:
                    total_other = 0
                    details_other = []
                    for item in items:
                        other_qty = last_round_finalizations.get(other_role, {}).get(item, 0)
                        other_val = last_round_values.get(other_role, {}).get(item, 0)
                        product_other = other_qty * other_val
                        total_other += product_other
                        details_other.append(f"{other_val} per {item} x {other_qty} = {product_other}")
                    coplayer_last_round_points_computed = "; ".join(details_other) + f" | Total: {total_other} points"
            else:
                last_round_points_computed = "0 points, since no agreement was reached"
                coplayer_last_round_points_computed = "0 points, since no agreement was reached"
            # -------------------------------------------

            # After computing last_round_points_computed and coplayer_last_round_points_computed, add cumulative points calculations based on historical rounds.
            rounds_roles = state.get("round_player_roles", [])
            rounds_points = state.get("round_points", [])
            cumulative_your_points = 0
            cumulative_coplayer_points = 0
            for mapping, rp in zip(rounds_roles, rounds_points):
                if self.player_name in mapping:
                    your_role = mapping[self.player_name]
                    cumulative_your_points += rp.get(your_role, 0)
                    # Determine the coplayer's name from the round mapping
                    coplayer_names = [p for p in mapping if p != self.player_name]
                    if coplayer_names:
                        cp = coplayer_names[0]
                        cp_role = mapping[cp]
                        cumulative_coplayer_points += rp.get(cp_role, 0)

            return prompt.replace("{rounds_per_game}", str(state.get("rounds_per_game", ""))) \
                        .replace("{last_round_points}", str(last_round_points)) \
                        .replace("{last_round_points_computed}", last_round_points_computed) \
                        .replace("{coplayer_last_round_points_computed_other}", coplayer_last_round_points_computed) \
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
                        .replace("{other_player_finalization}", str(other_player_finalization)) \
                        .replace("{remaining_messages}", \
                                 f"Minimum Messages: {min_msgs}, Maximum Messages: {max_msgs}, Current Number Sent: {current_sent}") \
                        .replace("{{finalize_sample_i_take}}", finalize_sample_i_take) \
                        .replace("{{finalize_sample_other}}", finalize_sample_other) \
                        .replace("{cumulative_round_points_your}", str(cumulative_your_points)) \
                        .replace("{cumulative_round_points_coplayer}", str(cumulative_coplayer_points))
        return ""

    def get_chat_history(self):
        return self.chat_history

    def add_to_chat_history(self, element: dict):
        self.chat_history.append(element)

    def step(self, input):
        """
        Processes the response from the model and updates the game state.

        Args:
            action (str): The action to be taken.
            state (dict): The current state of the game.
            llm_output (str): The output from the language model.

        scores:
            tuple: A tuple containing:
                - observation (dict): The new state of the game.
                - reward (float): The reward obtained from the action.
                - done (bool): Whether the game is finished.
                - info (dict): Additional information.
        """
        state, info, llm_output = input
        # Initiate what will be returned
        processed_response = None
        send_to_game = False
        is_finalization = False

        # Process response. Check for errors.
        is_error, error_message, is_finalization, processed_response = self.process_response(
            llm_output, state
        )

        if is_error:
            self.retries += 1
            self.error_message = error_message
            # Too many mistakes were made
            if self.retries > self.max_errors:
                self.error_message = False
                processed_response = "-------"
                send_to_game = True
                self.retries = 0

        else:
            self.retries = 0
            send_to_game = True

        # Add raw response to chat_history
        model_response = {
            "role": "assistant",
            "content": llm_output,
            "is_error": is_error,
            "is_finalization": is_finalization,
            "round_nb": state["round_number"],
        }

        self.add_to_chat_history(model_response)

        action = (is_finalization, processed_response)
        player_state = None
        player_info = {"player_name": self.player_name, "chat_history": self.chat_history}

        return action, player_state, send_to_game, player_info

    def get_info(self):
        return {"player_name": self.player_name, "chat_history": self.chat_history}

    # Optional render method
    def render(self, mode='human'):
        """
        Renders the environment for visualization.
        """
        # Implement rendering logic if needed
        pass

    # Optional close method
    def close(self):
        """
        Cleans up resources when the environment is no longer needed.
        """
        # Implement cleanup logic if needed
        pass



    def new_round(self):
        """
        Resets round attributes.
        """
        self.retries = 0
        self.error_message = None

    def reset(self, checkpoint=None):
        """
        Resets the message history of the LLM player or to a checkpoint if provided.

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
        Loads the player state from a checkpoint.

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





