import json
import regex as re
import copy
# local imports
from environments.dond.dond_game import DondGame
import math
from statistics import mean
import numpy as np
import re
import json

class DondPlayerHandler:
    def __init__(
        self,
        player_name,
        allow_reasoning,
        max_retries,
        mod_adpt_id,
        max_reasoning_chars,
        max_messages,
        max_chars_per_message,
        intro_prompt,
        goal_prompt,
        first_round_prompt,
        new_round_prompt,
        player_with_first_move_prompt,
        received_message_prompt,
        other_player_finalized_prompt
    ):
        """
        Initializes the DondPlayerHandler.

        Args:
            player_name (str): The name of the player.
            allow_reasoning (bool): Whether reasoning is allowed.
            max_retries (int): Maximum number of retries allowed.
            mod_adpt_id (str): The model adapter id to use.
            max_reasoning_chars (int): Maximum reasoning characters allowed.
            max_messages (int): Maximum number of messages allowed.
            max_chars_per_message (int): Maximum characters per message.
            intro_prompt (str): Prompt for the game introduction.
            goal_prompt (str): Prompt for the player's goal.
            first_round_prompt (str): Prompt for the first round.
            new_round_prompt (str): Prompt for the new round.
            player_with_first_move_prompt (str): Prompt when the player is assigned the first move in a round.
            received_message_prompt (str): Prompt when a message is received from the other player.
            other_player_finalized_prompt (str): Prompt to indicate that the other player has finalized.
        """
        self.player_name = player_name
        self.allow_reasoning = allow_reasoning
        self.max_retries = max_retries
        self.mod_adpt_id = mod_adpt_id
        self.max_reasoning_chars = max_reasoning_chars
        self.max_messages = max_messages
        self.max_chars_per_message = max_chars_per_message

        self.intro_prompt = intro_prompt
        self.goal_prompt = goal_prompt
        self.first_round_prompt = first_round_prompt
        self.new_round_prompt = new_round_prompt
        self.player_with_first_move_prompt = player_with_first_move_prompt
        self.received_message_prompt = received_message_prompt
        self.other_player_finalized_prompt = other_player_finalized_prompt

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

        if state["is_new_game"]:
            user_message += self.format_prompt(self.intro_prompt, state)
            user_message += self.format_prompt(self.goal_prompt, state)

        if state["is_new_round"]:
            self.new_round()
            # Use the first round prompt if round_number == 0; otherwise use the new round prompt.
            if state["round_number"] == 0:
                user_message += self.format_prompt(self.first_round_prompt, state)
            else:
                user_message += self.format_prompt(self.new_round_prompt, state)

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

        Returns:
            tuple: A tuple containing:
                - is_error (bool): Indicates if there is an error.
                - error_message (str): The error message if there is an error, otherwise an empty string.
                - is_finalization (bool): Indicates if the response is a finalization.
                - processed_response (str or dict): The extracted message or finalization details.
        """
        errors = []

        # 1) Validate presence of <think> tags (if reasoning is allowed)
        if self.allow_reasoning is not False:
            if "<think>" not in response or "</think>" not in response:
                errors.append("Missing <think>...</think> tag.")

        # 2) Check if exactly one of <message>...</message> or <finalize>...</finalize> is present,
        #    and ensure finalization rules are followed
        has_message = "<message>" in response and "</message>" in response
        has_finalization = "<finalize>" in response and "</finalize>" in response

        if (state["turn"] > state["max_turns"] - 2) and not has_finalization:
            errors.append("You must finalize before the turn limit!")
        if has_message and has_finalization:
            errors.append("Response contains both <message> and <finalize>; only one is allowed.")
        elif not has_message and not has_finalization:
            errors.append("Response must contain either <message> or <finalize>.")
        if state["has_finalized"] and not has_finalization:
            errors.append("The other player has finalized; you must finalize as well.")

        # 3) Check for excessive content outside valid tags (<think>, <message>, <finalize>)
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

        # 4) Process finalization if present
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

        # 5) Return results: check for errors, otherwise return parsed content
        if errors:
            return True, "Errors: " + "; ".join(errors), False, None

        if has_finalization:
            return False, "", True, {"i_take": i_take, "other_player_gets": other_player_gets}

        if has_message:
            message_content = response.split("<message>", 1)[1].split("</message>", 1)[0].strip()
            return False, "", False, message_content

        return True, "Unknown error: Invalid response format.", False, None

    def initialize_prompts(self, prompts):
        """
        Initializes and formats prompts from the configuration.

        Args:
            prompts (dict): Dictionary containing various prompts.

        Returns:
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

        Args:
            prompt (str): The prompt with placeholders.
            state (dict): The current state of the game.

        Returns:
            str: The formatted prompt.
        """
        if prompt: 
            if state.get("has_finalized"):
                other_player_finalization = state.get("last_message", "")
            else:
                other_player_finalization = ""
            
            # Get the values for the current player based on their role.
            values = state["role_values"][state["player_to_role"][state["current_player"]]]
            
            # Build a summary from the last round (if available)
            last_round_info = ""
            if state.get("round_number", 0) > 0:
                last_points = state["round_points"][-1] if state.get("round_points") else "N/A"
                last_finalizations = state["round_finalizations"][-1] if state.get("round_finalizations") else "N/A"
                last_round_info = f"Points: {last_points}, Finalizations: {last_finalizations}"

            return prompt.replace("{rounds_per_game}", str(state.get("rounds_per_game", ""))) \
                        .replace("{last_round_info}", last_round_info) \
                        .replace("{current_round}", str(state.get("current_round", ""))) \
                        .replace("{nb_rounds}", str(state["round_number"] + 1)) \
                        .replace("{quantities}", str(state.get("quantities", ""))) \
                        .replace("{values}", str(values)) \
                        .replace("{items}", str(state.get("items", ""))) \
                        .replace("{max_reasoning_chars}", str(self.max_reasoning_chars)) \
                        .replace("{max_messages}", str(self.max_messages)) \
                        .replace("{max_chars_per_message}", str(self.max_chars_per_message)) \
                        .replace("{max_retries}", str(self.max_retries)) \
                        .replace("{other_player_finalization}", str(other_player_finalization))
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

        Returns:
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
            if self.retries > self.max_retries:
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
        self.set_chat_returns_func(self.conversation_history, **self.set_chat_returns_func_args)

        with open(path, "w") as f:
            json.dump(game_stats, f)
            json.dump(self.conversation_history, f)





