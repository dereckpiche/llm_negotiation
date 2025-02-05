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
        new_round_prompt,
        initial_move_prompt,
        ongoing_moves_prompt,
        proposal_prompt
    ):
        """
        Initializes the DondPlayerHandler.

        Args:
            player_name (str): The name of the player.
            prompts (dict): Dictionary containing various prompts.
            allow_reasoning (bool): Whether reasoning is allowed.
            max_retries (int): Maximum number of retries allowed.
            mod_adpt_id (str): The name of the model used.
            max_reasoning_chars (int): Max number of reasoning characters.
            max_messages (int): Max number of messages.
            max_chars_per_message (int): Max number of characters per message.
            player_intro_prompt_appending (str): Additional intro prompt for the player.
            intro_prompt (list): List of strings for the game introduction prompt.
        """
        self.allow_reasoning = allow_reasoning
        self.player_name = player_name
        self.max_retries = max_retries
        self.mod_adpt_id = mod_adpt_id
        self.max_reasoning_chars = max_reasoning_chars
        self.max_messages = max_messages
        self.max_chars_per_message = max_chars_per_message

        self.intro_prompt = intro_prompt
        self.goal_prompt = goal_prompt
        self.new_round_prompt = new_round_prompt
        self.initial_move_prompt = initial_move_prompt
        self.ongoing_moves_prompt = ongoing_moves_prompt
        self.proposal_prompt = proposal_prompt
        self.game_id = None  # ID of player in game
        self.reset()

    def set_usr_message(self, state):
        """
        Constructs a user message based on the current game state.

        Args:
            state (dict): The current state of the game.

        Returns:
            str: The constructed user message.
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
            if not (state["round_number"] == 0 and len(self.chat_history) == 2):
                self.new_round()

                user_message += self.format_prompt(self.new_round_prompt, state)

        if state["has_proposed"]:
            user_message += self.format_prompt(self.proposal_prompt, state)

        elif state["last_message"] is None:
            user_message += self.initial_move_prompt
        else:
            user_message += self.format_prompt(self.ongoing_moves_prompt, state)

        user_message = {
            "role": "user",
            "content": user_message,
            "is_error": False,
            "round_nb": state["round_number"],
        }
        self.add_to_chat_history(user_message)

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
                - is_proposal (bool): Indicates if the response is a proposal.
                - processed_response (str or dict): The extracted message or proposal details.
        """
        errors = []

        # 1) Validate presence of <think> tags (if reasoning is allowed)
        if self.allow_reasoning is not False:
            if "<think>" not in response or "</think>" not in response:
                errors.append("Missing <think>...</think> tag.")

        # 2) Check if exactly one of <message>...</message> or <propose>...</propose> is present,
        #    and ensure proposal rules are followed
        has_message = "<message>" in response and "</message>" in response
        has_proposal = "<propose>" in response and "</propose>" in response

        if (state["turn"] > state["max_turns"] - 2) and not has_proposal:
            errors.append("You must propose before the turn limit!")
        if has_message and has_proposal:
            errors.append("Response contains both <message> and <propose>; only one is allowed.")
        elif not has_message and not has_proposal:
            errors.append("Response must contain either <message> or <propose>.")
        if state["has_proposed"] and not has_proposal:
            errors.append("The other player has proposed; you must propose as well.")

        # 3) Check for excessive content outside valid tags (<think>, <message>, <propose>)
        total_length = len(response)
        valid_tags = ["think", "message", "propose"]
        total_tag_content_length = 0
        for tag in valid_tags:
            pattern = rf"<{tag}>.*?</{tag}>"
            matches = re.findall(pattern, response, flags=re.S)
            for match in matches:
                total_tag_content_length += len(match)
        outside_length = total_length - total_tag_content_length
        if outside_length > 5:
            errors.append("Excessive content outside of valid tags.")

        # 4) Process proposal if present
        if has_proposal:
            proposal_content = response.split("<propose>", 1)[1].split("</propose>", 1)[0].strip()
            try:
                proposal_json = json.loads(proposal_content)
                if not isinstance(proposal_json, dict):
                    errors.append("The content within <propose> is not a valid dictionary.")
                    i_take = None
                    other_player_gets = None
                else:
                    i_take = proposal_json.get("i_take", {})
                    other_player_gets = proposal_json.get("other_player_gets", {})
                if not isinstance(i_take, dict) or not isinstance(other_player_gets, dict):
                    errors.append('"i_take" and "other_player_gets" must be dictionaries.')
                else:
                    for item in state["items"]:
                        if not isinstance(i_take.get(item), int):
                            errors.append(f'Value of "{item}" in "i_take" must be an integer.')
                        if not isinstance(other_player_gets.get(item), int):
                            errors.append(f'Value of "{item}" in "other_player_gets" must be an integer.')
            except json.JSONDecodeError:
                errors.append("The content within <propose> is not valid JSON.")

        # 5) Return results: check for errors, otherwise return parsed content
        if errors:
            return True, "Errors: " + "; ".join(errors), False, None

        if has_proposal:
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
        return prompt.replace("{rounds_per_game}", str(state.get("rounds_per_game", ""))) \
                     .replace("{last_round_info}", state.get("last_round_info", "")) \
                     .replace("{current_round}", str(state.get("current_round", ""))) \
                     .replace("{nb_rounds}", str(state.get("nb_rounds", ""))) \
                     .replace("{quantities}", str(state.get("quantities", ""))) \
                     .replace("{values}", str(state.get("values", ""))) \
                     .replace("{other_player_values}", state.get("other_player_values", "")) \
                     .replace("{max_reasoning_chars}", str(self.max_reasoning_chars)) \
                     .replace("{max_messages}", str(self.max_messages)) \
                     .replace("{max_chars_per_message}", str(self.max_chars_per_message)) \
                     .replace("{max_retries}", str(self.max_retries))

    def format_prompt_list(self, prompt_list, state):
        """
        Formats a list of prompts into a single string using the game state.

        Args:
            prompt_list (list): List of prompt strings.
            state (dict): The current state of the game.

        Returns:
            str: Concatenated and formatted prompt.
        """
        return "\n".join(self.format_prompt(prompt, state) for prompt in prompt_list)

    def format_intro_prompt_appending(self, appending):
        """
        Replaces placeholders in the player intro prompt appending with actual values.

        Args:
            appending (str): The player intro prompt appending with placeholders.

        Returns:
            str: The formatted player intro prompt appending.
        """
        return appending.replace("${max_reasoning_chars}", str(self.max_reasoning_chars)) \
                        .replace("${max_messages}", str(self.max_messages)) \
                        .replace("${max_chars_per_message}", str(self.max_chars_per_message))

    def format_intro_prompt(self, intro_prompt_list):
        """
        Concatenates and formats the intro prompt list into a single string.

        Args:
            intro_prompt_list (list): List of strings for the intro prompt.

        Returns:
            str: The concatenated and formatted intro prompt.
        """
        formatted_intro = ""
        for prompt_key in intro_prompt_list:
            prompt_value = self.prompts.get(prompt_key, "")
            formatted_intro += prompt_value + "\n"
        return formatted_intro

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
        is_proposal = False

        # Process response. Check for errors.
        is_error, error_message, is_proposal, processed_response = self.process_response(
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
            "is_proposal": is_proposal,
            "round_nb": state["round_number"],
        }

        self.add_to_chat_history(model_response)

        action = (is_proposal, processed_response)
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





