from typing import Dict, List, Optional
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tokencost import calculate_prompt_cost, count_string_tokens


class PlaningAgent:
    def __init__(self, task: str, default_actions: str):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.messages = [{"role": "system", "content": self.get_system_prompt(task, default_actions)}]

    def chat(self, task: str) -> Dict:
        # TODO: include state, actions, etc.

        # select next functions to call
        messages = self.messages + [{"role": "user", "content": task}]

        # Calculate total cost for all messages
        total_cost = calculate_prompt_cost(messages, self.model)
        total_tokens = count_string_tokens(" ".join([m["content"] for m in messages]), self.model)
        print(
            "Total prompt cost: ", f"${total_cost:,.2f}",
            "Total tokens: ", f"{total_tokens:,}",
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        # Only append the output message
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        # parse the response
        return json.loads(response.choices[0].message.content)

    def get_system_prompt(self, task: str, default_actions: str) -> str:
        # System prompts for the agent
        output_format = """
        {"action": "action_name", "params": {"param_name": "param_value"}}
        """

        AGENT_PROMPT = f"""
        You are a web scraping agent. Your task is to control the browser where, for every step, 
        you get the current state and a list of actions you can take as dictionary. 
        If you want to click on an element or input text, you need to specify the element id (c="") from the cleaned HTML.
        You have to select the next action in json format:
        {output_format}

        Your task is:
        {task}

        Available actions:
        {default_actions}
        """

        return AGENT_PROMPT
