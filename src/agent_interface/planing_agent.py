from typing import Dict, List, Optional
import json
import os
from openai import OpenAI
from dotenv import load_dotenv


class PlaningAgent:
    def __init__(self, task: str, default_actions: str):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.messages = [{"role": "system", "content": self.get_system_prompt(task, default_actions)}]

    def chat(self, task: str) -> Dict:
        # TODO: include state, actions, etc.

        # select next functions to call
        self.messages.append({"role": "user", "content": task})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            response_format={"type": "json_object"}
        )
        self.messages.append(response.choices[0].message)
        # parse the response
        return json.loads(response.choices[0].message.content)

    def get_system_prompt(self, task: str, default_actions: str) -> str:
        # System prompts for the agent
        output_format = """
        {"action": "action_name", "params": {"param_name": "param_value"}}
        """

        AGENT_PROMPT = f"""
        You are a web scraping agent. Your task is to control the browser where, for every step, 
        you get the current state and a list of actions you can take as dictionary. You have to select the next action in json format:
        {output_format}

        Your task is:
        {task}

        Available default actions:
        {default_actions}
        """

        return AGENT_PROMPT
