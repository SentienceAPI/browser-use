import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tokencost import calculate_prompt_cost, count_string_tokens
from src.actions.browser_actions import Action

from src.llm.service import LLM


class PlaningAgent:
	def __init__(self, task: str, default_actions: str):
		load_dotenv()
		# self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
		self.model = 'gpt-4o'
		self.llm = LLM(model=self.model)
		self.messages = [
			{'role': 'system', 'content': self.get_system_prompt(task, default_actions)}
		]

	async def chat(self, task: str, skip_call: bool = False) -> Dict:
		# TODO: include state, actions, etc.

		# select next functions to call
		messages = self.messages + [{'role': 'user', 'content': task}]

		# Calculate total cost for all messages
		total_cost = calculate_prompt_cost(messages, self.model)
		total_tokens = count_string_tokens(' '.join([m['content'] for m in messages]), self.model)
		print(
			'Total prompt cost: ',
			f'${total_cost:,.2f}',
			'Total tokens: ',
			f'{total_tokens:,}',
		)

		if skip_call:
			return {'action': 'nothing'}

		response = await self.llm.create_chat_completion(messages, Action)

		# response = self.client.chat.completions.create(
		# 	model=self.model, messages=messages, response_format={'type': 'json_object'}
		# )

		# Only append the output message
		self.messages.append({'role': 'assistant', 'content': response.choices[0].message.content})

		return json.loads(response.choices[0].message.content)

	def get_system_prompt(self, task: str, default_actions: str) -> str:
		"""
		Get the system prompt for the agent.

		Args:
		    task: The task description
		    default_actions: Available default actions

		Returns:
		    str: Formatted system prompt
		"""
		# System prompts for the agent
		output_format = """
        {"action": "action_name", "params": {"param_name": "param_value"}}
        """

		AGENT_PROMPT = f"""
        You are an AI agent that helps users navigate websites and perform actions. Your task is: {task}

        Available actions:
        {default_actions}

        The page content will be provided as numbered elements like this:
        0:<button>Click me</button>
        1:<a href="/test">Link text</a>
        2:Some visible text content

        To interact with elements, use their index number in the click() or input() actions.
        Each element has a unique index that can be used to interact with it.

        Provide your next action based on the available actions and visible elements. 
        Respond with a valid JSON object containing the action and any required parameters.

        Response format:
        {output_format}

        """

		return AGENT_PROMPT

		# Remember:
		# 1. Always check if elements exist before trying to interact with them
		# 2. Use the exact index numbers shown in the elements list
		# 3. If you need to accept cookies, use the accept_cookies action first
		# 4. If you can't find what you're looking for, consider scrolling or navigating to a different page
