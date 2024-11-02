from dotenv import load_dotenv
from tokencost import calculate_prompt_cost, count_string_tokens

from src.actions.browser_actions import Action
from src.llm.service import LLM, AvailableModel
from src.state_manager.utils import save_conversation


class PlaningAgent:
	def __init__(self, task: str, default_actions: str, model: AvailableModel):
		load_dotenv()
		self.model = model
		self.llm = LLM(model=self.model)
		self.system_prompt = [
			{'role': 'system', 'content': self.get_system_prompt(task, default_actions)}
		]
		self.messages_all = []
		self.messages = []

	async def chat(
		self, task: str, skip_call: bool = False, store_conversation: str = ''
	) -> Action:
		# TODO: include state, actions, etc.

		# select next functions to call
		input_messages = self.system_prompt + self.messages + [{'role': 'user', 'content': task}]

		try:
			# Calculate total cost for all messages
			total_cost = calculate_prompt_cost(input_messages, self.model)
			total_tokens = count_string_tokens(
				' '.join([m['content'] for m in input_messages]), self.model
			)
			print(
				'Total prompt cost: ',
				f'${total_cost:,.2f}',
				'Total tokens: ',
				f'{total_tokens:,}',
			)
		except Exception as e:
			print(f'Error calculating prompt cost: {e}')

		if skip_call:
			return Action(action='nothing')

		response = await self.llm.create_chat_completion(input_messages, Action)

		# self.messages.append({'role': 'user', 'content': '... execute action ...'})
		if store_conversation:
			# save conversation
			save_conversation(input_messages, response.model_dump_json(), store_conversation)

		# Only append the output message
		self.messages.append({'role': 'assistant', 'content': response.model_dump_json()})

		# keep newest 20 messages
		if len(self.messages) > 20:
			self.messages = self.messages[-20:]

		return response

	def update_system_prompt(self, user_input: str):
		self.system_prompt.append({'role': 'user', 'content': user_input})

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
        {"action": "action_name", "params": {"param_name": "param_value"}, "goal": "short description what you want to achieve", "valuation_previous_goal": "Success if completed, else short sentence of why not successful."}
        """

		AGENT_PROMPT = f"""
        You are an AI agent that helps users interact with websites. 

		Your input are all the interactive elements of the current page from which you can choose which to click or input. They will be provided like this:
		0:Chat-Fenster
		1:Skip to Main Content.
		2:<a href="https://www.example.de/"></a>
		3:<button>Book now</button>
		4:<div>Book now</div>
		26:<span>Other interactive elements</span>

		Additional you get a list of previous actions and their results.

		Available actions:
        {default_actions}

        To interact with elements, use their index number in the click() or text_input() actions. 

		If you need more than the interactive elements from the page you can use the extract_content action.

		Validate if the previous goal is achieved, if not, try to achieve it with the next action.
		If you get stuck, try to find a new element that can help you achieve your goal or if persistent, go back or reload the page.
        Respond with a valid JSON object containing the action, any required parameters and your current goal of this action.
		You can send_user_text or ask_user for clarification if you are completely stuck. 

        Response format:
        {output_format}

        """
		return AGENT_PROMPT

		# Remember:
		# 1. Always check if elements exist before trying to interact with them
		# 2. Use the exact index numbers shown in the elements list
		# 3. If you need to accept cookies, use the accept_cookies action first
		# 4. If you can't find what you're looking for, consider scrolling or navigating to a different page
