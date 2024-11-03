import decimal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from tokencost import calculate_all_costs_and_tokens

from src.browser.service import BrowserService

# from src.llm.service import LLM, AvailableModel
from src.planning.prompts import PlanningSystemPrompt
from src.planning.views import PlanningAgentAction
from src.state_manager.utils import encode_image, save_conversation

load_dotenv()


class PlaningService:
	def __init__(self, task: str, model: BaseChatModel, browser: BrowserService | None = None):
		"""
		Planning service.

		Args:
			task (str): Task to be performed.
			model (AvailableModel): Model to be used.
			browser (BrowserService | None): You can reuse an existing browser service or (automatically) create a new one.
		"""
		self.browser = browser or BrowserService()

		self.model = model
		# self.system_prompt = [
		# 	{'role': 'system', 'content': PlanningSystemPrompt(task, default_actions).get_prompt()}
		# ]
		self.messages_all: list[BaseMessage] = []
		self.messages: list[BaseMessage] = []

	async def chat(
		self, task: str, store_conversation: str = '', images: str = ''
	) -> PlanningAgentAction:
		# TODO: include state, actions, etc.

		# select next functions to call
		if images:
			# Format message for vision model
			new_message = HumanMessage(content=[task])
			# combined_input = [
			# 	{'type': 'text', 'text': task},
			# 	{
			# 		'type': 'image_url',
			# 		'image_url': {
			# 			'url': f'data:image/png;base64,{encode_image(images)}',
			# 		},
			# 	},
			# ]
		else:
			new_message = HumanMessage(content=task)
			combined_input = [new_message]
		input_messages = (
			self.system_prompt + self.messages + [{'role': 'user', 'content': combined_input}]
		)

		response = await self.llm.create_chat_completion(input_messages, Action)

		if store_conversation:
			# save conversation
			save_conversation(input_messages, response.model_dump_json(), store_conversation)

		# Only append the output message
		self.messages.append({'role': 'assistant', 'content': response.model_dump_json()})

		try:
			# Calculate total cost for all messages

			# check if multiple input

			# can not handly list of content
			output = calculate_all_costs_and_tokens(
				self.system_prompt + self.messages + [{'role': 'user', 'content': task}],
				response.model_dump_json(),
				self.model,
			)
			if images:
				# resolution 1512 x 767
				image_cost = 0.000213
				total_cost = (
					output['prompt_cost']
					+ output['completion_cost']
					+ decimal.Decimal(str(image_cost))
				)
				print(
					f'Text ${output["prompt_cost"]:,.4f} + Image ${image_cost:,.4f} = ${total_cost:,.4f} for {output["prompt_tokens"] + output["completion_tokens"]}  tokens'
				)
			else:
				total_cost = output['prompt_cost'] + output['completion_cost']
				print(
					f'Total cost: ${total_cost:,.4f} for {output["prompt_tokens"] + output["completion_tokens"]} tokens'
				)

		except Exception as e:
			print(f'Error calculating prompt cost: {e}')

		# keep newest 20 messages
		if len(self.messages) > 20:
			self.messages = self.messages[-20:]

		return response
