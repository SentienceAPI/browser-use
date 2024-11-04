import json

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agent.prompts import AgentMessagePrompt, AgentSystemPrompt
from src.agent.views import AgentOutput, Output
from src.controller.service import ControllerService
from src.controller.views import ControllerActionResult, ControllerPageState
from src.utils import time_execution_async, time_execution_sync

load_dotenv()


class AgentService:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		controller: ControllerService | None = None,
		use_vision: bool = False,
		save_file: str | None = None,
	):
		"""
		Agent service.

		Args:
			task (str): Task to be performed.
			llm (AvailableModel): Model to be used.
			controller (ControllerService | None): You can reuse an existing or (automatically) create a new one.
		"""
		self.controller = controller or ControllerService()

		self.use_vision = use_vision

		self.llm = llm
		system_prompt = AgentSystemPrompt(
			task, default_action_description=self._get_action_description()
		).get_system_message()

		print(system_prompt)
		first_message = HumanMessage(content=f'Your main task is: {task}')

		# self.messages_all: list[BaseMessage] = []
		self.messages: list[BaseMessage] = [system_prompt, first_message]
		self.save_file = save_file
		if save_file is not None:
			print(f'Saving conversation to {save_file}')
		self.n = 0

	async def step(self) -> tuple[AgentOutput, ControllerActionResult]:
		state = self.controller.get_current_state(screenshot=self.use_vision)
		action = await self.get_next_action(state)

		if action.ask_human and action.ask_human.question:
			action = await self._take_human_input(action.ask_human.question)

		result = self.controller.act(action)
		self.n += 1

		return action, result

	async def _take_human_input(self, question: str) -> AgentOutput:
		human_input = input(f'Human input required: {question}')

		self.messages.append(HumanMessage(content=human_input))

		structured_llm = self.llm.with_structured_output(AgentOutput)
		action: AgentOutput = await structured_llm.ainvoke(self.messages)  # type: ignore

		self.messages.append(AIMessage(content=action.model_dump_json()))

		return action

	@time_execution_async('get_next_action')
	async def get_next_action(self, state: ControllerPageState) -> AgentOutput:
		# TODO: include state, actions, etc.

		new_message = AgentMessagePrompt(state).get_user_message()

		input_messages = self.messages + [new_message]
		# if self.use_vision:
		# 	print(f'model input content with image: {new_message.content[0]}')
		# else:
		# 	print(f'model input: {new_message}')

		structured_llm = self.llm.with_structured_output(Output, include_raw=False)

		# print(f'state:\n{state}')
		#
		response: Output = await structured_llm.ainvoke(input_messages)  # type: ignore
		# raw_response, response = invoke_response
		# if store_conversation:
		# 	# save conversation
		# 	save_conversation(input_messages, response.model_dump_json(), store_conversation)

		# Only append the output message
		history_new_message = AgentMessagePrompt(state).get_message_for_history()
		self.messages.append(history_new_message)
		self.messages.append(AIMessage(content=response.model_dump_json()))
		print(f'current state\n: {response.current_state.model_dump_json(indent=4)}')
		print(f'action\n: {response.action.model_dump_json(indent=4)}')
		self._save_conversation(input_messages, response)

		# try:
		# 	# Calculate total cost for all messages

		# 	# check if multiple input

		# 	# can not handly list of content
		# 	output = calculate_all_costs_and_tokens(
		# 		[system_prompt] + self.messages + [new_message],
		# 		response.model_dump_json(),
		# 		self.model,
		# 	)
		# 	if images:
		# 		# resolution 1512 x 767
		# 		image_cost = 0.000213
		# 		total_cost = (
		# 			output['prompt_cost']
		# 			+ output['completion_cost']
		# 			+ decimal.Decimal(str(image_cost))
		# 		)
		# 		print(
		# 			f'Text ${output["prompt_cost"]:,.4f} + Image ${image_cost:,.4f} = ${total_cost:,.4f} for {output["prompt_tokens"] + output["completion_tokens"]}  tokens'
		# 		)
		# 	else:
		# 		total_cost = output['prompt_cost'] + output['completion_cost']
		# 		print(
		# 			f'Total cost: ${total_cost:,.4f} for {output["prompt_tokens"] + output["completion_tokens"]} tokens'
		# 		)

		# except Exception as e:
		# 	print(f'Error calculating prompt cost: {e}')

		# # keep newest 20 messages
		# if len(self.messages) > 20:
		# 	self.messages = self.messages[-20:]

		return response.action

	def _get_action_description(self) -> str:
		return AgentOutput.description()

	@time_execution_sync('_save_conversation')
	def _save_conversation(self, input_messages: list[BaseMessage], response: Output):
		if self.save_file is not None:
			with open(self.save_file + f'_{self.n}.txt', 'w') as f:
				# Write messages with proper formatting
				for message in input_messages:
					f.write('=' * 33 + f' {message.__class__.__name__} ' + '=' * 33 + '\n\n')

					# Handle different content types
					if isinstance(message.content, list):
						# Handle vision model messages
						for item in message.content:
							if item['type'] == 'text':
								f.write(item['text'].strip() + '\n')
					elif isinstance(message.content, str):
						try:
							# Try to parse and format JSON content
							content = json.loads(message.content)
							f.write(json.dumps(content, indent=2) + '\n')
						except json.JSONDecodeError:
							# If not JSON, write as regular text
							f.write(message.content.strip() + '\n')

					f.write('\n')

				# Write final response as formatted JSON
				f.write('=' * 33 + ' Response ' + '=' * 33 + '\n\n')
				f.write(json.dumps(json.loads(response.model_dump_json()), indent=2))
