from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agent.prompts import AgentMessagePrompt, AgentSystemPrompt
from src.agent.views import AgentAction
from src.controller.service import ControllerService
from src.controller.views import ControllerActionResult, ControllerPageState
from src.utils import time_execution_async

load_dotenv()


class AgentService:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		controller: ControllerService | None = None,
		use_vision: bool = False,
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

	async def step(self) -> tuple[AgentAction, ControllerActionResult]:
		state = self.controller.get_current_state(screenshot=self.use_vision)
		action = await self.get_next_action(state)

		if action.ask_human and action.ask_human.question:
			action = await self._take_human_input(action.ask_human.question)

		result = self.controller.act(action)

		return action, result

	async def _take_human_input(self, question: str) -> AgentAction:
		human_input = input(f'Human input required: {question}')

		self.messages.append(HumanMessage(content=human_input))

		structured_llm = self.llm.with_structured_output(AgentAction)
		action: AgentAction = await structured_llm.ainvoke(self.messages)  # type: ignore

		self.messages.append(AIMessage(content=action.model_dump_json()))

		return action

	@time_execution_async('get_next_action')
	async def get_next_action(self, state: ControllerPageState) -> AgentAction:
		# TODO: include state, actions, etc.

		new_message = AgentMessagePrompt(state).get_user_message()

		input_messages = self.messages + [new_message]
		# if self.use_vision:
		# 	print(f'model input content with image: {new_message.content[0]}')
		# else:
		# 	print(f'model input: {new_message}')

		structured_llm = self.llm.with_structured_output(AgentAction)

		# print(f'state:\n{state}')

		response: AgentAction = await structured_llm.ainvoke(input_messages)  # type: ignore

		# print('response', response)

		# if store_conversation:
		# 	# save conversation
		# 	save_conversation(input_messages, response.model_dump_json(), store_conversation)

		# Only append the output message
		history_new_message = AgentMessagePrompt(state).get_message_for_history()
		self.messages.append(history_new_message)
		self.messages.append(AIMessage(content=response.model_dump_json()))

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

		return response

	def _get_action_description(self) -> str:
		return AgentAction.description()
