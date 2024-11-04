from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agent.service import AgentService
from src.agent.views import AgentActionResult, AgentPageState
from src.planning.prompts import PlanningMessagePrompt, PlanningSystemPrompt
from src.planning.views import PlanningAgentAction

load_dotenv()


class PlaningService:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		agent: AgentService | None = None,
		use_vision: bool = False,
	):
		"""
		Planning service.

		Args:
			task (str): Task to be performed.
			llm (AvailableModel): Model to be used.
			browser (BrowserService | None): You can reuse an existing browser service or (automatically) create a new one.
		"""
		self.agent = agent or AgentService()

		self.use_vision = use_vision

		self.llm = llm
		system_prompt = PlanningSystemPrompt(
			task, self._get_action_description()
		).get_system_message()
		first_message = HumanMessage(content=f'Your task is: {task}')

		# self.messages_all: list[BaseMessage] = []
		self.messages: list[BaseMessage] = [system_prompt, first_message]

	async def step(self) -> tuple[PlanningAgentAction, AgentActionResult]:
		state = self.agent.get_current_state(screenshot=self.use_vision)
		action = await self.get_next_action(state)

		if action.ask_human and action.ask_human.question:
			action = await self._take_human_input(action.ask_human.question)

		result = self.agent.act(action)
		input('continue? ...')

		return action, result

	async def _take_human_input(self, question: str) -> PlanningAgentAction:
		human_input = input(f'Human input required: {question}')

		self.messages.append(HumanMessage(content=human_input))
		# chain = (
		# 	ChatPromptTemplate.from_messages(self.messages)
		# 	| self.model
		# 	| PydanticOutputParser(pydantic_object=PlanningAgentAction)
		# )
		structured_llm = self.llm.with_structured_output(PlanningAgentAction)
		action: PlanningAgentAction = await structured_llm.ainvoke(self.messages)  # type: ignore

		self.messages.append(AIMessage(content=action.model_dump_json()))

		return action

	async def get_next_action(self, state: AgentPageState) -> PlanningAgentAction:
		# TODO: include state, actions, etc.

		new_message = PlanningMessagePrompt(state).get_user_message()

		input_messages = self.messages + [new_message]
		if self.use_vision:
			print(f'model input content with image: {new_message.content[0]}')
		else:
			print(f'model input: {new_message}')

		structured_llm = self.llm.with_structured_output(PlanningAgentAction)

		# print(f'state:\n{state}')

		response: PlanningAgentAction = await structured_llm.ainvoke(input_messages)  # type: ignore

		# print('response', response)

		# if store_conversation:
		# 	# save conversation
		# 	save_conversation(input_messages, response.model_dump_json(), store_conversation)

		# Only append the output message
		history_new_message = PlanningMessagePrompt(state).get_message_for_history()
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
		return PlanningAgentAction.description()
