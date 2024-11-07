import json
import logging

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from browser_use.agent.prompts import AgentMessagePrompt, AgentSystemPrompt
from browser_use.agent.views import (
	AgentHistory,
	AgentOutput,
	ClickElementControllerHistoryItem,
	InputTextControllerHistoryItem,
	Output,
)
from browser_use.controller.service import ControllerService
from browser_use.controller.views import (
	ControllerActionResult,
	ControllerPageState,
)
from browser_use.utils import time_execution_async

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
	level=logging.INFO,
	force=True,  # Prevent changing root logger config
)


class AgentService:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		controller: ControllerService | None = None,
		use_vision: bool = True,
		save_conversation_path: str | None = None,
	):
		"""
		Agent service.

		Args:
			task (str): Task to be performed.
			llm (AvailableModel): Model to be used.
			controller (ControllerService | None): You can reuse an existing or (automatically) create a new one.
		"""
		self.task = task
		self.use_vision = use_vision

		self.controller_injected = controller is not None
		self.controller = controller or ControllerService()

		self.llm = llm
		system_prompt = AgentSystemPrompt(
			task, default_action_description=self._get_action_description()
		).get_system_message()

		# Init messages
		first_message = HumanMessage(content=f'Your main task is: {task}')
		self.messages: list[BaseMessage] = [system_prompt, first_message]
		self.n = 0

		self.save_conversation_path = save_conversation_path
		if save_conversation_path is not None:
			logger.info(f'Saving conversation to {save_conversation_path}')

		self.action_history: list[AgentHistory] = []

	async def run(self, max_steps: int = 100):
		"""
		Execute the task.

		@dev ctrl+c to interrupt
		"""

		try:
			logger.info('\n' + '=' * 50)
			logger.info(f'🚀 Starting task: {self.task}')
			logger.info('=' * 50)

			for i in range(max_steps):
				logger.info(f'\n📍 Step {i+1}')

				action, result = await self.step()

				if result.done:
					logger.info('\n✅ Task completed successfully')
					logger.info(f'Extracted content: \n{result.extracted_content}')
					return action.done, self.action_history

			logger.info('\n' + '=' * 50)
			logger.info('❌ Failed to complete task in maximum steps')
			logger.info('=' * 50)
			return None, self.action_history
		finally:
			if not self.controller_injected:
				self.controller.browser.close()

	@time_execution_async('--step')
	async def step(self) -> tuple[AgentHistory, ControllerActionResult]:
		state = self.controller.get_current_state(screenshot=self.use_vision)
		action = await self.get_next_action(state)

		if action.ask_human and action.ask_human.question:
			action = await self._take_human_input(action.ask_human.question)

		result = self.controller.act(action)
		self.n += 1

		if result.error:
			self.messages.append(HumanMessage(content=f'Error: {result.error}'))

		if result.extracted_content:
			self.messages.append(
				HumanMessage(content=f'Extracted content:\n {result.extracted_content}')
			)

		# Convert action to history and update click/input fields if present
		history_item = self._make_history_item(action, state)
		self.action_history.append(history_item)

		return history_item, result

	def _make_history_item(self, action: AgentOutput, state: ControllerPageState) -> AgentHistory:
		return AgentHistory(
			search_google=action.search_google,
			go_to_url=action.go_to_url,
			nothing=action.nothing,
			go_back=action.go_back,
			done=action.done,
			click_element=ClickElementControllerHistoryItem(
				id=action.click_element.id, xpath=state.selector_map.get(action.click_element.id)
			)
			if action.click_element and state.selector_map.get(action.click_element.id)
			else None,
			input_text=InputTextControllerHistoryItem(
				id=action.input_text.id,
				xpath=state.selector_map.get(action.input_text.id),
				text=action.input_text.text,
			)
			if action.input_text and state.selector_map.get(action.input_text.id)
			else None,
			extract_page_content=action.extract_page_content,
			switch_tab=action.switch_tab,
			open_tab=action.open_tab,
			ask_human=action.ask_human,
			url=state.url,
		)

	async def _take_human_input(self, question: str) -> AgentOutput:
		human_input = input(f'\nHi, your input is required: {question}\n\n')
		logger.info('-' * 50)
		self.messages.append(HumanMessage(content=human_input))

		structured_llm = self.llm.with_structured_output(AgentOutput)

		action: AgentOutput = await structured_llm.ainvoke(self.messages)  # type: ignore

		self.messages.append(AIMessage(content=action.model_dump_json()))

		return action

	@time_execution_async('--get_next_action')
	async def get_next_action(self, state: ControllerPageState) -> AgentOutput:
		# TODO: include state, actions, etc.

		new_message = AgentMessagePrompt(state).get_user_message()
		logger.info(f'current tabs: {state.tabs}')
		input_messages = self.messages + [new_message]

		structured_llm = self.llm.with_structured_output(Output, include_raw=False)

		#
		response: Output = await structured_llm.ainvoke(input_messages)  # type: ignore

		# Only append the output message
		history_new_message = AgentMessagePrompt(state).get_message_for_history()
		self.messages.append(history_new_message)
		self.messages.append(AIMessage(content=response.model_dump_json()))
		logger.info(f'current state\n: {response.current_state.model_dump_json(indent=4)}')
		logger.info(f'action\n: {response.action.model_dump_json(indent=4)}')
		self._save_conversation(input_messages, response)

		return response.action

	def _get_action_description(self) -> str:
		return AgentOutput.description()

	def _save_conversation(self, input_messages: list[BaseMessage], response: Output):
		if self.save_conversation_path is not None:
			with open(self.save_conversation_path + f'_{self.n}.txt', 'w') as f:
				# Write messages with proper formatting
				for message in input_messages:
					f.write('=' * 33 + f' {message.__class__.__name__} ' + '=' * 33 + '\n\n')

					# Handle different content types
					if isinstance(message.content, list):
						# Handle vision model messages
						for item in message.content:
							if isinstance(item, dict) and item.get('type') == 'text':
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
