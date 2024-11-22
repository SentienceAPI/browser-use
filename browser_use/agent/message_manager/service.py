from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Type

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
	AIMessage,
	BaseMessage,
	HumanMessage,
	SystemMessage,
	get_buffer_string,
)
from langchain_openai import ChatOpenAI

from browser_use.agent.message_manager.views import MessageHistory, MessageMetadata
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import ActionResult
from browser_use.browser.views import BrowserState

logger = logging.getLogger(__name__)


class MessageManager:
	def __init__(
		self,
		llm: BaseChatModel,
		task: str,
		action_descriptions: str,
		system_prompt_class: Type[SystemPrompt],
		max_input_tokens: int = 128000,
	):
		self.llm = llm
		self.system_prompt_class = system_prompt_class
		self.max_input_tokens = max_input_tokens
		self.history = MessageHistory()
		self.task = task
		self.action_descriptions = action_descriptions

		system_message = self.system_prompt_class(
			self.action_descriptions, current_date=datetime.now()
		).get_system_message()
		self._add_message_with_tokens(system_message)

		# Add task message
		task_message = HumanMessage(content=f'Your task is: {task}')
		self._add_message_with_tokens(task_message)

	def add_state_message(self, state: BrowserState) -> None:
		"""Add browser state as human message"""
		from browser_use.agent.prompts import AgentMessagePrompt

		state_message = AgentMessagePrompt(state).get_user_message()
		self._add_message_with_tokens(state_message)

	def add_model_output(self, model_output: AIMessage) -> None:
		"""Add model output as AI message"""
		self._add_message_with_tokens(model_output)

	def update_with_action_result(
		self, result: ActionResult, include_in_state: bool = True
	) -> None:
		"""Update history with action result"""
		if not result.extracted_content and not result.error:
			return

		content = result.extracted_content or result.error
		if content is None:
			return

		if result.include_in_memory:
			message = HumanMessage(content=str(content))
			self._add_message_with_tokens(message)
		elif include_in_state:
			# Update the content of the last state message
			if self.history.messages:
				last_message = self.history.messages[-1].message
				if isinstance(last_message, HumanMessage):
					last_message.content = str(last_message.content) + f'\n{content}'

	def get_messages(self) -> List[BaseMessage]:
		"""Get current message list, potentially trimmed to max tokens"""
		return [m.message for m in self.history.messages]

	def _add_message_with_tokens(self, message: BaseMessage) -> None:
		"""Add message with token count metadata"""
		token_count = self._count_tokens(message)
		metadata = MessageMetadata(
			input_tokens=token_count, output_tokens=0, total_tokens=token_count
		)
		self.history.add_message(message, metadata)

	def _count_tokens(self, message: BaseMessage) -> int:
		"""Count tokens in a message using the model's tokenizer"""
		if isinstance(self.llm, (ChatOpenAI, ChatAnthropic)):
			return self.llm.get_num_tokens(get_buffer_string([message]))
		return len(str(message.content)) // 4  # Rough estimate if no tokenizer available
