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
from browser_use.agent.prompts import AgentMessagePrompt, SystemPrompt
from browser_use.agent.views import ActionResult, AgentOutput
from browser_use.browser.views import BrowserState

logger = logging.getLogger(__name__)


class MessageManager:
	# system prompt
	# task message
	# state message
	# model output
	# state message
	# model output
	# state message
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

	def add_state_message(self, state: BrowserState, result: Optional[ActionResult] = None) -> None:
		"""Add browser state as human message"""

		# if keep in memory, add to directly to history and add state without result
		if result and result.include_in_memory:
			if result.extracted_content:
				msg = HumanMessage(content=str(result.extracted_content))
				self._add_message_with_tokens(msg)
			if result.error:
				msg = HumanMessage(content=str(result.error))
				self._add_message_with_tokens(msg)
			result = None
		# otherwise add state message and result to next message (which will not stay in memory)
		else:
			state_message = AgentMessagePrompt(state, result).get_user_message()
			self._add_message_with_tokens(state_message)

	def _remove_last_state_message(self) -> None:
		"""Remove last state message from history"""
		if self.history.messages:
			self.history.messages.pop()

	def add_model_output(self, model_output: AgentOutput) -> None:
		"""Add model output as AI message"""
		self._remove_last_state_message()
		content = model_output.model_dump_json(exclude_unset=True)
		msg = AIMessage(content=content)
		self._add_message_with_tokens(msg)

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
		tokens = 0
		if isinstance(self.llm, (ChatOpenAI, ChatAnthropic)):
			try:
				tokens = self.llm.get_num_tokens(get_buffer_string([message]))
			except Exception as e:
				tokens = len(str(message.content)) // 4  # Rough estimate if no tokenizer available
				logger.warning(
					f'Error counting tokens: {e} - using estimate of characters/4 tokens'
				)
		else:
			tokens = len(str(message.content)) // 4  # Rough estimate if no tokenizer available
		return tokens
