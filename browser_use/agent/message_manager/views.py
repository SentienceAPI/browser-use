from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class MessageMetadata(BaseModel):
	"""Metadata for a message including token counts"""

	input_tokens: Optional[int] = None
	output_tokens: Optional[int] = None
	total_tokens: Optional[int] = None


class ManagedMessage(BaseModel):
	"""A message with its metadata"""

	message: BaseMessage
	metadata: MessageMetadata = Field(default_factory=MessageMetadata)


class MessageHistory(BaseModel):
	"""Container for message history with metadata"""

	messages: List[ManagedMessage] = Field(default_factory=list)
	total_tokens: int = 0

	def add_message(self, message: BaseMessage, metadata: Optional[MessageMetadata] = None) -> None:
		"""Add a message with optional metadata"""
		self.messages.append(
			ManagedMessage(message=message, metadata=metadata or MessageMetadata())
		)
		if metadata and metadata.total_tokens:
			self.total_tokens += metadata.total_tokens

	def remove_last_state_message(self) -> None:
		"""Remove last state message from history"""
		if self.messages:
			self.messages.pop()
