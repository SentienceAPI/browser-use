import logging
from typing import List, Literal, TypeVar

from litellm import LiteLLM, acompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


AvailableModel = Literal[
	'gpt-4o',
	'gpt-4o-mini',
	'gpt-4o-2024-08-06',
	'claude-3-5-sonnet-latest',
	'claude-3-5-sonnet-20241022',
	'claude-3-5-sonnet-20240620',
]


class ChatCompletionResponseMessage(BaseModel):
	role: Literal['user', 'assistant', 'system']
	content: str


ResponseModel = TypeVar('ResponseModel', bound=BaseModel)


class LLM:
	def __init__(self, model: AvailableModel):
		self.logger = logger
		self.model = model
		self.llm = LiteLLM()

	async def create_chat_completion(
		self,
		messages: list,
		response_model: type[ResponseModel],
	) -> ResponseModel:
		response = await acompletion(
			self.model, messages, response_format=response_model, stream=False
		)
		content: str = response.choices[0].message.content  # type: ignore
		# Access the completion content directly from the response

		return response_model.model_validate_json(content)
