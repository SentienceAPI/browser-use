from pydantic import BaseModel
import pytest

from src.llm.service import LLM


@pytest.mark.skip(reason='openai costs')
async def test_completion():
	messages = [{'role': 'user', 'content': 'List 5 important events in the XIX century'}]

	class CalendarEvent(BaseModel):
		name: str
		date: str
		participants: list[str]

	class EventsList(BaseModel):
		events: list[CalendarEvent]

	llm = LLM(model='gpt-4o-2024-08-06')

	resp = await llm.create_chat_completion(messages, EventsList)

	print('Received={}'.format(resp))
