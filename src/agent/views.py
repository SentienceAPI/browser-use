from typing import Literal, Optional

from pydantic import BaseModel

from src.browser.views import BrowserState


class SearchGoogleAgentAction(BaseModel):
	query: str


class GoToUrlAgentAction(BaseModel):
	url: str


class ClickElementAgentAction(BaseModel):
	id: int


class InputTextAgentAction(BaseModel):
	id: int
	text: str


class AgentActions(BaseModel):
	"""
	Agent actions you can use to interact with the agent.
	"""

	search_google: Optional[SearchGoogleAgentAction] = None
	go_to_url: Optional[GoToUrlAgentAction] = None
	nothing: Optional[Literal[True]] = None
	go_back: Optional[Literal[True]] = None
	done: Optional[Literal[True]] = None
	click_element: Optional[ClickElementAgentAction] = None
	input_text: Optional[InputTextAgentAction] = None
	extract_page_content: Optional[Literal[True]] = None


class AgentActionResult(BaseModel):
	done: bool
	extracted_content: Optional[str] = None
	error: Optional[str] = None


class AgentPageState(BrowserState):
	screenshot: Optional[str] = None
