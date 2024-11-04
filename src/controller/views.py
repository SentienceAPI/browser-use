from typing import Literal, Optional

from pydantic import BaseModel

from src.browser.views import BrowserState


class SearchGoogleControllerAction(BaseModel):
	query: str


class GoToUrlControllerAction(BaseModel):
	url: str


class ClickElementControllerAction(BaseModel):
	id: int


class InputTextControllerAction(BaseModel):
	id: int
	text: str


class ControllerActions(BaseModel):
	"""
	Controller actions you can use to interact.
	"""

	search_google: Optional[SearchGoogleControllerAction] = None
	go_to_url: Optional[GoToUrlControllerAction] = None
	nothing: Optional[Literal[True]] = None
	go_back: Optional[Literal[True]] = None
	done: Optional[Literal[True]] = None
	click_element: Optional[ClickElementControllerAction] = None
	input_text: Optional[InputTextControllerAction] = None
	extract_page_content: Optional[Literal[True]] = None

	@staticmethod
	def description() -> str:
		"""
		Returns a human-readable description of available actions.
		"""
		return """
- Search Google with a query
  Example: {"search_google": {"query": "weather today"}}

- Navigate directly to a URL
  Example: {"go_to_url": {"url": "https://abc.com"}}

- Do nothing/wait
  Example: {"nothing": true}

- Go back to previous page
  Example: {"go_back": true}

- Mark task as complete
  Example: {"done": true}

- Click an element by its ID
  Example: {"click_element": {"id": 1}}

- Input text into an element by its ID
  Example: {"input_text": {"id": 1, "text": "Hello world"}}

- Extract and return the page content in markdown
  Example: {"extract_page_content": true}
"""


class ControllerActionResult(BaseModel):
	done: bool
	extracted_content: Optional[str] = None
	error: Optional[str] = None


class ControllerPageState(BrowserState):
	screenshot: Optional[str] = None
