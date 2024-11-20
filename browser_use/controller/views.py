from typing import Literal

from pydantic import BaseModel


# Action Input Models
class SearchGoogleAction(BaseModel):
	query: str


class GoToUrlAction(BaseModel):
	url: str


class ClickElementAction(BaseModel):
	index: int
	num_clicks: int = 1


class InputTextAction(BaseModel):
	index: int
	text: str


class DoneAction(BaseModel):
	text: str


class SwitchTabAction(BaseModel):
	page_id: int


class OpenTabAction(BaseModel):
	url: str


class ExtractPageContentAction(BaseModel):
	value: Literal['text', 'markdown', 'html'] = 'text'
