from pydantic import BaseModel


class DomContentItem(BaseModel):
	index: int
	text: str


class ProcessedDomContent(BaseModel):
	items: list[DomContentItem]
	selector_map: dict[int, str]

	def dom_items_to_string(self) -> str:
		"""Convert the processed DOM content to HTML."""
		return '\n'.join([f'{item.index}:{item.text}' for item in self.items])
