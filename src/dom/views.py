from pydantic import BaseModel


class DomContentItem(BaseModel):
	index: int
	text: str
	clickable: bool


class ProcessedDomContent(BaseModel):
	items: list[DomContentItem]
	selector_map: dict[int, str]

	def dom_items_to_string(self) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = ''
		for item in self.items:
			if item.clickable:
				formatted_text += f'{item.index}:{item.text}\n'
			else:
				formatted_text += f'_:{item.text}\n'
		return formatted_text
