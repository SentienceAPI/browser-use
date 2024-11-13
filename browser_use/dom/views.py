from pydantic import BaseModel


class DomContentItem(BaseModel):
	index: int
	text: str
	clickable: bool
	depth: int


SelectorMap = dict[int, str]


class ProcessedDomContent(BaseModel):
	items: list[DomContentItem]
	selector_map: SelectorMap

	def dom_items_to_string(self) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = ''
		for item in self.items:
			item_depth = '\t' * item.depth * 1
			if item.clickable:
				formatted_text += f'{item.index}:{item_depth}{item.text}\n'
			else:
				formatted_text += f'{item.index}:{item_depth}{item.text}\n'
		return formatted_text
