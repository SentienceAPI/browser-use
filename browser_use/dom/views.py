from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

# Avoid circular import issues
if TYPE_CHECKING:
	from .views import DOMElementNode


@dataclass(frozen=False)
class DOMBaseNode:
	is_visible: bool
	# Use None as default and set parent later to avoid circular reference issues
	parent: Optional['DOMElementNode']


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
	text: str
	type: str = 'TEXT_NODE'

	def has_parent_with_highlight_index(self) -> bool:
		current = self.parent
		while current is not None:
			if current.highlight_index is not None:
				return True
			current = current.parent
		return False


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
	tag_name: str
	xpath: str
	attributes: Dict[str, str]
	children: List[DOMBaseNode]
	is_interactive: bool = False
	is_top_element: bool = False
	iframe_context: Optional[str] = None
	shadow_root: bool = False
	highlight_index: Optional[int] = None

	def get_all_text_till_next_clickable_element(self) -> str:
		text_parts = []

		def collect_text(node: DOMBaseNode) -> None:
			# Skip this branch if we hit a highlighted element (except for the current node)
			if (
				isinstance(node, DOMElementNode)
				and node != self
				and node.highlight_index is not None
			):
				return

			if isinstance(node, DOMTextNode):
				text_parts.append(node.text)
			elif isinstance(node, DOMElementNode):
				for child in node.children:
					collect_text(child)

		collect_text(self)
		return '\n'.join(text_parts).strip()

	def clickable_elements_to_string(self, use_tabs: bool = True) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = []

		def process_node(node: DOMBaseNode, depth: int) -> None:
			indent = '\t' * depth if use_tabs else ''

			if isinstance(node, DOMElementNode):
				# Add element with highlight_index
				if node.highlight_index is not None:
					formatted_text.append(
						f'{node.highlight_index}[:]{indent}<{node.tag_name}>{node.get_all_text_till_next_clickable_element()}</{node.tag_name}>'
					)

				# Process children regardless
				for child in node.children:
					process_node(child, depth + 1)

			elif isinstance(node, DOMTextNode):
				# Add text only if it doesn't have a highlighted parent
				if not node.has_parent_with_highlight_index():
					formatted_text.append(f'_[:]{indent}{node.text}')

		process_node(self, 0)
		return '\n'.join(formatted_text)


class ElementTreeSerializer:
	@staticmethod
	def serialize_clickable_elements(element_tree: DOMElementNode) -> str:
		return element_tree.clickable_elements_to_string()

	# def reconstruct_html_from_clickable_elements(self, element_tree: ElementNode) -> str:
	# 	"""Convert the clickable elements back into HTML format."""
	# 	html_parts = []

	# 	def process_node(node: BaseNode) -> None:
	# 		if isinstance(node, ElementNode):
	# 			# Start tag
	# 			attributes = ' '.join(f'{k}="{v}"' for k, v in node.attributes.items())
	# 			tag_start = f'<{node.tag_name}'
	# 			if attributes:
	# 				tag_start += f' {attributes}'
	# 			html_parts.append(f'{tag_start}>')

	# 			# Process children
	# 			for child in node.children:
	# 				process_node(child)

	# 			# End tag
	# 			html_parts.append(f'</{node.tag_name}>')
	# 		elif isinstance(node, TextNode) and node.is_visible:
	# 			html_parts.append(node.text)

	# 	process_node(element_tree)
	# 	return ''.join(html_parts)


SelectorMap = dict[int, DOMElementNode]


@dataclass
class DOMState:
	element_tree: DOMElementNode
	selector_map: SelectorMap
