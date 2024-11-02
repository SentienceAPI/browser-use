"""
Dom Service
"""

from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel
from selenium import webdriver


class ProcessedDomContent(BaseModel):
	output_string: str
	selector_map: dict[int, str]


class DomService:
	def __init__(self, driver: webdriver.Chrome):
		self.driver = driver

	def get_current_state(self) -> ProcessedDomContent:
		html_content = self.driver.page_source
		return self._process_content(html_content)

	def _process_content(self, html_content: str) -> ProcessedDomContent:
		"""
		Process HTML content to extract and clean relevant elements.
		Args:
		    html_content: Raw HTML string to process
		Returns:
		    ProcessedDomContent: Processed DOM content
		"""
		# Parse HTML content using BeautifulSoup
		soup = BeautifulSoup(html_content, 'html.parser')

		candidate_elements: list[tuple[Tag | NavigableString, int]] = []  # (element, depth)
		dom_queue = [
			(element, 0) for element in (soup.body.children if soup.body else [])
		]  # (element, depth)
		xpath_cache = {}

		# Find candidate elements
		while dom_queue:
			element, depth = dom_queue.pop()
			should_add_element = False

			# Handle both Tag elements and text nodes
			if isinstance(element, Tag):
				if self._is_element_accepted(element):
					# Add children to queue in reverse order with increased depth
					for child in reversed(list(element.children)):
						dom_queue.append((child, depth + 1))

					# Check if element is interactive or leaf element
					if self._is_interactive_element(element) or self._is_leaf_element(element):
						if (
							self._is_active(element)
							and self._is_top_element(element)
							and self._is_visible(element)
						):
							should_add_element = True

			elif isinstance(element, NavigableString) and element.strip():
				if self._is_visible(element):
					should_add_element = True

			if should_add_element:
				if not isinstance(element, (Tag, NavigableString)):
					continue
				candidate_elements.append((element, depth))

		# Process candidates
		selector_map: dict[int, str] = {}
		output_string = ''

		for index, (element, depth) in enumerate(candidate_elements):
			indent = '\t' * depth  # Create indentation based on depth

			xpath = xpath_cache.get(element)
			if not xpath:
				xpath = self._generate_xpath(element)
				xpath_cache[element] = xpath

			# Skip text nodes that are direct children of already processed elements
			if isinstance(element, NavigableString) and element.parent in [
				e for e, _ in candidate_elements
			]:
				continue

			if isinstance(element, NavigableString):
				text_content = element.strip()
				if text_content:
					output_string += f'{index}:{indent}{text_content}\n'
			else:
				tag_name = element.name
				attributes = self._get_essential_attributes(element)

				opening_tag = f"<{tag_name}{' ' + attributes if attributes else ''}>"
				closing_tag = f'</{tag_name}>'
				text_content = element.get_text().strip() or ''

				output_string += f'{index}:{indent}{opening_tag}{text_content}{closing_tag}\n'

			selector_map[index] = xpath

		return ProcessedDomContent(output_string=output_string, selector_map=selector_map)

	def _is_interactive_element(self, element: Tag) -> bool:
		"""Check if element is interactive based on tag name and attributes."""
		interactive_elements = {
			'a',
			'button',
			'details',
			'embed',
			'input',
			'label',
			'menu',
			'menuitem',
			'object',
			'select',
			'textarea',
			'summary',
		}
		interactive_roles = {
			'button',
			'menu',
			'menuitem',
			'link',
			'checkbox',
			'radio',
			'slider',
			'tab',
			'tabpanel',
			'textbox',
			'combobox',
			'grid',
			'listbox',
			'option',
			'progressbar',
			'scrollbar',
			'searchbox',
			'switch',
			'tree',
			'treeitem',
			'spinbutton',
			'tooltip',
		}

		return (
			element.name in interactive_elements
			or element.get('role') in interactive_roles
			or element.get('aria-role') in interactive_roles
		)

	def _is_leaf_element(self, element: Tag) -> bool:
		"""Check if element is a leaf element."""
		if not element.get_text(strip=True):
			return False

		if not list(element.children):
			return True

		# Check for simple text-only elements
		children = list(element.children)
		if len(children) == 1 and isinstance(children[0], str):
			return True

		return False

	def _is_element_accepted(self, element: Tag) -> bool:
		"""Check if element is accepted based on tag name."""
		leaf_element_deny_list = {'svg', 'iframe', 'script', 'style', 'link'}

		# First check if it's in deny list
		if element.name in leaf_element_deny_list:
			return False

		return element.name not in leaf_element_deny_list

	def _generate_xpath(self, element: Tag | NavigableString) -> str:
		"""Generate XPath for given element."""
		# Handle NavigableString - return parent's xpath
		if isinstance(element, NavigableString):
			if element.parent:
				return self._generate_xpath(element.parent)
			return ''

		# Handle elements that may not have all Tag methods
		if not hasattr(element, 'name'):
			return ''

		# Check for ID using getattr to handle elements without get() method
		element_id = getattr(element, 'get', lambda x: None)('id')
		if element_id:
			return f"//*[@id='{element_id}']"

		parts = []
		current = element

		while current and getattr(current, 'name', None):
			# Skip document node
			if current.name == '[document]':
				break

			# Safely get parent and children
			parent = getattr(current, 'parent', None)
			siblings = list(parent.children) if parent and hasattr(parent, 'children') else []

			# Filter siblings that are elements with matching names
			same_type_siblings = [
				s for s in siblings if hasattr(s, 'name') and s.name == current.name
			]

			if len(same_type_siblings) > 1:
				try:
					index = same_type_siblings.index(current) + 1
					parts.insert(0, f'{current.name}[{index}]')
				except ValueError:
					parts.insert(0, current.name)
			else:
				parts.insert(0, current.name)

			current = parent

		# Ensure we start with html and body tags for a complete path
		if parts and parts[0] != 'html':
			parts.insert(0, 'html')
		if len(parts) > 1 and parts[1] != 'body':
			parts.insert(1, 'body')

		return '//' + '/'.join(parts) if parts else ''

	def _get_essential_attributes(self, element: Tag) -> str:
		"""
		Collects essential attributes from an element.
		Args:
		    element: The BeautifulSoup PageElement
		Returns:
		    A string of formatted essential attributes
		"""
		essential_attributes = [
			'id',
			# 'class',
			'href',
			'src',
			'readonly',
			'disabled',
			'checked',
			'selected',
			'role',
			'type',  # Important for inputs, buttons
			'name',  # Important for form elements
			'value',  # Current value of form elements
			'placeholder',  # Helpful for understanding input purpose
			'title',  # Additional descriptive text
			'alt',  # Alternative text for images
			'for',  # Important for label associations
			'autocomplete',  # Form field behavior
		]

		# Collect essential attributes that have values
		attrs = []
		for attr in essential_attributes:
			if attr in element.attrs:
				attrs.append(f'{attr}="{element[attr]}"')

		state_attributes_prefixes = (
			'aria-',
			'data-',
		)

		# Collect data- attributes
		for attr in element.attrs:
			if attr.startswith(state_attributes_prefixes):
				attrs.append(f'{attr}="{element[attr]}"')

		return ' '.join(attrs)

	def _is_visible(self, element: Tag | NavigableString) -> bool:
		"""Check if element is visible using JavaScript."""
		if not isinstance(element, Tag):
			return self._is_text_visible(element)

		element_id = element.get('id', '')
		if element_id:
			js_selector = f'document.getElementById("{element_id}")'
		else:
			xpath = self._generate_xpath(element)
			js_selector = f'document.evaluate("{xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue'

		visibility_check = f"""
			return (function() {{
				const element = {js_selector};
				
				if (!element) {{
					return false;
				}}

				// Force return as boolean
				return Boolean(element.checkVisibility({{
					checkOpacity: true,
					checkVisibilityCSS: true
				}}));
			}}());
		"""

		try:
			# todo: parse responses with pydantic
			is_visible = self.driver.execute_script(visibility_check)
			return bool(is_visible)
		except Exception:
			return False

	def _is_text_visible(self, element: NavigableString) -> bool:
		"""Check if text node is visible using JavaScript."""
		parent = element.parent
		if not parent:
			return False

		xpath = self._generate_xpath(parent)
		visibility_check = f"""
			return (function() {{
				const parent = document.evaluate("{xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
				
				if (!parent) {{
					return false;
				}}
				
				const range = document.createRange();
				const textNode = parent.childNodes[{list(parent.children).index(element)}];
				range.selectNodeContents(textNode);
				const rect = range.getBoundingClientRect();
				
				if (rect.width === 0 || rect.height === 0 || 
					rect.top < 0 || rect.top > window.innerHeight) {{
					return false;
				}}
				
				// Force return as boolean
				return Boolean(parent.checkVisibility({{
					checkOpacity: true,
					checkVisibilityCSS: true
				}}));
			}}());
		"""
		try:
			is_visible = self.driver.execute_script(visibility_check)
			return bool(is_visible)
		except Exception:
			return False

	def _is_top_element(self, element: Tag | NavigableString, rect=None) -> bool:
		"""Check if element is the topmost at its position."""
		xpath = self._generate_xpath(element)
		check_top = f"""
			return (function() {{
				const elem = document.evaluate("{xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
				if (!elem) {{
					return false;
				}}
				
				const rect = elem.getBoundingClientRect();
				const points = [
					{{x: rect.left + rect.width * 0.25, y: rect.top + rect.height * 0.25}},
					{{x: rect.left + rect.width * 0.75, y: rect.top + rect.height * 0.25}}, 
					{{x: rect.left + rect.width * 0.25, y: rect.top + rect.height * 0.75}},
					{{x: rect.left + rect.width * 0.75, y: rect.top + rect.height * 0.75}},
					{{x: rect.left + rect.width / 2, y: rect.top + rect.height / 2}}
				];
				
				return Boolean(points.some(point => {{
					const topEl = document.elementFromPoint(point.x, point.y);
					let current = topEl;
					while (current && current !== document.body) {{
						if (current === elem) return true;
						current = current.parentElement;
					}}
					return false;
				}}));
			}}());
		"""
		try:
			is_top = self.driver.execute_script(check_top)
			return bool(is_top)
		except Exception:
			return False

	def _is_active(self, element: Tag) -> bool:
		"""Check if element is active (not disabled)."""
		return not (
			element.get('disabled') is not None
			or element.get('hidden') is not None
			or element.get('aria-disabled') == 'true'
		)
