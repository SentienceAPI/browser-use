import logging

from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from selenium import webdriver
from selenium.webdriver.common.by import By

from browser_use.dom.views import DomContentItem, ProcessedDomContent
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DomService:
	def __init__(self, driver: webdriver.Chrome):
		self.driver = driver
		self.xpath_cache = {}  # Add cache at instance level

	def get_clickable_elements(self) -> ProcessedDomContent:
		# Clear xpath cache on each new DOM processing
		self.xpath_cache = {}
		html_content = self.driver.page_source
		return self._process_content(html_content)

	@time_execution_sync('--_process_content')
	def _process_content(self, html_content: str) -> ProcessedDomContent:
		"""
		Process HTML content to extract and clean relevant elements.
		Args:
		    html_content: Raw HTML string to process
		Returns:
		    ProcessedDomContent: Processed DOM content

		@dev TODO: instead of of using enumerated index, use random 4 digit numbers -> a bit more tokens BUT updates on the screen wont click on incorrect items -> tricky because you have to consider that same elements need to have the same index ...
		"""

		# Parse HTML content using BeautifulSoup with html.parser
		soup = BeautifulSoup(html_content, 'html.parser')

		candidate_elements: list[Tag | NavigableString] = []

		def _process_element(element: PageElement):
			should_add_element = False

			# if not self._quick_element_filter(element):
			# 	if isinstance(element, Tag):
			# 		element.decompose()
			# 	return

			if isinstance(element, Tag):
				# Don't add any children of non-interactive elements
				if not self._is_element_accepted(element):
					element.decompose()
					return

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
				if isinstance(element, (Tag, NavigableString)):
					candidate_elements.append(element)

			if isinstance(element, Tag):
				for child in element.children:
					_process_element(child)

		for element in soup.body.children if soup.body else []:
			_process_element(element)

		# Process candidates
		selector_map: dict[int, str] = {}
		output_items: list[DomContentItem] = []

		for index, element in enumerate(candidate_elements):
			xpath = self._generate_xpath(element)

			depth = max(len(xpath.split('/')) - 2, 0)

			# Skip text nodes that are direct children of already processed elements
			if isinstance(element, NavigableString) and element.parent in [
				e for e in candidate_elements
			]:
				continue

			if isinstance(element, NavigableString):
				text_content = self._cap_text_length(element.strip())
				if text_content:
					output_string = f'{text_content}'
					output_items.append(
						DomContentItem(
							index=index, text=output_string, clickable=False, depth=depth
						)
					)
					continue
				else:
					# don't add empty text nodes
					continue

			else:
				text_content = self._extract_text_from_all_children(element)

				tag_name = element.name
				attributes = self._get_essential_attributes(element)

				opening_tag = f"<{tag_name}{' ' + attributes if attributes else ''}>"
				closing_tag = f'</{tag_name}>'

				output_string = f'{opening_tag}{text_content}{closing_tag}'
				output_items.append(
					DomContentItem(index=index, text=output_string, clickable=True, depth=depth)
				)

			selector_map[index] = xpath

		# Remove all elements from selector map that are not in output items
		selector_map = {
			k: v for k, v in selector_map.items() if k in [i.index for i in output_items]
		}

		return ProcessedDomContent(items=output_items, selector_map=selector_map)

	def _cap_text_length(self, text: str, max_length: int = 150) -> str:
		if len(text) > max_length:
			half_length = max_length // 2
			return text[:half_length] + '...' + text[-half_length:]
		return text

	def _extract_text_from_all_children(self, element: Tag) -> str:
		# Tell BeautifulSoup that button tags can contain content
		# if not hasattr(element.parser, 'BUTTON_TAGS'):
		# 	element.parser.BUTTON_TAGS = set()

		text_content = ''
		for child in element.descendants:
			if isinstance(child, NavigableString):
				current_child_text = child.strip()
			else:
				current_child_text = child.get_text(strip=True)

			text_content += '\n' + current_child_text

		return self._cap_text_length(text_content.strip()) or ''

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
			# 'dialog',
			# 'div',
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
			# 'dialog',  # added
			# 'alertdialog',  # added
			'menuitemcheckbox',
			'menuitemradio',
		}

		return (
			element.name in interactive_elements
			or element.get('role') in interactive_roles
			or element.get('aria-role') in interactive_roles
			or element.get('tabindex') == '0'
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
		"""Check if element is accepted based on tag name and special cases."""
		leaf_element_deny_list = {'svg', 'iframe', 'script', 'style', 'link', 'meta'}

		# First check if it's in deny list
		if element.name in leaf_element_deny_list:
			return False

		return element.name not in leaf_element_deny_list

	def _generate_xpath(self, element: Tag | NavigableString) -> str:
		# Generate cache key based on element properties
		cache_key = None
		if isinstance(element, Tag):
			attributes = [
				element.get('id', ''),
				element.get('class', ''),
				element.name,
				element.get_text().strip(),
			]
			cache_key = '|'.join(str(attr) for attr in attributes)

			# Return cached xpath if exists
			if cache_key in self.xpath_cache:
				return self.xpath_cache[cache_key]

		if isinstance(element, NavigableString):
			if element.parent:
				return self._generate_xpath(element.parent)
			return ''

		if not hasattr(element, 'name'):
			return ''

		element_id = getattr(element, 'get', lambda x: None)('id')
		if element_id:
			xpath = f"//*[@id='{element_id}']"
			if cache_key:
				self.xpath_cache[cache_key] = xpath
			return xpath

		parts = []
		current = element

		while current and getattr(current, 'name', None):
			if current.name == '[document]':
				break

			parent = getattr(current, 'parent', None)
			if parent and hasattr(parent, 'children'):
				# Get only visible element nodes
				siblings = [
					s for s in parent.find_all(current.name, recursive=False) if isinstance(s, Tag)
				]

				if len(siblings) > 1:
					try:
						index = siblings.index(current) + 1
						parts.insert(0, f'{current.name}[{index}]')
					except ValueError:
						parts.insert(0, current.name)
				else:
					parts.insert(0, current.name)

			current = parent

		if parts and parts[0] != 'html':
			parts.insert(0, 'html')
		if len(parts) > 1 and parts[1] != 'body':
			parts.insert(1, 'body')

		xpath = '//' + '/'.join(parts)

		# Cache the generated xpath
		if cache_key:
			self.xpath_cache[cache_key] = xpath

		return xpath

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
			'class',
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
				element_attr = element[attr]
				if isinstance(element_attr, str):
					element_attr = element_attr[:50]
				elif isinstance(element_attr, (list, tuple)):
					element_attr = ' '.join(str(v)[:50] for v in element_attr)

				attrs.append(f'{attr}="{element_attr}"')

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
			logger.error(f'Error checking top element: {element}')
			return False

	def _is_active(self, element: Tag) -> bool:
		"""Check if element is active (not disabled)."""
		return not (
			element.get('disabled') is not None
			or element.get('hidden') is not None
			or element.get('aria-disabled') == 'true'
		)

	# def _quick_element_filter(self, element: PageElement) -> bool:
	# 	"""
	# 	Quick pre-filter to eliminate elements before expensive checks.
	# 	Returns True if element passes initial filtering.
	# 	"""
	# 	if isinstance(element, NavigableString):
	# 		# Quick check for empty or whitespace-only strings
	# 		return bool(element.strip())

	# 	if not isinstance(element, Tag):
	# 		return False

	# 	style = element.get('style')

	# 	# Quick attribute checks that would make element invisible/non-interactive
	# 	if any(
	# 		[
	# 			element.get('aria-hidden') == 'true',
	# 			element.get('hidden') is not None,
	# 			element.get('disabled') is not None,
	# 			style and ('display: none' in style or 'visibility: hidden' in style),
	# 			element.has_attr('class')
	# 			and any(cls in element['class'] for cls in ['hidden', 'invisible']),
	# 			# Common hidden class patterns
	# 			element.get('type') == 'hidden',
	# 		]
	# 	):
	# 		return False

	# 	# Skip elements that definitely won't be interactive or visible
	# 	non_interactive_display = ['none', 'hidden']
	# 	computed_style = element.get('style', '') or ''
	# 	if any(display in computed_style for display in non_interactive_display):
	# 		return False

	# 	return True
