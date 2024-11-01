"""
Dom Service
"""

from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel


class ProcessedDomContent(BaseModel):
    output_string: str
    selector_map: dict[int, str]


class DomService:
    def process_content(self, html_content: str) -> ProcessedDomContent:
        """
        Process HTML content to extract and clean relevant elements.
        Args:
            html_content: Raw HTML string to process
        Returns:
            ProcessedDomContent: Processed DOM content
        """
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        candidate_elements: list[Tag | NavigableString] = []
        dom_queue = list(soup.body.children) if soup.body else []
        xpath_cache = {}

        # Find candidate elements
        while dom_queue:
            element = dom_queue.pop()
            should_add_element = False

            # Handle both Tag elements and text nodes
            if isinstance(element, Tag):
                if self._is_element_accepted(element):
                    # Add children to queue in reverse order only if element is accepted
                    for child in reversed(list(element.children)):
                        dom_queue.append(child)

                    # Check if element is interactive or leaf element
                    if self._is_interactive_element(element) or self._is_leaf_element(element):
                        should_add_element = True

            elif isinstance(element, NavigableString) and element.strip():
                should_add_element = True

            if should_add_element:
                if not isinstance(element, (Tag, NavigableString)):
                    continue
                candidate_elements.append(element)

        # Process candidates
        selector_map = {}
        output_string = ''

        for index, element in enumerate(candidate_elements):
            xpath = xpath_cache.get(element)
            if not xpath:
                xpath = self._generate_xpath(element)
                xpath_cache[element] = xpath

            # Skip text nodes that are direct children of already processed elements
            if isinstance(element, NavigableString) and element.parent in candidate_elements:
                continue

            if isinstance(element, str):
                text_content = element.strip()
                if text_content:
                    output_string += f'{index}:{text_content}\n'
            else:
                tag_name = element.name
                attributes = self._get_essential_attributes(element)

                opening_tag = f"<{tag_name}{' ' + attributes if attributes else ''}>"
                closing_tag = f'</{tag_name}>'
                text_content = element.get_text().strip() or ''

                output_string += f'{index}:{opening_tag}{text_content}{closing_tag}\n'

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

        # Check for ID
        element_id = getattr(element, 'get', lambda x: None)('id')
        if element_id:
            return f"//[@id='{element_id}']"

        parts = []
        current = element

        while current and getattr(current, 'name', None):
            # Skip document node
            if current.name == '[document]':
                current = getattr(current, 'parent', None)
                continue

            parent = getattr(current, 'parent', None)
            if parent and hasattr(parent, 'children'):
                siblings = [s for s in parent.children
                            if hasattr(s, 'name') and s.name == current.name]

                if len(siblings) > 1:
                    try:
                        index = siblings.index(current) + 1
                        parts.insert(0, f"{current.name}[{index}]")
                    except ValueError:
                        parts.insert(0, current.name)
                else:
                    parts.insert(0, current.name)
            else:
                parts.insert(0, current.name)

            current = parent

        xpath = '//' + '/'.join(p for p in parts if p != '[document]')
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
            # 'class',
            'href',
            'src',
            'aria-label',
            'aria-name',
            'aria-role',
            'aria-description',
            'aria-expanded',
            'aria-haspopup',
        ]

        # Collect essential attributes that have values
        attrs = []
        for attr in essential_attributes:
            if attr in element.attrs:
                attrs.append(f'{attr}="{element[attr]}"')

        # Collect data- attributes
        # for attr in element.attrs:
        # 	if attr.startswith('data-'):
        # 		attrs.append(f'{attr}="{element[attr]}"')

        return ' '.join(attrs)
