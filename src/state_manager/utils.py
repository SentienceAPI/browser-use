from bs4 import BeautifulSoup, Tag
import re
import os


def save_formatted_html(html_content, output_file_name):
    """
    Format HTML content using BeautifulSoup and save to file

    Args:
        html_content (str): Raw HTML content to format
        output_file_name (str): Name of the file where formatted HTML will be saved
    """
    # Format HTML with BeautifulSoup for nice indentation
    soup = BeautifulSoup(html_content, 'html.parser')
    formatted_html = soup.prettify()

    # create temp folder if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Save formatted HTML to file
    with open("temp/"+output_file_name, 'w', encoding='utf-8') as f:
        f.write(formatted_html)


def save_markdown(markdown_content, output_file_name):
    """Save markdown content to a file"""
    if not os.path.exists("temp"):
        os.makedirs("temp")

    with open("temp/"+output_file_name, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


def cleanup_html(html_content):
    """
    Clean up HTML content by processing visible and interactive elements.
    Returns both cleaned HTML and main content.
    """
    if not html_content:
        return {"html": "", "main_content": ""}

    soup = BeautifulSoup(html_content, "html.parser")
    class_counter = 0

    def is_interactive_element(element):
        """Check if element is interactive based on tag or attributes"""
        interactive_tags = [
            "a", "button", "input", "select", "textarea", "details",
            "embed", "label", "menu", "menuitem", "object", "summary"
        ]

        interactive_roles = [
            "button", "link", "menuitem", "tab", "checkbox", "radio",
            "combobox", "listbox", "menu", "menuitem", "option", "slider"
        ]

        return (
            element.name in interactive_tags or
            element.get('role') in interactive_roles or
            element.get('tabindex') == '0' or
            element.get('onclick') or
            element.get('aria-haspopup')
        )

    def is_visible(element):
        """Check if element would be visible"""
        return not (
            element.get('aria-hidden') == 'true' or
            element.get('hidden') or
            element.get('style') and ('display: none' in element.get('style') or
                                      'visibility: hidden' in element.get('style')) or
            element.get('class') and ('hidden' in ' '.join(element.get('class')) or
                                      'invisible' in ' '.join(element.get('class')))
        )

    def is_leaf_element(element):
        """Check if element is a leaf node with meaningful content"""
        return (
            bool(element.string and element.string.strip()) and
            not element.find_all(True) and
            element.name not in ['script', 'style', 'meta', 'link', 'svg']
        )

    def process_element(element):
        """Process a single element and its children"""
        nonlocal class_counter

        if not isinstance(element, Tag):
            return

        # Skip hidden elements
        if not is_visible(element):
            element.decompose()
            return

        # Process interactive elements
        if is_interactive_element(element):
            if element.get('class'):
                element['c'] = str(class_counter)
                class_counter += 1
                del element['class']

            # Keep only essential attributes
            allowed_attrs = ["c", "href", "aria-label", "placeholder", "id", "type", "role"]
            attrs = dict(element.attrs)
            for attr in attrs:
                if attr not in allowed_attrs:
                    del element[attr]

        # Process leaf elements with content
        elif is_leaf_element(element):
            if element.get('class'):
                element['c'] = str(class_counter)
                class_counter += 1
                del element['class']

        # Remove unnecessary elements
        elif element.name in ['script', 'style', 'meta', 'link', 'svg', 'iframe']:
            element.decompose()
            return

        # Process children recursively
        for child in element.children:
            process_element(child)

    # Process the entire document
    process_element(soup.body)

    # Extract main content
    main_content = ""
    main_elem = soup.find('main') or soup.find(id='main')
    if main_elem:
        main_content = main_elem.get_text(strip=True)

    # Clean up the final HTML
    cleaned_html = str(soup)
    cleaned_html = " ".join(cleaned_html.split())

    # Remove empty tags
    empty_tags_pattern = r"<[^/>][^>]*></[^>]+>"
    while re.search(empty_tags_pattern, cleaned_html):
        cleaned_html = re.sub(empty_tags_pattern, "", cleaned_html)

    return {
        "html": cleaned_html,
        "main_content": main_content
    }
