from bs4 import BeautifulSoup
import re
import os
from bs4 import Comment, Tag


def cleanup_html(html_content):
    """
    Clean up html content by removing unnecessary HTML elements and formatting.
    Returns a dictionary with cleaned HTML and main content.
    """
    if not html_content:
        return {"html": "", "main_content": ""}

    soup = BeautifulSoup(html_content, "html.parser")
    class_counter = 0

    # Extract main content before cleaning
    main_content = ""
    main_element = soup.find('main') or soup.find(id='main')
    if main_element:
        main_content = main_element.get_text(strip=True)
    else:
        # If no main element, try to get content from body
        content_containers = soup.select('div[role="main"], div.content, div#content, article, .main-content')
        if content_containers:
            main_content = content_containers[0].get_text(strip=True)
        else:
            # Fallback to body text if no specific content container found
            body = soup.find('body')
            if body:
                main_content = body.get_text(strip=True)

    # Remove hidden elements and unnecessary content
    for element in soup.find_all():
        if element is None:
            continue

        # Remove if element is hidden
        if hasattr(element, 'attrs') and element.attrs and (
            element.get('aria-hidden') == 'true' or
            element.get('hidden') or
            element.get('style') and ('display: none' in element.get('style') or 'visibility: hidden' in element.get('style')) or
            element.get('class') and ('hidden' in ' '.join(element.get('class')) or
                                      'invisible' in ' '.join(element.get('class')))):
            element.decompose()
            continue

        # Remove all non-essential elements
        if element.name in ['script', 'style', 'noscript', 'iframe', 'link', 'meta',
                            'head', 'svg', 'path', 'defs', 'clipPath']:
            element.decompose()
            continue

    # Find and clean interactive elements
    interactive_selectors = [
        "button", "a", "input", "select", "textarea",
        "[role='button']", "[role='link']", "[role='combobox']",
        "[tabindex='0']", "[aria-haspopup]",
        "[class*='modal']", "[class*='popup']", "[class*='overlay']",
        "[id*='cookie']", "[class*='cookie']", "[aria-label*='cookie']"
    ]

    # First pass: mark overlay/popup elements
    overlay_elements = []
    for selector in interactive_selectors[9:]:  # Only popup/overlay selectors
        for element in soup.select(selector):
            if element and not element.get('c'):
                element["c"] = str(class_counter)
                class_counter += 1
                overlay_elements.append(element)

    # Second pass: mark other interactive elements if no overlays found
    if not overlay_elements:
        for selector in interactive_selectors[:9]:  # Non-popup selectors
            for element in soup.select(selector):
                if element is None or element.get('c'):
                    continue

                # Check if element is visible
                if hasattr(element, 'attrs') and element.attrs and (
                        element.get('aria-hidden') == 'true' or
                        element.get('hidden') or
                        element.get('style') and ('display: none' in element.get('style'))):
                    continue

                # Add c attribute to interactive element
                element["c"] = str(class_counter)
                class_counter += 1

    # Clean up attributes
    for element in soup.find_all():
        if element and element.get('c'):
            # Keep only essential attributes
            allowed_attrs = ["c", "href", "aria-label", "placeholder", "id", "type"]
            attrs = dict(element.attrs)
            for attr in attrs:
                if attr not in allowed_attrs:
                    del element[attr]

    # Get the HTML string and clean it up
    cleaned_html = str(soup)
    cleaned_html = " ".join(cleaned_html.split())

    # Remove empty tags with regex
    empty_tags_pattern = r"<[^/>][^>]*></[^>]+>"
    while re.search(empty_tags_pattern, cleaned_html):
        cleaned_html = re.sub(empty_tags_pattern, "", cleaned_html)

    return {
        "html": cleaned_html,
        "main_content": main_content
    }


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


def cleanup_html_old(html_content):
    """
    Clean up html content by removing unnecessary HTML elements and formatting.

    Args:
        html_content (str): Raw HTML content to clean

    Returns:
        str: Cleaned HTML content
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove unwanted tags completely
    unwanted_tags = ["style", "script", "font", "link", "meta"]
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()

    # Function to check if a tag only contains another single tag
    def is_wrapper_tag(tag):
        # Check if tag has exactly one child that is a tag
        children = list(tag.children)
        tag_children = [child for child in children if isinstance(child, Tag)]

        return (
            len(tag_children) == 1
            and len(children) == 1
            and not tag.get("id")  # Preserve tags with ID
            and not tag.get("class")  # Preserve tags with class
            and not tag.string  # Ensure there's no direct text content
        )

    # Remove nested wrapper tags
    modified = True
    while modified:
        modified = False
        # Look for just divs and spans
        for tag in soup.find_all(["div", "span"]):
            if is_wrapper_tag(tag):
                tag.replace_with(tag.contents[0])
                modified = True

    # Add counter for class enumeration
    class_counter = 0

    # Remove all attributes except allowed ones
    for tag in soup.find_all(True):
        allowed_attrs = [
            "id",
            "name",
            "aria-label",
            "aria-hidden",
            "href",
            "alt",
            # "class",
            # "data-node-index",
            # "data-p",
            # "class",
        ]
        attrs = dict(tag.attrs)
        for attr in attrs:
            if attr == "class":
                # Replace class with enumerated value
                tag["c"] = str(class_counter)
                class_counter += 1
                del tag[attr]
            elif attr not in allowed_attrs:
                del tag[attr]

    # Get the HTML string
    cleaned_html = str(soup)

    # Remove multiple spaces and newlines
    cleaned_html = " ".join(cleaned_html.split())

    # Remove empty tags
    empty_tags_pattern = r"<[^/>][^>]*></[^>]+>"
    cleaned_html = re.sub(empty_tags_pattern, "", cleaned_html)

    return cleaned_html
