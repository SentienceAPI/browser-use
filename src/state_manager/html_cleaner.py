from bs4 import BeautifulSoup, Comment
import re


def cleanup_html(html_content):
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
