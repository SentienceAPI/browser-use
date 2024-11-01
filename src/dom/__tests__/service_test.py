import os
from src.dom.service import DomService


def test_process_html_file():
	# Get the absolute path to the test file
	current_dir = os.path.dirname(os.path.abspath(__file__))
	test_file_path = os.path.join(current_dir, '..', '..', '..', 'temp', 'page.html')

	# Ensure test file exists
	assert os.path.exists(test_file_path), f'Test file not found at {test_file_path}'

	# Read the HTML file content
	with open(test_file_path, 'r', encoding='utf-8') as file:
		html_content = file.read()

	# Process the HTML file
	dom_service = DomService()
	result = dom_service.process_content(html_content)

	# Add assertions based on expected content of page.html
	print(f'Processed DOM content: {result.output_string}')
	print(f'Selector map: {result.selector_map}')
