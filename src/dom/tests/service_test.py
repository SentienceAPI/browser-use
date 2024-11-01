import time

from src.dom.service import DomService
from src.driver.service import DriverService


def test_process_html_file():
	driver = DriverService().get_driver()

	driver.get('https://www.kayak.ch')
	# driver.get('https://example.com/')

	time.sleep(5)

	# Process the HTML file
	dom_service = DomService(driver)
	result = dom_service.get_current_state()

	# Add assertions based on expected content of page.html
	print(f'Processed DOM content: {result.output_string}')
	print(f'Selector map: {result.selector_map}')
