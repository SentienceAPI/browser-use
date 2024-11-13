import base64
import os
import time

from browser_use.browser.service import BrowserService
from browser_use.dom.service import DomService
from browser_use.utils import time_execution_sync


def test_highlight_elements():
	browser = BrowserService(headless=False)

	driver = browser.init()

	dom_service = DomService(driver)

	browser.go_to_url('https://www.kayak.ch')
	# browser.go_to_url('https://google.com/flights')

	time.sleep(1)
	# browser._click_element_by_xpath(
	# 	'/html/body/div[5]/div/div[2]/div/div/div[3]/div/div[1]/button[1]'
	# )
	browser._click_element_by_xpath("//button[div/div[text()='Alle akzeptieren']]")

	elements = time_execution_sync('get_clickable_elements')(dom_service.get_clickable_elements)()

	# time_execution_sync('highlight_selector_map_elements')(browser.highlight_selector_map_elements)(
	# 	elements.selector_map
	# )

	image = time_execution_sync('take_screenshot')(browser.take_screenshot)(elements.selector_map)

	temp_image_path = os.path.join(os.path.dirname(__file__), 'temp', 'temp.png')
	with open(temp_image_path, 'wb') as f:
		f.write(base64.b64decode(image))

	# time_execution_sync('remove_highlights')(browser.remove_highlights)()

	input('Press Enter to continue...')
