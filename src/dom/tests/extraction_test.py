import time

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tokencost import count_string_tokens
from src.browser.service import BrowserService

from src.dom.service import DomService
from src.utils import time_execution_sync


# @pytest.mark.skip("slow af")
def test_process_html_file():
	browser = BrowserService(headless=False)

	driver = browser.init()

	dom_service = DomService(driver)

	# driver.get('https://www.kayak.ch')
	browser.go_to_url('https://www.kayak.ch')
	# driver.get('https://google.com/flights')

	# driver.get('https://example.com/')

	# driver.implicitly_wait()
	time.sleep(3)

	# Wait for accept cookies button to appear
	# accept_cookies_button = WebDriverWait(driver, 10).until(
	# 	EC.presence_of_element_located(
	# 		(By.XPATH, '/html/body/div[4]/div/div[2]/div/div/div[3]/div/div[1]/button[1]/div/div')
	# 	)
	# )

	elements = time_execution_sync('get_clickable_elements')(
		dom_service.get_clickable_elements().dom_items_to_string
	)()

	print(elements)

	# for item in result.items:
	# 	print(item)

	# accept_cookies_button.click()

	# Wait for banner to disappear
	# time.sleep(1)

	# # Process the HTML file
	# dom_service = DomService(driver)
	# result = dom_service.get_clickable_elements()

	# # Add assertions based on expected content of page.html
	# print(f'Processed DOM content: {result.output_html}')
	# # print(f'Selector map: {result.selector_map}')

	# print('Tokens:', count_string_tokens(result.output_html, 'gpt-4o'))

	# input('Press enter to exit')
