import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from tokencost import count_string_tokens

from browser_use.browser.service import BrowserService
from browser_use.dom.service import DomService
from browser_use.utils import time_execution_sync


# @pytest.mark.skip("slow af")
def test_process_html_file():
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

	elements = time_execution_sync('get_clickable_elements')(
		dom_service.get_clickable_elements().dom_items_to_string
	)()

	print(elements)
	print('Tokens:', count_string_tokens(elements, model='gpt-4o'))

	input('Press Enter to continue...')
