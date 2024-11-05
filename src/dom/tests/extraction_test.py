import time

from src.browser.service import BrowserService
from src.dom.service import DomService
from src.utils import time_execution_sync


# @pytest.mark.skip("slow af")
def test_process_html_file():
	browser = BrowserService(headless=False)

	driver = browser.init()

	dom_service = DomService(driver)

	browser.go_to_url('https://www.kayak.ch')

	time.sleep(3)

	elements = time_execution_sync('get_clickable_elements')(
		dom_service.get_clickable_elements().dom_items_to_string
	)()

	print(elements)
