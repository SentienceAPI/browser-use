from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class BrowserActions:
	def __init__(self, driver: webdriver.Chrome):
		self.driver = driver
		self.wait = WebDriverWait(driver, 10)

	def execute_action(self, action: dict):
		print(action.keys())
		action_name = action['action']

		if 'params' in action:
			params = action['params']
		else:
			params = {}

		if action_name == 'search_google':
			self.search_google(params['query'])
		elif action_name == 'go_to_url':
			self.go_to_url(params['url'])
		elif action_name == 'go_back':
			self.go_back()
		elif action_name == 'done':
			return True
		elif action_name == 'input':
			self.input_text_by_c(params['c'], params['text'])
		elif action_name == 'click':
			self.click_element_by_c(params['c'])
		elif action_name == 'accept_cookies':
			self.driver.accept_cookies()
		else:
			raise Exception(f'Action {action_name} not found')

	# default actions
	def search_google(self, query: str):
		"""
		Performs a Google search with the provided query.
		"""
		# TODO: replace with search api
		self.driver.get('https://www.google.com')
		search_box = self.wait.until(EC.presence_of_element_located((By.NAME, 'q')))
		search_box.send_keys(query)
		search_box.submit()

	def go_to_url(self, url: str):
		"""
		Navigates the browser to the specified URL.
		"""
		self.driver.get(url)

	def go_back(self):
		"""
		Navigates back in the browser history.
		"""
		self.driver.back()

	def click_element_by_c(self, c_value: str):
		"""
		Clicks an element identified by its c attribute.
		"""
		element = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, f'[c="{c_value}"]')))
		element.click()

	def input_text_by_c(self, c_value: str, text: str):
		"""
		Inputs text into a field identified by its c attribute.
		"""
		element = self.wait.until(
			EC.presence_of_element_located((By.CSS_SELECTOR, f'[c="{c_value}"]'))
		)
		element.clear()
		element.send_keys(text)

	def get_default_actions(self) -> dict[str, str]:
		return {
			'search_google': 'query: string',
			'go_to_url': 'url: string',
			'done': '',
			'go_back': '',
			'click': 'c: int',
			'input': 'c: int, text: string',
			'accept_cookies': '',
		}
