from typing import Dict, List

import requests
from pydantic import BaseModel
from Screenshot import Screenshot
from selenium import webdriver

from src.dom.service import DomService


class ProcessedContent(BaseModel):
	output_string: str
	selector_map: Dict[int, str]


class PageState(BaseModel):
	current_url: str
	page_title: str
	interactable_elements: str
	selector_map: Dict[int, str]
	screenshot: str


class StateManager:
	def __init__(self, driver: webdriver.Chrome):
		self.driver = driver
		self.dom_service = DomService(driver)
		self.ob = Screenshot.Screenshot()

	def get_current_state(self, run_folder: str, full_page_screenshot: bool = False) -> PageState:
		"""
		Retrieves current URL and interactable elements from processed HTML.

		Returns:
		    PageState: Current state including URL, page title, interactable elements and selector map
		"""
		processed_content = self.dom_service.get_current_state()

		if full_page_screenshot:
			screenshot = self.ob.full_screenshot(
				self.driver,
				save_path=run_folder,
				image_name='selenium_full_screenshot.png',
				is_load_at_runtime=True,
				load_wait_time=0,
			)
			screenshot = run_folder + '/selenium_full_screenshot.png'
		else:
			file_name = run_folder + '/window_screenshot.png'
			screenshot = self.driver.get_screenshot_as_file(file_name)
			screenshot = file_name if screenshot else ''

		print(screenshot)
		return PageState(
			current_url=self.driver.current_url,
			page_title=self.driver.title,
			interactable_elements=processed_content.output_string,
			selector_map=processed_content.selector_map,
			screenshot=screenshot,
		)

	def get_functions(self) -> List[Dict]:
		"""
		Retrieves available functions from cleaned HTML.
		"""
		return []

	def get_main_content(self) -> ProcessedContent:
		"""
		Retrieves main content from the page.
		"""
		try:
			# Get HTML using requests
			response = requests.get(self.driver.current_url)
			response.encoding = 'utf-8'
			content = response.text

			# Process content using DomService
			processed_content = self.dom_service._process_content(content)
			return ProcessedContent(
				output_string=processed_content.output_string,
				selector_map=processed_content.selector_map,
			)

		except Exception as e:
			print(f'Error getting main content: {e}')
			return ProcessedContent(output_string='', selector_map={})
