from typing import Dict, List

import requests
from pydantic import BaseModel
from selenium import webdriver

from src.dom.service import DomService


class ProcessedContent(BaseModel):
	output_string: str
	selector_map: Dict[str, str]


class PageState(BaseModel):
	current_url: str
	page_title: str
	interactable_elements: str
	selector_map: Dict[str, str]


class StateManager:
	def __init__(self, driver: webdriver.Chrome):
		self.driver = driver
		self.dom_service = DomService(driver)

	def get_current_state(self) -> PageState:
		"""
		Retrieves current URL and interactable elements from processed HTML.

		Returns:
		    PageState: Current state including URL, page title, interactable elements and selector map
		"""
		processed_content = self.dom_service.get_current_state()

		return PageState(
			current_url=self.driver.current_url,
			page_title=self.driver.title,
			interactable_elements=processed_content.output_string,
			selector_map=processed_content.selector_map,
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
