from typing import Dict, List

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

from src.dom.service import DomService


class StateManager:
    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver
        self.dom_service = DomService()

    def get_current_state(self) -> Dict:
        """
        Retrieves current URL and interactable elements from processed HTML.

        Returns:
            Dict: Current state including URL, page title, interactable elements and selector map
        """
        html_content = self.driver.page_source
        processed_content = self.dom_service.process_content(html_content)

        return {
            'current_url': self.driver.current_url,
            'page_title': self.driver.title,
            'interactable_elements': processed_content.output_string,
            'selector_map': processed_content.selector_map
        }

    def get_functions(self) -> List[Dict]:
        """
        Retrieves available functions from cleaned HTML.
        """
        return []

    def get_main_content(self) -> str:
        """
        Retrieves main content from the page.
        """
        try:
            # Get HTML using requests
            response = requests.get(self.driver.current_url)
            response.encoding = 'utf-8'
            content = response.text

            # Process content using DomService
            processed_content = self.dom_service.process_content(content)
            return processed_content.output_string

        except Exception as e:
            print(f'Error getting main content: {e}')
            return ''
