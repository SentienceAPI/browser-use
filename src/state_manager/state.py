from selenium import webdriver
from selenium.webdriver.common.by import By
from typing import Dict, List
from bs4 import BeautifulSoup
from .html_cleaner import cleanup_html


class StateManager:
    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver

    def get_current_state(self) -> Dict:
        """
        Retrieves current URL and interactable elements from cleaned HTML.

        Returns:
            Dict: Current state including URL, page title, and interactable elements
        """
        html_content = self.driver.page_source
        cleaned_html = cleanup_html(html_content)
        functions = self.get_functions()

        return {
            "current_url": self.driver.current_url,
            "page_title": self.driver.title,
            "interactable_elements": cleaned_html,
            "functions": functions
        }

    def get_functions(self) -> List[Dict]:
        """
        Retrieves available functions from cleaned HTML.
        """
        return []
