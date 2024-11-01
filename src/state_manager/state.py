from selenium import webdriver
from selenium.webdriver.common.by import By
from typing import Dict, List
from bs4 import BeautifulSoup
from src.state_manager.utils import cleanup_html
import requests
from main_content_extractor import MainContentExtractor


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
        cleaned_content = cleanup_html(html_content)

        return {
            "current_url": self.driver.current_url,
            "page_title": self.driver.title,
            "interactable_elements": cleaned_content["html"],
            # "main_content": cleaned_content["main_content"]
        }

    def get_functions(self) -> List[Dict]:
        """
        Retrieves available functions from cleaned HTML.
        """
        return []

    def get_main_content(self) -> str:
        """
        Retrieves main content from cleaned Markdown.
        """
        try:
            # Get HTML using requests
            response = requests.get(self.driver.current_url)
            response.encoding = 'utf-8'
            content = response.text

            # Get HTML with main content extracted from HTML
            # extracted_html = MainContentExtractor.extract(content)

            # Get HTML with main content extracted from Markdown
            extracted_markdown = MainContentExtractor.extract(content, output_format="markdown")

            return extracted_markdown
        except Exception as e:
            print(f"Error getting main content: {e}")
            return ""
