from selenium import webdriver
from selenium.webdriver.common.by import By
from typing import Dict, List


class StateManager:
    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver

    def get_current_state(self) -> Dict:
        """
        Retrieves current URL and interactable elements.

        Returns:
            Dict: Current state including URL and interactable elements
        """
        return {
            "current_url": self.driver.current_url,
            "interactable_elements": self.get_interactable_elements()
        }

    def get_interactable_elements(self) -> List:
        """
        Extracts interactable elements from the page.

        Returns:
            List: List of interactable elements with their properties
        """
        interactable_elements = []

        # Find buttons
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            interactable_elements.append({
                "type": "button",
                "identifier": {
                    "id": button.get_attribute("id"),
                    "class": button.get_attribute("class"),
                    "text": button.text
                }
            })

        # Find input fields
        inputs = self.driver.find_elements(By.TAG_NAME, "input")
        for input_field in inputs:
            interactable_elements.append({
                "type": "input",
                "identifier": {
                    "id": input_field.get_attribute("id"),
                    "name": input_field.get_attribute("name"),
                    "placeholder": input_field.get_attribute("placeholder")
                }
            })

        return interactable_elements
