from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class BrowserActions:
    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver
        self.wait = WebDriverWait(driver, 10)
        self.selector_map = {}  # Will store current selector map

    def update_selector_map(self, selector_map: dict):
        """Update the current selector map"""
        self.selector_map = selector_map

    def execute_action(self, action: dict, selector_map: dict):
        print(action.keys())
        action_name = action['action']
        self.update_selector_map(selector_map)

        if 'params' in action:
            params = action['params']
        else:
            params = {}

        if action_name == 'search_google':
            self.search_google(params['query'])
        elif action_name == 'nothing':
            pass
        elif action_name == 'go_to_url':
            self.go_to_url(params['url'])
        elif action_name == 'go_back':
            self.go_back()
        elif action_name == 'done':
            return True
        elif action_name == 'input':
            self.input_text_by_index(params['id'], params['text'])
        elif action_name == 'click':
            self.click_element_by_index(params['id'])
        # elif action_name == 'accept_cookies':
        #     self.driver.accept_cookies()
        else:
            raise Exception(f'Action {action_name} not found')

    def click_element_by_index(self, index: int):
        """
        Clicks an element using its index from the selector map.
        """
        if str(index) not in self.selector_map:
            raise Exception(f'Element index {index} not found in selector map')

        xpath = self.selector_map[str(index)]
        element = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        element.click()

    def input_text_by_index(self, index: int, text: str):
        """
        Inputs text into a field using its index from the selector map.
        """
        if str(index) not in self.selector_map:
            raise Exception(f'Element index {index} not found in selector map')

        xpath = self.selector_map[str(index)]
        element = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
        element.clear()
        element.send_keys(text)

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

    def get_default_actions(self) -> dict[str, str]:
        return {
            'search_google': 'query: string',
            'go_to_url': 'url: string',
            'done': '',
            'go_back': '',
            'click': 'id: int',
            'input': 'id: int, text: string',
            'nothing': '',
        }
        # 'accept_cookies': '',
