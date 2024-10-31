import unittest
from src.utils.selenium_utils import setup_selenium_driver
from src.actions.browser_actions import BrowserActions
from src.state_manager.state import StateManager


class TestKayakSearch(unittest.TestCase):
    def setUp(self):
        self.driver = setup_selenium_driver(headless=False)
        self.actions = BrowserActions(self.driver)
        self.state_manager = StateManager(self.driver)

    def tearDown(self):
        self.driver.quit()


if __name__ == '__main__':
    unittest.main()
