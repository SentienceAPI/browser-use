import time
import unittest

from src.actions.browser_actions import BrowserActions
from src.agent_interface.planing_agent import PlaningAgent
from src.state_manager.state import StateManager
from src.state_manager.utils import save_formatted_html, save_markdown
from src.utils.selenium_utils import setup_selenium_driver


class TestKayakSearch(unittest.TestCase):
    def setUp(self):
        self.driver = setup_selenium_driver(headless=False)
        self.actions = BrowserActions(self.driver)
        self.state_manager = StateManager(self.driver)

    def tearDown(self):
        self.driver.quit()

    def test_kayak_flight_search(self):
        # Define the task
        task = 'Go to directly to the url https://www.kayak.ch/ and look there for flights from Zurich to New York for Nov 5th, 2024 for 2 people'

        default_actions = self.actions.get_default_actions()
        print(f'Default actions: {default_actions}')

        agent = PlaningAgent(task, default_actions)
        url_history = []

        # Main interaction loop
        max_steps = 10
        for i in range(max_steps):
            # Get current state
            input('\n\n\nPress Enter to continue...')

            current_state = self.state_manager.get_current_state()
            save_formatted_html(current_state['interactable_elements'], f'current_state_{i}.html')
            print(f'Current state map: {current_state["selector_map"]}')
            # save_markdown(current_state["main_content"], f"current_state_{i}.md")
            # Get next action from agent
            url_history.append(self.driver.current_url)
            text = f'Elements: {current_state["interactable_elements"]}, Url history: {url_history}'
            print(f'\n{text}\n')

            action = agent.chat(text, skip_call=False)
            print(f'Selected action: {action}')

            out = self.actions.execute_action(action, current_state['selector_map'])
            if out:
                print('Task completed')
                break

        else:
            self.fail('Failed to complete flight search task in maximum steps')


if __name__ == '__main__':
    unittest.main()
