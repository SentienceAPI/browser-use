import time
import pytest

from src.actions.browser_actions import BrowserActions
from src.agent_interface.planing_agent import PlaningAgent
from src.driver.service import DriverService
from src.state_manager.state import StateManager
from src.state_manager.utils import save_formatted_html


@pytest.fixture
async def setup():
	driver = DriverService().get_driver()
	actions = BrowserActions(driver)
	state_manager = StateManager(driver)
	yield driver, actions, state_manager
	driver.quit()


@pytest.mark.asyncio
async def test_kayak_flight_search(setup):
	driver, actions, state_manager = setup

	# Define the task
	task = 'Go to directly to the url ebay.com find a newly listed pre-owned man nike shoe in size 9 with free local pickup under 75 dollars.'

	default_actions = actions.get_default_actions()
	print(f'Default actions: {default_actions}')

	agent = PlaningAgent(task, str(default_actions), 'gpt-4o')
	url_history = []

	# Main interaction loop
	max_steps = 50
	for i in range(max_steps):
		# Get current state
		# input('\n\n\nPress Enter to continue...')

		current_state = state_manager.get_current_state()
		save_formatted_html(current_state.interactable_elements, f'current_state_{i}.html')
		print(f'Current state map: {current_state.selector_map}')
		# save_markdown(current_state["main_content"], f"current_state_{i}.md")
		# Get next action from agent
		url_history.append(driver.current_url)
		text = f'Elements: {current_state.interactable_elements}, Url history: {url_history}'
		print(f'\n{text}\n')

		action = await agent.chat(text, skip_call=False)
		print(f'Selected action: {action}')

		out = actions.execute_action(action, current_state.selector_map)
		if out:
			print('Task completed')
			break

		# Wait for 1 second
		time.sleep(1)

	else:
		assert False, 'Failed to complete flight search task in maximum steps'
