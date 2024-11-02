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

	print('\n' + '=' * 50)
	print('🚀 Starting flight search task')
	print('=' * 50)

	task = (
		' find a flight from Zürich to Bali on 2024-11-25 with return on 2024-12-09 for 2 people.'
	)
	default_actions = actions.get_default_actions()

	agent = PlaningAgent(task, str(default_actions), 'gpt-4o')
	url_history = []

	max_steps = 50
	for i in range(max_steps):
		print(f'\n📍 Step {i+1}')
		current_state = state_manager.get_current_state()
		save_formatted_html(current_state.interactable_elements, f'current_state_{i}.html')
		# print(f'Current state map: {current_state.selector_map}')
		# save_markdown(current_state["main_content"], f"current_state_{i}.md")
		# Get next action from agent
		url_history.append(driver.current_url)
		state_text = f'Elements: {current_state.interactable_elements}, Url history: {url_history}'

		action = await agent.chat(state_text, skip_call=False)
		out = actions.execute_action(action, current_state.selector_map)

		if out:
			print('\n' + '=' * 50)
			print('✅ Task completed successfully!')
			print('=' * 50)
			break

		time.sleep(0.5)

	else:
		print('\n' + '=' * 50)
		print('❌ Failed to complete task in maximum steps')
		print('=' * 50)
		assert False, 'Failed to complete flight search task in maximum steps'
