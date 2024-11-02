import os
import time

import pytest

from src.actions.browser_actions import ActionResult, BrowserActions
from src.agent_interface.planing_agent import PlaningAgent
from src.driver.service import DriverService
from src.state_manager.state import PageState, StateManager
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

	# Create run folder
	# timestamp = 'meta'
	# task = 'apply for a job at meta internship for ml research in sunny vale'
	timestamp = 'flight_search_bali_to_kirgistan'
	task = 'find a flight from Bali to Kirgistan on 2024-11-25 for 2 people one way.'
	vision = True

	run_folder = f'temp/{timestamp}'
	if not os.path.exists(run_folder):
		os.makedirs(run_folder)

	print('\n' + '=' * 50)
	print('🚀 Starting flight search task')
	print('=' * 50)

	default_actions = actions.get_default_actions()

	agent = PlaningAgent(task, str(default_actions), 'gpt-4o')
	agent.update_system_prompt(f'Your task is: {task}')

	url_history = []
	output = ActionResult()

	max_steps = 50
	for i in range(max_steps):
		print(f'\n📍 Step {i+1}')
		current_state: PageState = state_manager.get_current_state(run_folder)
		save_formatted_html(
			current_state.interactable_elements, f'{run_folder}/current_state_{i}.html'
		)
		# save normal html
		save_formatted_html(driver.page_source, f'{run_folder}/html_{i}.html')

		# show image and pause
		# if vision:
		# 	import matplotlib.pyplot as plt

		# 	# read path and show image
		# 	# image = plt.imread(current_state.screenshot)
		# 	# # no axis
		# 	# plt.axis('off')

		# 	# plt.imshow(image)
		# plt.pause(5)

		url_history.append(driver.current_url)

		state_text = f'Current interactive elements: {current_state.interactable_elements}'
		if output.extracted_content:
			state_text += f', Extracted content: {output.extracted_content}'
		if output.user_input:
			agent.update_system_prompt(output.user_input)
			state_text += f', User input: {output.user_input}'
		if output.error:
			state_text += f', Previous action error: {output.error}'

		action = await agent.chat(
			state_text, skip_call=False, store_conversation=f'{run_folder}/conversation_{i}.txt'
		)
		output: ActionResult = actions.execute_action(action, current_state.selector_map)

		# check if output is exactly True (boolean)
		if output.done:
			break

		time.sleep(0.5)

	else:
		print('\n' + '=' * 50)
		print('❌ Failed to complete task in maximum steps')
		print('=' * 50)
		assert False, 'Failed to complete task in maximum steps'
