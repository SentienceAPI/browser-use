import os
import time

import pytest
from src.browser.service import BrowserService
from src.planning.service import PlaningService


from src.state_manager.utils import encode_image, save_formatted_html
from langchain_openai import ChatOpenAI


# @pytest.mark.skip
@pytest.mark.asyncio
async def test_kayak_flight_search():
	task = 'go to kayak.com andfind a flight from Bali to Kirgistan on 2024-11-25 for 2 people one way.'
	timestamp = 'kayak_com_flight_search'

	browser = BrowserService()
	model = ChatOpenAI(model='gpt-4o')
	planning_service = PlaningService(task, model, browser)

	vision = True

	run_folder = f'temp/{timestamp}'
	if not os.path.exists(run_folder):
		os.makedirs(run_folder)

	else:
		# Remove run folder if it exists
		for file in os.listdir(run_folder):
			os.remove(os.path.join(run_folder, file))
		os.rmdir(run_folder)
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
		current_state: PageState = state_manager.get_current_state(run_folder=run_folder, step=i)
		save_formatted_html(
			current_state.interactable_elements, f'{run_folder}/current_state_{i}.html'
		)
		# save normal html
		save_formatted_html(driver.page_source, f'{run_folder}/html_{i}.html')

		url_history.append(driver.current_url)

		state_text = f'Current interactive elements: {current_state.interactable_elements}'
		if output.extracted_content:
			state_text += f', Extracted content: {output.extracted_content}'
		if output.user_input:
			agent.update_system_prompt(output.user_input)
			state_text += f', User input: {output.user_input}'
		if output.error:
			state_text += f', Previous action error: {output.error}'

		input('Press Enter to continue...')

		action = await agent.chat(
			state_text,
			images=current_state.screenshot,
			store_conversation=f'{run_folder}/conversation_{i}.txt',
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
