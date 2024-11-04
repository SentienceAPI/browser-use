import datetime
import os

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.decorators import observe

from src.agent.service import AgentService
from src.controller.service import ControllerService

langfuse = Langfuse(
	secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
	public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
	host=os.getenv('LANGFUSE_HOST'),
)


def setup_run_folder(timestamp_prefix: str) -> str:
	timestamp = f'{timestamp_prefix}_{datetime.datetime.now().strftime("%Y-%m-%d")}'
	run_folder = f'temp/{timestamp}'
	if not os.path.exists(run_folder):
		os.makedirs(run_folder)
	else:
		# Remove run folder if it exists
		for file in os.listdir(run_folder):
			os.remove(os.path.join(run_folder, file))
		os.rmdir(run_folder)
		os.makedirs(run_folder)
	return run_folder


# run with pytest src/tests/test_kayak_search.py -v -s
# @pytest.mark.skip
@pytest.mark.asyncio
@observe()
async def test_kayak_flight_search():
	task = 'go to kayak.com andfind a flight from Bali to Kirgistan on 2024-11-25 for 2 people one way.'
	run_folder = setup_run_folder('kayak_com_flight_search2')

	controller = ControllerService()

	model = ChatAnthropic(
		model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0
	)
	model = ChatOpenAI(model='gpt-4o', temperature=0.0)

	agent = AgentService(task, model, controller, use_vision=False)

	print('\n' + '=' * 50)
	print('🚀 Starting flight search task')
	print('=' * 50)

	try:
		max_steps = 50
		for i in range(max_steps):
			print(f'\n📍 Step {i+1}')
			action, result = await agent.step()

			print('action:\n', action)
			print('result:\n', result)

			# current_state = agent.get_current_state()
			# save_formatted_html(
			# 	current_state.interactable_elements, f'{run_folder}/current_state_{i}.html'
			# )
			# # save normal html
			# save_formatted_html(driver.page_source, f'{run_folder}/html_{i}.html')

			# url_history.append(driver.current_url)

			# state_text = f'Current interactive elements: {current_state.interactable_elements}'
			# if output.extracted_content:
			# 	state_text += f', Extracted content: {output.extracted_content}'
			# if output.user_input:
			# 	agent.update_system_prompt(output.user_input)
			# 	state_text += f', User input: {output.user_input}'
			# if output.error:
			# 	state_text += f', Previous action error: {output.error}'

			# input('Press Enter to continue...')

			# action = await agent.chat(
			# 	state_text,
			# 	images=current_state.screenshot,
			# 	store_conversation=f'{run_folder}/conversation_{i}.txt',
			# )
			# output: ActionResult = actions.execute_action(action, current_state.selector_map)

			# # check if output is exactly True (boolean)
			if result.done:
				print('\n✅ Task completed successfully')
				print('Extracted content:', result.extracted_content)
				break

			# time.sleep(0.5)
	except KeyboardInterrupt:
		print('\n\nReceived interrupt, closing browser...')
		controller.browser.close()
		raise
	else:
		print('\n' + '=' * 50)
		print('❌ Failed to complete task in maximum steps')
		print('=' * 50)
		assert False, 'Failed to complete task in maximum steps'
	finally:
		controller.browser.close()
