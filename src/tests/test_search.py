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
	timestamp = f'{timestamp_prefix}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
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


# clear && pytest src/tests/test_search.py -v -s
# clear && pytest src/tests/test_search.py::test_wikipedia_tabs -v -s
@pytest.mark.asyncio
@observe()
async def test_kayak_flight_search():
	task = 'go to kayak.com andfind a flight from Bali to Kirgistan on 2024-11-25 for 2 people one way.'
	run_folder = setup_run_folder('kayak_com_flight_search2')

	controller = ControllerService()

	model = ChatAnthropic(
		model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
	)
	model = ChatOpenAI(model='gpt-4o', temperature=0.3)

	agent = AgentService(
		task,
		model,
		controller,
		use_vision=True,
		save_file=f'{run_folder}/conversation',
	)

	print('\n' + '=' * 50)
	print('🚀 Starting flight search task')
	print('=' * 50)

	try:
		max_steps = 50
		for i in range(max_steps):
			print(f'\n📍 Step {i+1}')
			action, result = await agent.step()
			if result.done:
				print('\n✅ Task completed successfully')
				print('Extracted content:', result.extracted_content)
				break

			print('result:\n', result.model_dump_json(indent=4))

	except KeyboardInterrupt:
		print('\n\nReceived interrupt, closing browser...')
		controller.browser.close()
		raise
	except:
		print('\n' + '=' * 50)
		print('❌ Failed to complete task in maximum steps')
		print('=' * 50)
		assert False, 'Failed to complete task in maximum steps'
	finally:
		controller.browser.close()


# clear && pytest src/tests/test_search.py::test_wikipedia_tabs -v -s
@pytest.mark.asyncio
@observe()
async def test_wikipedia_tabs():
	# task = 'open 3 wikipedia pages in different tabs and then go back to the first one and summarize me the content of the page.'
	task = 'open 3 wikipedia pages in different tabs and summarize me the content of all pages.'
	run_folder = setup_run_folder('wikipedia_tabs')

	controller = ControllerService(keep_open=True)

	model = ChatAnthropic(
		model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
	)
	model = ChatOpenAI(model='gpt-4o', temperature=0.3)

	agent = AgentService(
		task, model, controller, use_vision=True, save_file=f'{run_folder}/conversation'
	)

	try:
		max_steps = 50
		for i in range(max_steps):
			print(f'\n📍 Step {i+1}')
			action, result = await agent.step()
			if result.done:
				print('\n✅ Task completed successfully')
				print('Extracted content:', result.extracted_content)
				break
			print('result:\n', result.model_dump_json(indent=4))

	except KeyboardInterrupt:
		print('\nReceived interrupt, closing browser...')
		controller.browser.close()
		raise
	except:
		print('\n' + '=' * 50)
		print('❌ Failed to complete task in maximum steps')
		print('=' * 50)
		assert False, 'Failed to complete task in maximum steps'
	finally:
		controller.browser.close()


# clear && pytest src/tests/test_search.py::test_albert_image_search -v -s
@pytest.mark.asyncio
@observe()
async def test_albert_image_search():
	# task = 'open 3 wikipedia pages in different tabs and then go back to the first one and summarize me the content of the page.'
	task = 'find an image of albert einstein and ask me for further instructions.'
	run_folder = setup_run_folder('albert_einstein_image_search')

	controller = ControllerService(keep_open=True)

	model = ChatAnthropic(
		model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
	)
	model = ChatOpenAI(model='gpt-4o', temperature=0.3)

	agent = AgentService(
		task, model, controller, use_vision=True, save_file=f'{run_folder}/conversation'
	)

	try:
		max_steps = 50
		for i in range(max_steps):
			print(f'\n📍 Step {i+1}')
			action, result = await agent.step()
			if result.done:
				print('\n✅ Task completed successfully')
				print('Extracted content:', result.extracted_content)
				break
			print('result:\n', result.model_dump_json(indent=4))

	except KeyboardInterrupt:
		print('\nReceived interrupt, closing browser...')
		controller.browser.close()
		raise
	except:
		print('\n' + '=' * 50)
		print('❌ Failed to complete task in maximum steps')
		print('=' * 50)
		assert False, 'Failed to complete task in maximum steps'
	finally:
		controller.browser.close()
