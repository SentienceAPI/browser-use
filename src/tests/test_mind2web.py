import datetime
import json
import os
from typing import Dict, List

import pytest
from langchain_openai import ChatOpenAI

from src.agent.service import AgentService
from src.controller.service import ControllerService

#


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


def load_mind2web_samples(num_samples: int = 5) -> List[Dict]:
	"""Load samples from mind2web dataset"""
	dataset_path = 'src/tests/data/mind2web_processed.json'

	with open(dataset_path, 'r', encoding='utf-8') as f:
		data = json.load(f)

	# Take first n samples or all if less available
	return data[: min(num_samples, len(data))]


@pytest.mark.asyncio
async def test_mind2web_samples():
	# Load samples
	samples = load_mind2web_samples(num_samples=5)
	run_folder = setup_run_folder('mind2web_test')

	# Track results
	results = {'successful': 0, 'failed': 0, 'errors': []}

	print('\n' + '=' * 50)
	print('🚀 Starting Mind2Web samples test')
	print(f'Testing {len(samples)} samples')
	print('=' * 50)

	for i, sample in enumerate(samples):
		print(f'\n📋 Testing sample {i+1}/{len(samples)}')
		task = f'Go to {sample["website"]}.com and {sample["confirmed_task"]}'
		print(f'Task: {task}')
		print('-' * 50)

		# Initialize new agent and model for each sample
		controller = ControllerService()
		model = ChatOpenAI(model='gpt-4o')
		agent = AgentService(task, model, controller, use_vision=True)

		try:
			max_steps = 50
			sample_success = False

			for step in range(max_steps):
				print(f'\n📍 Step {step+1}')
				action, result = await agent.step()

				print('Action:', action)
				print('Result:', result)

				# Save step details
				step_file = f'{run_folder}/sample_{i}_step_{step}.json'
				with open(step_file, 'w') as f:
					json.dump(
						{'step': step, 'action': str(action), 'result': str(result)}, f, indent=2
					)

				if result.done:
					print('\n✅ Sample completed successfully')
					results['successful'] += 1
					sample_success = True
					break

			if not sample_success:
				print('\n❌ Sample failed to complete in maximum steps')
				results['failed'] += 1
				results['errors'].append(
					{'sample': i, 'task': sample['task'], 'error': 'Max steps reached'}
				)

		except KeyboardInterrupt:
			print('\nReceived interrupt, closing browser...')
			controller.browser.close()
			raise
		except Exception as e:
			print(f'\n❌ Error processing sample: {str(e)}')
			results['failed'] += 1
			results['errors'].append({'sample': i, 'task': sample['task'], 'error': str(e)})
		finally:
			controller.browser.close()

	# Save final results
	results_file = f'{run_folder}/results.json'
	with open(results_file, 'w') as f:
		json.dump(
			{
				'total_samples': len(samples),
				'successful': results['successful'],
				'failed': results['failed'],
				'success_rate': results['successful'] / len(samples),
				'errors': results['errors'],
			},
			f,
			indent=2,
		)

	print('\n' + '=' * 50)
	print('📊 Test Results:')
	print(f'Total samples: {len(samples)}')
	print(f'Successful: {results["successful"]}')
	print(f'Failed: {results["failed"]}')
	print(f'Success rate: {(results["successful"] / len(samples)) * 100:.2f}%')
	print('=' * 50)

	# Assert overall success rate is acceptable
	min_success_rate = 0.7  # 70% success rate required
	assert (
		results['successful'] / len(samples) >= min_success_rate
	), f'Success rate {results["successful"] / len(samples):.2f} below minimum {min_success_rate}'


@pytest.mark.asyncio
async def test_single_mind2web_sample():
	"""Test a single sample from mind2web dataset for debugging"""
	samples = load_mind2web_samples(num_samples=1)
	sample = samples[0]
	run_folder = setup_run_folder('mind2web_single_test')
	task = f'Go to {sample["website"]}.com and {sample["confirmed_task"]}'

	print('\n' + '=' * 50)
	print('🚀 Testing single Mind2Web sample')
	print(f'Task: {task}')
	print('=' * 50)

	controller = ControllerService()
	model = ChatOpenAI(model='gpt-4o')
	agent = AgentService(task, model, controller, use_vision=True)

	try:
		max_steps = 50
		for step in range(max_steps):
			print(f'\n📍 Step {step+1}')
			action, result = await agent.step()

			print('Action:', action)
			print('Result:', result)

			# Save detailed step information
			step_file = f'{run_folder}/step_{step}.json'
			with open(step_file, 'w') as f:
				json.dump({'step': step, 'action': str(action), 'result': str(result)}, f, indent=2)

			if result.done:
				print('\n✅ Task completed successfully')
				break
		else:
			print('\n❌ Failed to complete task in maximum steps')
			assert False, 'Failed to complete task in maximum steps'

	except KeyboardInterrupt:
		print('\nReceived interrupt, closing browser...')
		controller.browser.close()
		raise
	finally:
		controller.browser.close()


if __name__ == '__main__':
	pytest.main([__file__, '-v', '-s'])
