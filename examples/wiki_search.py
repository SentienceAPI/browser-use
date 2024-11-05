import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_anthropic import ChatAnthropic

from src.agent.service import AgentService
from src.controller.service import ControllerService

task = 'Open 3 wikipedia pages in different tabs and summarize the content of all pages.'
controller = ControllerService()
model = ChatAnthropic(
	model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
)
agent = AgentService(task, model, controller, use_vision=True)


async def main():
	max_steps = 50
	# Run the agent step by step
	for i in range(max_steps):
		print(f'\n📍 Step {i+1}')
		action, result = await agent.step()

		print('Action:', action)
		print('Result:', result)

		if result.done:
			print('\n✅ Task completed successfully!')
			print('Extracted content:', result.extracted_content)
			break


asyncio.run(main())
