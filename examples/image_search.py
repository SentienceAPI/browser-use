import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use.agent.service import AgentService
from browser_use.controller.service import ControllerService

people = ['Albert Einstein', 'Oprah Winfrey', 'Steve Jobs']
task = f'Opening new tabs and searching for images for these people: {", ".join(people)}. Then ask me for further instructions.'
controller = ControllerService(keep_open=True)
model = ChatOpenAI(model='gpt-4o')
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
