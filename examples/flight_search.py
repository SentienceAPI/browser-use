"""
@dev You need to add ANTHROPIC_API_KEY to your environment variables.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from browser_use.agent.service import AgentService
from browser_use.controller.service import ControllerService

task = 'Go to kayak.com and find a one-way flight from Zürich to San Francisco on 12 January 2025.'
controller = ControllerService()
# model = ChatAnthropic(
# 	model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
# )
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
