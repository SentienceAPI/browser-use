import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_anthropic import ChatAnthropic

from src import Agent, Controller

logging.basicConfig(level=logging.INFO)


# Persist the browser state across agents
controller = Controller()

# Initialize browser agent
agent1 = Agent(
	task='Open 5 VCs websites in the New York area.',
	llm=ChatAnthropic(
		model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
	),
	controller=controller,
)
agent2 = Agent(
	task='Give me the names of the founders of the companies in all tabs.',
	llm=ChatAnthropic(
		model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
	),
	controller=controller,
)


# Let it work its magic
async def main():
	await agent1.run()
	founders = await agent2.run()

	print(founders)


asyncio.run(main())
