import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from src import Agent

logging.basicConfig(level=logging.INFO)


llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
	task='find and open the websites of 5 kantonschulen in switzerland and return the urls.',
	llm=llm,
)


async def main():
	await agent.run()


asyncio.run(main())
