"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
	task='Find flights on kayak.com from Zurich to Beijing on 25.12.2024 to 02.02.2025',
	llm=llm,
)


async def main():
	await agent.run()


asyncio.run(main())
