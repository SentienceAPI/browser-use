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
	task='Find a one-way flight from Bali to Oman on 12 January 2025 on Google Flights. Return me the cheapest option.',
	llm=llm,
)


async def main():
	await agent.run()


asyncio.run(main())
