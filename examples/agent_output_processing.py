import os
import sys

from browser_use.controller.service import Controller

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o')

agent = Agent(
	task="Navigate to 'google.com' and type 'OpenAI' into the search bar. Then click the search button.",
	llm=llm,
	controller=Controller(keep_open=True, headless=False),
)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
