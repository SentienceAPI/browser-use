"""
Simple try of the agent.

@dev You need to add ANTHROPIC_API_KEY to your environment variables.
"""

import logging
import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio

from src import Agent

logging.basicConfig(level=logging.INFO)


def get_llm(provider: str):
	if provider == 'anthropic':
		return ChatAnthropic(
			model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
		)
	elif provider == 'openai':
		return ChatOpenAI(model='gpt-4o', temperature=0.3)
	else:
		raise ValueError(f'Unsupported provider: {provider}')


parser = argparse.ArgumentParser()
parser.add_argument('query', type=str, help='The query to process')
parser.add_argument(
	'--provider',
	type=str,
	choices=['openai', 'anthropic'],
	default='openai',
	help='The model provider to use (default: openai)',
)

args = parser.parse_args()

llm = get_llm(args.provider)

agent = Agent(
	task=args.query,
	llm=llm,
)


async def main():
	result, history = await agent.run()
	print(result)
	print(history)


asyncio.run(main())
