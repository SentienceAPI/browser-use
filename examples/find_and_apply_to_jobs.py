import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use.agent.service import Agent
from browser_use.browser.service import Browser
from browser_use.controller.service import Controller

# Initialize controller first
controller = Controller()


class Job(BaseModel):
	title: str
	link: str
	company: str
	salary: Optional[str] = None


class Jobs(BaseModel):
	jobs: List[Job]


@controller.action('Save jobs', param_model=Jobs)
def save_jobs(params: Jobs):
	with open('jobs.txt', 'a') as f:
		for job in params.jobs:
			f.write(f'{job.title} at {job.company}: {job.salary} ({job.link})\n')


# read from file
@controller.action('Read jobs from file')
def read_jobs():
	with open('jobs.txt', 'r') as f:
		return f.read()


# Without Pydantic model - using simple parameters
@controller.action('Ask me for information')
def ask_human(question: str) -> str:
	return input(f'\n{question}\nInput: ')


async def main():
	task = (
		'Find machine learning engineer jobs in Berlin and '
		'start applying for them over in new tabs - not via linkedin. '
		'If you need more information, ask me for it.'
	)
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
