import csv
import os
import sys
from pathlib import Path

from PyPDF2 import PdfReader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

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


@controller.action('Save jobs to file', param_model=Jobs)
def save_jobs(params: Jobs):
	with open('jobs.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		for job in params.jobs:
			writer.writerow([job.title, job.company, job.salary, job.link])


# read from file
@controller.action('Read jobs from file')
def read_jobs():
	with open('jobs.csv', 'r') as f:
		return f.read()


# Without Pydantic model - using simple parameters
@controller.action('Ask me for information / help')
def ask_human(question: str) -> str:
	return input(f'\n{question}\nInput: ')


@controller.action('Read my cv')
def read_cv():
	pdf = PdfReader('cv_04_24.pdf')

	text = ''

	for page in pdf.pages:
		text += page.extract_text() or ''

	return text


from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


@controller.action('Upload cv to index', requires_browser=True)
def upload_cv(index: int, browser: Browser):
	ActionChains(browser._get_driver()).send_keys(Keys.ESCAPE).perform()
	file_input = browser.wait_for_element('input[type="file"]')
	my_cv = Path.cwd() / 'cv_04_24.pdf'

	file_input.send_keys(my_cv.as_posix())

	return f'Uploaded cv to index {index}'


@controller.action('Close file dialog', requires_browser=True)
def close_file_dialog(browser: Browser):
	ActionChains(browser._get_driver()).send_keys(Keys.ESCAPE).perform()


async def main():
	# task = (
	# 	'Read my cv & find machine learning engineer jobs in Zurich for me. '
	# 	'Save them to a file'
	# 	'then start applying for them in new tabs - please not via job portals like linkedin, indeed, etc. '
	# 	'If you need more information or help, ask me.'
	# )

	task = 'go to https://jobs.lever.co/anybotics/edba1c62-9473-4043-8b1d-a353b2527587/apply# and Upload my cv '
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
