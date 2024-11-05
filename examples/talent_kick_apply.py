"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

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
	task='Apply to 2025 batch of Talent Kick. Use dummy data. Here is some information about me: Name: John Doe, Email: john.doe@example.com, Phone: +1234567890, LinkedIn: https://www.linkedin.com/in/john-doe, Github: https://github.com/john-doe, Twitter: https://twitter.com/john-doe, StackOverflow: https://stackoverflow.com/users/123456/john-doe, Youtube: https://www.youtube.com/user/john-doe, Education: BSc Computer Science from MIT (2020-2024), Work Experience: Software Engineer at Google (2024-present), Skills: Python, JavaScript, React, Node.js, AWS, Docker, Kubernetes, Projects: Built an AI-powered chatbot with 10k+ users, Created an open-source library with 1k+ stars on Github, Achievements: Won first place in MIT Hackathon 2023, Published paper on ML at ICML 2024, Languages: English (Native), Spanish (Fluent), Mandarin (Basic), Interests: AI/ML, Open Source, Competitive Programming, Hobbies: Playing guitar, Rock climbing, Chess',
	llm=llm,
)


async def main():
	await agent.run()


asyncio.run(main())
