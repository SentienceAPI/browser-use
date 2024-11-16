# 🌐 Browser Use

Make websites accessible for AI agents 🤖.

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)

Browser use is the easiest way to connect your AI agents with the browser. If you have used Browser Use for your project feel free to show it off in our [Discord](https://link.browser-use.com/discord).

# Quick start

With pip:

```bash
pip install browser-use
```

Spin up your agent:

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

agent = Agent(
    task="Find a one-way flight from Bali to Oman on 12 January 2025 on Google Flights. Return me the cheapest option.",
    llm=ChatOpenAI(model="gpt-4o"),
)

# ... inside an async function
await agent.run()
```

And don't forget to add your API keys to your `.env` file.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

# Demos

<div>
    <a href="https://www.loom.com/share/63612b5994164cb1bb36938d62fe9983">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/63612b5994164cb1bb36938d62fe9983-7133f9e169672e6f-full-play.gif">
    </a>
    <p><i>Prompt: Go to hackernews on show hn and give me top 10 post titles, their points and hours. Calculate for each the ratio of points per hour. (1x speed) </i></p>
</div>
<div>
    <a href="https://www.loom.com/share/2af938b9f8024647950a9e18b3946054">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/2af938b9f8024647950a9e18b3946054-b99c733cf670e568-full-play.gif">
    </a>
    <p><i>Prompt: Search the top 3 AI companies 2024 and find what out what concrete hardware each is using for their model. (1x speed)</i></p>
</div>

<video>
    <source src="https://github.com/user-attachments/assets/6203bd11-71e6-40a2-b28e-e38b7cea3f93" >
</video>
aaa

<div>
    <video style="max-width:300px;" controls>
        <source src="https://github.com/user-attachments/assets/6203bd11-71e6-40a2-b28e-e38b7cea3f93" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <p><i>Solves amazon captcha and finds laptop prices.</i></p>
</div>
bbb
<div>
    <video style="max-width:300px;" controls>
        <source src="" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <p><i>Prompt: Find developer jobs in San Francisco at YC startups, and save them.</i></p>
</div>
ccc
<video controls>
    <source src="https://github.com/user-attachments/assets/6203bd11-71e6-40a2-b28e-e38b7cea3f93" type="video/mp4">
</video>
ddd

# Features ⭐

- Vision + html extraction
- Automatic multi-tab management
- Extract clicked elements XPaths
- Add custom actions (e.g. add data to database which the LLM can use)
- Self-correcting
- Use any LLM supported by LangChain (e.g. gpt4o, gpt4o mini, claude 3.5 sonnet, llama 3.1 405b, etc.)

## Register custom actions

If you want to add custom actions your agent can take, you can register them like this:

```python
from browser_use.agent.service import Agent
from browser_use.browser.service import Browser
from browser_use.controller.service import Controller

# Initialize controller first
controller = Controller()

@controller.action('Ask user for information')
def ask_human(question: str, display_question: bool) -> str:
	return input(f'\n{question}\nInput: ')
```

Or define your parameters using Pydantic

```python
class JobDetails(BaseModel):
title: str
company: str
job_link: str
salary: Optional[str] = None

@controller.action('Save job details which you found on page', param_model=JobDetails, requires_browser=True)
def save_job(params: JobDetails, browser: Browser):
	print(params)

  # use the browser normally
  browser.driver.get(params.job_link)
```

and then run your agent:

```python
model = ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3)
agent = Agent(task=task, llm=model, controller=controller)

await agent.run()
```

## Get XPath history

To get the entire history of everything the agent has done, you can use the output of the `run` method:

```python
history: list[AgentHistory] = await agent.run()

print(history)
```

## More examples

For more examples see the [examples](examples) folder or join the [Discord](https://link.browser-use.com/discord) and show off your project.

# Contributing

Contributions are welcome! Feel free to open issues for bugs or feature requests.

## Local Setup

1. Create a virtual environment and install dependencies:

```bash
# To install all dependencies including dev
pip install . ."[dev]"
```

2. Add your API keys to the `.env` file:

```bash
cp .env.example .env
```

or copy the following to your `.env` file:

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

You can use any LLM model supported by LangChain by adding the appropriate environment variables. See [langchain models](https://python.langchain.com/docs/integrations/chat/) for available options.

### Building the package

```bash
hatch build
```

Feel free to join the [Discord](https://link.browser-use.com/discord) for discussions and support.

---

<div align="center">
  <b>Star ⭐ this repo if you find it useful!</b><br>
  Made with ❤️ by the Browser-Use team
</div>
