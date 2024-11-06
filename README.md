<div align="center">

# 🌐 Browser-Use

### Open-Source Web Automation with LLMs

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://discord.gg/uaCtrbbv)

</div>

Let LLMs interact with websites through a simple interface.

## Short Example

```bash
pip install browser-use
```

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
from asyncio import run

load_dotenv()
agent = Agent(
    task="Go to hackernews on show hn and give me top 10 post titles, their points and hours. Calculate for each the ratio of points per hour.",
    llm=ChatOpenAI(model="gpt-4o"),
)

run(agent.run())
```

## Demo

<div>
    <a href="https://www.loom.com/share/63612b5994164cb1bb36938d62fe9983">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/63612b5994164cb1bb36938d62fe9983-11f47a9490613568-full-play.gif">
    </a>
    <p><i>Prompt: Go to hackernews on show hn and give me top 10 post titles, their points and hours. Calculate for each the ratio of points per hour. (1x speed) </i></p>
</div>

<div>
    <a href="https://www.loom.com/share/2af938b9f8024647950a9e18b3946054">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/2af938b9f8024647950a9e18b3946054-b99c733cf670e568-full-play.gif">
    </a>
    <p><i>Prompt: Search the top 3 AI companies 2024 and find what out what concrete hardware each is using for their model. (1x speed)</i></p>
</div>

<div style="display: flex; justify-content: space-between; margin-top: 20px;">
    <div style="flex: 1; margin-right: 10px;">
        <img style="width: 100%;" src="./static/kayak.gif" alt="Kayak flight search demo">
        <p><i>Prompt: Go to kayak.com and find a one-way flight from Zürich to San Francisco on 12 January 2025. (2.5x speed)</i></p>
    </div>
    <div style="flex: 1; margin-left: 10px;">
        <img style="width: 100%;" src="./static/photos.gif" alt="Photos search demo">
        <p><i>Prompt: Opening new tabs and searching for images for these people: Albert Einstein, Oprah Winfrey, Steve Jobs. (2.5x speed)</i></p>
    </div>
</div>
</div>

## Local Setup

1. Create a virtual environment and install dependencies:

```bash
# I recommend using uv
pip install .
```

2. Add your API keys to the `.env` file:

```bash
cp .env.example .env
```
E.g. for OpenAI:
```bash
OPENAI_API_KEY=
```

You can use any LLM model supported by LangChain by adding the appropriate environment variables. See [langchain models](https://python.langchain.com/docs/integrations/chat/) for available options.

## Features

- Universal LLM Support - Works with any Language Model
- Interactive Element Detection - Automatically finds interactive elements
- Multi-Tab Management - Seamless handling of browser tabs
- XPath Extraction for scraping functions - No more manual DevTools inspection
- Vision Model Support - Process visual page information
- Customizable Actions - Add your own browser interactions (e.g. add data to database which the LLM can use)
- Handles dynamic content - dont worry about cookies or changing content
- Chain-of-thought prompting with memory - Solve long-term tasks
- Self-correcting - If the LLM makes a mistake, the agent will self-correct its actions

## Advanced Examples

### Chain of Agents

You can persist the browser across multiple agents and chain them together.

```python

# Persist browser state across agents
controller = Controller()

# Initialize browser agent
agent1 = Agent(
	task='Open 5 VCs websites in the New York area.',
	llm=ChatAnthropic(model_name='claude-3-sonnet', timeout=25, stop=None, temperature=0.3),
	controller=controller,
)
agent2 = Agent(
	task='Give me the names of the founders of the companies in all tabs.',
	llm=ChatAnthropic(model_name='claude-3-sonnet', timeout=25, stop=None, temperature=0.3),
	controller=controller,
)

await agent1.run()
founders, history = await agent2.run()

print(founders)
```

You can use the `history` to run the agents again deterministically.

## Command Line Usage

Run examples directly from the command line (clone the repo first):

```bash
python examples/try.py "Your query here" --provider [openai|anthropic]
```

### Anthropic

You need to add `ANTHROPIC_API_KEY` to your environment variables. Example usage:

```bash

python examples/try.py "Search the top 3 AI companies 2024 and find out in 3 new tabs what hardware each is using for their models" --provider anthropic
```

### OpenAI

You need to add `OPENAI_API_KEY` to your environment variables. Example usage:

```bash
python examples/try.py "Go to hackernews on show hn and give me top 10 post titles, their points and hours. Calculate for each the ratio of points per hour. " --provider anthropic
```

## 🤖 Supported Models

All LangChain chat models are supported. Tested with:

- GPT-4o
- GPT-4o Mini
- Claude 3.5 Sonnet
- LLama 3.1 405B

## Limitations

- When extracting page content, the message length increases and the LLM gets slower.
- Currently one agent costs about 0.01$
- Sometimes it tries to repeat the same task over and over again.
- Some elements might not be extracted which you want to interact with.
- What should we focus on the most?
  - Robustness
  - Speed
  - Cost reduction

## Roadmap

- [x] Save agent actions and execute them deterministically
- [ ] Pydantic forced output
- [ ] Third party SERP API for faster Google Search results
- [ ] Multi-step action execution to increase speed
- [ ] Test on mind2web dataset
- [ ] Add more browser actions

## Contributing

Contributions are welcome! Feel free to open issues for bugs or feature requests.

Feel free to join the [Discord](https://discord.gg/Wy9qE4TKHZ) for discussions and support.

---

<div align="center">
  <b>Star ⭐ this repo if you find it useful!</b><br>
  Made with ❤️ by the Browser-Use team
</div>
