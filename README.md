<div align="center">

# 🌐 Browser-Use
### Open-Source Web Automation with LLMs

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

Let LLMs interact with websites through a simple interface.

## Short Example

```python
from src import Agent
from langchain_openai import ChatOpenAI

agent = Agent(
    task='Go to hackernews on show hn and give me top 10 post titels, their points and hours. Calculate for each the ratio of points per hour. All in a json format title, points, hours, ratio. Then ask me for further instructions.',
    llm=ChatOpenAI(model='gpt-4o'),
)

await agent.run()
```

## Demo
<div style="position: relative; padding-bottom: 64.92178098676294%; height: 0;"><iframe src="https://www.loom.com/embed/2af938b9f8024647950a9e18b3946054?sid=d95ece0b-6a17-477e-a223-b558645e6e89" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

<div>
    <a href="https://www.loom.com/share/2af938b9f8024647950a9e18b3946054">
      <p>gregpr07/browser-use at Example-improvement - 6 November 2024 - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/2af938b9f8024647950a9e18b3946054">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/2af938b9f8024647950a9e18b3946054-b99c733cf670e568-full-play.gif">
    </a>
    <p><i>Prompt: Search the top 3 AI companies 2024 and find what out what concrete hardware each is using for their model</i></p>
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

## Setup

1. Create a virtual environment and install dependencies:
```bash
# I recommend using uv
pip install -r requirements.txt
```

2. Add your API keys to the `.env` file:
```bash
cp .env.example .env
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

## Advanced Examples

### Chain of Agents

You can persist the browser across multiple agents and chain them together.

```python
from langchain_anthropic import ChatAnthropic
from src import Agent, Controller

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
Run examples directly from the command line:

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
python examples/try.py "Go to hackernews on show hn and give me top 10 post titels, their points and hours. Calculate for each the ratio of points per hour. Then ask me for further instructions." --provider anthropic
```

## 🤖 Supported Models

All LangChain chat models are supported. Tested with:
- GPT-4o
- GPT-4o Mini
- Claude 3.5 Sonnet
- LLama 3.1 405B

## Roadmap

- [x] Save agent actions and execute them deterministically
- [ ] Pydantic forced output
- [ ] Third party SERP API for faster Google Search results
- [ ] Multi-step action execution to increase speed
- [ ] Test on mind2web dataset
- [ ] Add more browser actions 

## Contributing

Contributions are welcome! Feel free to open issues for bugs or feature requests.

---

<div align="center">
  <b>Star ⭐ this repo if you find it useful!</b><br>
  Made with ❤️ by the Browser-Use team
</div>

