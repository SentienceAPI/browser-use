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
    task='Find cheapest flight from London to Kyrgyzstan and return the url.',
    llm=ChatOpenAI(model='gpt-4o'),
)

await agent.run()
```

## Demo

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; margin-right: 10px;">
    <img style="max-width:100%;" src="./static/kayak.gif" alt="Kayak flight search demo">
    <p><i>Prompt: Go to kayak.com and find a one-way flight from Zürich to San Francisco on 12 January 2025.</i></p>
  </div>
  <div style="flex: 1; margin-left: 10px;">
    <img style="max-width:100%;" src="./static/photos.gif" alt="Photos search demo">
    <p><i>Prompt: Opening new tabs and searching for images for these people: Albert Einstein, Oprah Winfrey, Steve Jobs.</i></p>
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
python examples/try.py "Find cheapest flight from London to Paris" --provider anthropic
```

### OpenAI

You need to add `OPENAI_API_KEY` to your environment variables. Example usage:

```bash
python examples/try.py "Search for top AI companies" --provider openai
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
- [ ] Multi-step action execution
- [ ] Add more browser actions

## Contributing

Contributions are welcome! Feel free to open issues for bugs or feature requests.

---

<div align="center">
  <b>Star ⭐ this repo if you find it useful!</b><br>
  Made with ❤️ by the Browser-Use team
</div>

# Future Roadmap

- [x] Save agent actions and execute them deterministically (for QA testing etc)
- [ ] Pydantic forced output
- [ ] Third party SERP API for faster Google Search results

