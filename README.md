<div align="center">

# 🌐 Browser-Use

### Open-Source Web Automation with LLMs

<!-- <p align="center">
  <img src="assets/demo.gif" alt="Browser-Use Demo" width="600">
</p> -->

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

_Let LLMs interact with websites naturally_

[Key Features](#-key-features) •
[Live Demos](#-live-demos) •
[Quick Start](#-quick-start) •
[Examples](#-examples) •
[Models](#-supported-models)

</div>

---

## 🎥 Live Demos

Watch Browser-Use tackle real-world tasks:

<table>
  <tr>
    <td width="33%" align="center">
      <a href="your_loom_link_1">
        <img src="assets/jobs-demo.png" alt="Jobs Demo" width="200"><br>
        <b>Job Applications</b>
      </a>
      <br>Apply to 5 SF tech jobs
    </td>
    <td width="33%" align="center">
      <a href="your_loom_link_2">
        <img src="assets/images-demo.png" alt="Images Demo" width="200"><br>
        <b>Multi-Tab Search</b>
      </a>
      <br>Find images across tabs
    </td>
    <td width="33%" align="center">
      <a href="your_loom_link_3">
        <img src="assets/flights-demo.png" alt="Flights Demo" width="200"><br>
        <b>Flight Search</b>
      </a>
      <br>Find cheapest flights
    </td>
  </tr>
</table>

## 🚀 Key Features

- 🤖 **Universal LLM Support** - Works with any Language Model
- 🎯 **Smart Element Detection** - Automatically finds interactive elements
- 📑 **Multi-Tab Management** - Seamless handling of browser tabs
- 🔍 **XPath Extraction** - No more manual DevTools inspection
- 👁️ **Vision Model Support** - Process visual page information
- 🛠️ **Customizable Actions** - Add your own browser interactions

## 💻 Quick Start

Create a virtual environment:

```bash
uv venv
```

Then install the dependencies:

```bash
# Install with uv (recommended)
uv pip install -r requirements.txt
```

Add your API keys to the `.env` file.

```bash
cp .env.example .env
```

You can use any LLM model that is supported by LangChain by adding correct environment variables. Head over to the [langchain models](https://python.langchain.com/docs/integrations/chat/) page to see all available models.

## 📝 Examples

```python
from src import Agent
from langchain_openai import ChatOpenAI

# Initialize browser agent
agent = Agent(
	task='Find cheapest flight from London to Kyrgyzstan and return the url.',
	llm=ChatOpenAI(model='gpt-4o'),
)

# Let it work its magic
await agent.run()
```

### Chain of Agents

You can persist the browser across multiple agents and chain them together.

```python
from langchain_anthropic import ChatAnthropic
from src import Agent, Controller

# Persist the browser state across agents
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

# Let it work its magic
await agent1.run()
founders, history = await agent2.run()

print(founders)
```

You can use the `history` to run the agents again deterministically.

## 🤖 Supported Models

All LangChain chat models are supported.

### Tested

- GPT-4o
- GPT-4o Mini
- Claude 3.5 Sonnet
- LLama 3.1 405B

## 🤝 Contributing

Contributions are welcome! Also feel free to open issues for any bugs or feature requests.

---

<div align="center">
  <b>Star ⭐ this repo if you find it useful!</b><br>
  Made with ❤️ by the Browser-Use team
</div>

# Future Roadmap

- [x] Save agent actions and execute them deterministically (for QA testing etc)
- [ ] Pydantic forced output
- [ ] Third party SERP API for faster Google Search results
