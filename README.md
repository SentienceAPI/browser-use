<div align="center">

# 🌐 Browser-Use

### Open-Source Web Automation with LLMs

<!-- <p align="center">
  <img src="assets/demo.gif" alt="Browser-Use Demo" width="600">
</p> -->

[![GitHub stars](https://img.shields.io/github/stars/yourusername/browser-use?style=social)](https://github.com/yourusername/browser-use/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

*Let LLMs interact with websites naturally*

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

```bash
# Install with uv (recommended)
uv pip install -r requirements.txt

# Generate requirements
uv pipreqs --ignore .venv --force
```

## 📝 Examples

```python
from browser_use import AgentService, ControllerService
from langchain_anthropic import ChatAnthropic

# Initialize browser agent
agent = AgentService(
    task="Find cheapest flight from London to Kyrgyzstan",
    model=ChatAnthropic(model="claude-3-sonnet"),
    controller=ControllerService(),
    use_vision=True
)

# Let it work its magic
await agent.run()
```

## 🤖 Supported Models

<table>
  <tr>
    <td align="center"><b>GPT-4o</b></td>
    <td align="center"><b>GPT-4o Mini</b></td>
    <td align="center"><b>Claude 3.5 Sonnet</b></td>
  </tr>
</table>



## 🤝 Contributing

Contributions are welcome! 


---

<div align="center">
  <b>Star ⭐ this repo if you find it useful!</b><br>
  Made with ❤️ by the Browser-Use team
</div>