# Browser-Use: Open-Source Web Automation with LLMs

Browser-Use makes it easy for LLMs to interact with websites naturally. 

## 🎥 See it in Action

Watch these demos to see Browser-Use handling different scenarios:

1. [Job Application Automation](your_loom_link_1) - Apply for five software engineering jobs in San Francisco 
2. [Multi-Tab Image Search](your_loom_link_2) - Opening new tabs and searching for images of Albert Einstein, Oprah Winfrey and Steve Jobs
3. [Flight Search Automation](your_loom_link_3) - Find the cheapest flight from London to Kyrgyzstan for one person on December 25 (one-way)

## 🚀 Key Features

- **Universal LLM Support**: Works with any Language Model (OpenAI, Anthropic, etc.)
- **Extract interactive elements**: Automatically identifies and makes interactive elements accessible
- **Multi-Tab Management**: Handles multiple browser tabs seamlessly
- **XPath Extraction**: Captures element paths for custom scraping functions without manual DevTools inspection
- **Vision Model Support**: Can process visual information from web pages
- **Customizable Actions**: Extensible set of browser interactions and custom functions like adding items to database


# Supported models

- GPT-4o
- GPT-4o Mini
- Claude 3.5 Sonnet

<!-- We plan to add more models in the future (LLama 3). -->

## Requirements

```bash
uv pip install -r requirements.txt
```

```bash
uv pipreqs --ignore .venv --force
```
