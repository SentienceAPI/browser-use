# Website Use

This repository contains an agent that can plan and execute actions on a website with or without a vision model.

# Installation

```bash
pip install website-use
```

# Supported models

- GPT-4o
- GPT-4o Mini
- Claude 3.5 Sonnet

<!-- We plan to add more models in the future (LLama 3). -->

# Codebase Structure

> The code structure inspired by https://github.com/Netflix/dispatch.

Very good structure on how to make a scalable codebase is also in [this repo](https://github.com/zhanymkanov/fastapi-best-practices).

Just a brief document about how we should structure our backend codebase.

## Code Structure

```markdown
src/
/<service name>/
models.py
services.py
prompts.py
views.py
utils.py
routers.py

    	/_<subservice name>/
```

### Service.py

Always a single file, except if it becomes too long - more than ~500 lines, split it into \_subservices

### Views.py

Always split the views into two parts

```python
# All
...

# Requests
...

# Responses
...
```

If too long → split into multiple files

### Prompts.py

Single file; if too long → split into multiple files (one prompt per file or so)

### Routers.py

Never split into more than one file

## Requirements

```bash
uv pip install -r requirements.txt
```

```bash
uv pipreqs --ignore .venv --force
```
