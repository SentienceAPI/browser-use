# Product Requirements Document (PRD)

---

## **1. Project Overview**

**Project Name**: AgentWeb Foundation Layer

**Objective**: Develop a foundational layer that enables agents (specifically Large Language Models - LLMs) to interact with the web via a set of predefined browser actions. This layer allows developers to build higher-level planning agents capable of automating complex tasks, such as applying for jobs, data scraping, form filling, etc.

---

## **2. Goals and Objectives**

- **Modular Architecture**: Create a modular system with clear separation of concerns, allowing easy maintenance and scalability.
  
- **Detailed Documentation**: Provide comprehensive documentation, including JSON structures, function names, file names, and file structure.

- **Clean State Representation**: Maintain a minimal and clean state representation after each action, focusing on interactable elements.

- **Action Execution and Validation**: Execute actions reliably using Selenium and validate outcomes, handling ambiguities with LLM assistance.

- **Testing Layer**: Implement an abstract testing layer using the OpenAI API to simulate agent interactions.

---

## **3. Detailed Requirements**

### **3.1 Architecture Overview**

The system consists of the following components:

- **Agent Interface**: Handles communication with the LLM agent, providing state and receiving actions.

- **Action Executor**: Executes browser actions using Selenium WebDriver.

- **State Manager**: Manages the browser state, including current URL and interactable elements.

- **HTML Cleaner**: Cleans up HTML content to minimize token usage when interacting with LLMs.

- **Validation Module**: Validates the success of actions, potentially using a small LLM for ambiguity checks.

- **Testing Layer**: An abstract layer that uses the OpenAI API for testing agent interactions without actual web execution.

---

### **3.2 File Structure**

The project follows this directory structure:

```
agentweb/
├── env/                            # Virtual environment directory
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── src/
│   ├── agent_interface/
│   │   ├── __init__.py
│   │   ├── agent.py                # Agent class and logic
│   │   └── prompts.py              # Agent and state prompts
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── browser_actions.py      # Browser action implementations
│   │   └── action_validator.py     # Action validation logic
│   ├── state_manager/
│   │   ├── __init__.py
│   │   ├── state.py                # State management
│   │   └── html_cleaner.py         # HTML cleaning functions
│   ├── utils/
│   │   ├── __init__.py
│   │   └── selenium_utils.py       # Selenium setup and utilities
│   └── tests/
│       ├── __init__.py
│       ├── test_agent.py           # Tests for agent interface
│       ├── test_actions.py         # Tests for actions
│       ├── test_state_manager.py   # Tests for state manager
│       └── test_utils.py           # Tests for utilities
└── temp/                           # Temporary files (e.g., screenshots)
```

---

### **3.3 Module Descriptions**

#### **agent_interface/**

- **agent.py**

  ```python
  class Agent:
      def __init__(self):
          # Initialize agent state
          pass

      def receive_state(self, state: dict, actions: list):
          """
          Receives the current state and possible actions.
          """
          pass

      def decide_next_action(self) -> dict:
          """
          Determines the next action based on the current state.
          """
          pass
  ```

- **prompts.py**

  ```python
  AGENT_PROMPT = """
  You are a web scraping agent. Your task is to control the browser where, for every step, you get the current state and a list of actions you can take.

  Your task is:
  Apply for 50 jobs for software engineering in Zurich.

  Extra context:
  {extra_context}

  In every step, please only control the web browser with actions.
  Actions can also give you back follow-up questions about the state of the website if your instructions are ambiguous.
  """

  STATE_PROMPT = """
  {
      "state": {state},
      "actions": {actions}
  }
  """
  ```

#### **actions/**

- **browser_actions.py**

  ```python
  from selenium import webdriver

  class BrowserActions:
      def __init__(self, driver: webdriver.Chrome):
          self.driver = driver

      def search_google(self, query: str):
          # Implementation of Google search
          pass

      def go_to_url(self, url: str):
          # Navigate to specified URL
          pass

      def click_element(self, identifier: dict):
          # Click on an element based on identifier
          pass

      def input_text(self, identifier: dict, text: str):
          # Input text into a field
          pass

      def navigate_back(self):
          # Navigate back in browser history
          pass

      def reload_page(self):
          # Reload the current page
          pass

      def close_tab(self):
          # Close the current browser tab
          pass
  ```

- **action_validator.py**

  ```python
  class ActionValidator:
      def __init__(self, driver: webdriver.Chrome):
          self.driver = driver

      def is_action_successful(self, action: dict) -> bool:
          # Check if the action was successful
          pass

      def check_ambiguity(self, action: dict) -> bool:
          # Use LLM to check for ambiguity
          pass
  ```

#### **state_manager/**

- **state.py**

  ```python
  class StateManager:
      def __init__(self, driver: webdriver.Chrome):
          self.driver = driver

      def get_current_state(self) -> dict:
          # Retrieve current URL and interactable elements
          pass

      def get_interactable_elements(self) -> list:
          # Extract interactable elements from the page
          pass
  ```

- **html_cleaner.py**

  ```python
  import re
  from bs4 import BeautifulSoup, Comment, Tag

  def cleanup_html(html_content: str) -> str:
      """
      Clean up HTML content by removing unnecessary elements and formatting.
      """
      # Implementation as provided
      pass
  ```

#### **utils/**

- **selenium_utils.py**

  ```python
  from selenium import webdriver
  from selenium.webdriver.chrome.options import Options
  from webdriver_manager.chrome import ChromeDriverManager

  def setup_selenium_driver(headless: bool = False) -> webdriver.Chrome:
      """
      Sets up and returns a Selenium WebDriver instance.
      """
      # Configure Chrome options
      chrome_options = Options()
      if headless:
          chrome_options.add_argument("--headless")

      # Disable automation flags
      chrome_options.add_argument('--disable-blink-features=AutomationControlled')
      chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
      chrome_options.add_experimental_option('useAutomationExtension', False)

      # Initialize the Chrome driver
      driver = webdriver.Chrome(
          service=Service(ChromeDriverManager().install()),
          options=chrome_options
      )
      return driver
  ```

#### **tests/**

- **test_agent.py**

  ```python
  import unittest
  from src.agent_interface.agent import Agent

  class TestAgent(unittest.TestCase):
      def test_decide_next_action(self):
          # Test the agent's decision-making
          pass
  ```

- **test_actions.py**

  ```python
  import unittest
  from src.actions.browser_actions import BrowserActions

  class TestBrowserActions(unittest.TestCase):
      def test_search_google(self):
          # Test Google search action
          pass
  ```

- **test_state_manager.py**

  ```python
  import unittest
  from src.state_manager.state import StateManager

  class TestStateManager(unittest.TestCase):
      def test_get_current_state(self):
          # Test state retrieval
          pass
  ```

- **test_utils.py**

  ```python
  import unittest
  from src.utils.selenium_utils import setup_selenium_driver

  class TestSeleniumUtils(unittest.TestCase):
      def test_setup_selenium_driver(self):
          # Test Selenium driver setup
          pass
  ```

---

### **3.4 JSON Structures**

#### **Agent Prompt Structure**

```json
{
  "task": "Apply for 50 jobs for software engineering in Zurich.",
  "extra_context": "{user_resume_and_additional_info}",
  "state": {
    "current_url": "string",
    "interactable_elements": [
      {
        "type": "button",
        "identifier": {
          "id": "string",
          "class": "string",
          "name": "string",
          "text": "string"
        }
      },
      {
        "type": "input",
        "identifier": {
          "id": "string",
          "class": "string",
          "name": "string",
          "placeholder": "string"
        }
      },
      // Additional elements...
    ]
  },
  "actions": [
    {
      "type": "search_google",
      "params": {
        "query": "string"
      }
    },
    {
      "type": "go_to_url",
      "params": {
        "url": "string"
      }
    },
    {
      "type": "click_element",
      "params": {
        "identifier": {
          "id": "string",
          "class": "string",
          "name": "string",
          "text": "string"
        }
      }
    },
    // Additional actions...
  ]
}
```

#### **Action Response from Agent**

```json
{
  "action": {
    "type": "search_google",
    "params": {
      "query": "software engineering jobs in Zurich"
    }
  }
}
```

#### **State Representation**

```json
{
  "current_url": "https://www.example.com",
  "interactable_elements": [
    {
      "type": "button",
      "identifier": {
        "id": "submit-btn",
        "text": "Apply Now"
      }
    },
    {
      "type": "input",
      "identifier": {
        "name": "email",
        "placeholder": "Enter your email"
      }
    },
    // Additional elements...
  ]
}
```

---

### **3.5 Function Names and Descriptions**

- **Agent Class (`agent.py`):**

  - `receive_state(state: dict, actions: list)`: Receives the current state and possible actions from the system.

  - `decide_next_action() -> dict`: Determines the next action to execute based on the current state.

- **Browser Actions (`browser_actions.py`):**

  - `search_google(query: str)`: Performs a Google search with the provided query.

  - `go_to_url(url: str)`: Navigates the browser to the specified URL.

  - `click_element(identifier: dict)`: Clicks an element identified by attributes like ID, class, or text.

  - `input_text(identifier: dict, text: str)`: Inputs text into a field identified by attributes.

  - `navigate_back()`: Navigates back in the browser history.

  - `reload_page()`: Reloads the current page.

  - `close_tab()`: Closes the current browser tab.

- **Action Validator (`action_validator.py`):**

  - `is_action_successful(action: dict) -> bool`: Checks if the action was executed successfully.

  - `check_ambiguity(action: dict) -> bool`: Determines if the action is ambiguous and requires clarification.

- **State Manager (`state.py`):**

  - `get_current_state() -> dict`: Retrieves the current browser state, including URL and interactable elements.

  - `get_interactable_elements() -> list`: Extracts interactable elements from the page.

- **HTML Cleaner (`html_cleaner.py`):**

  - `cleanup_html(html_content: str) -> str`: Cleans up HTML content to reduce token usage.

- **Selenium Utils (`selenium_utils.py`):**

  - `setup_selenium_driver(headless: bool = False) -> webdriver.Chrome`: Sets up and returns a Selenium WebDriver instance.

---

## **4. Workflows**

### **4.1 Agent Interaction Flow**

1. **Initialization**: The agent is provided with a task and extra context.

2. **State Presentation**: The system sends the current state and possible actions to the agent.

3. **Agent Decision**: The agent decides on the next action and parameters.

4. **Action Execution**: The system executes the action using Selenium.

5. **State Update**: The state manager updates the current state.

6. **Validation**: The action validator checks if the action was successful.

   - If ambiguous, the agent is prompted for clarification.

7. **Loop**: Steps 2-6 repeat until the task is completed.

---

## **5. Testing**

### **5.1 Testing Layer**

- Implement an abstract testing layer using the OpenAI API to simulate agent decisions without actual web interactions.

### **5.2 Unit Tests**

- **Agent Interface**: Test agent's ability to process state and decide actions.

- **Browser Actions**: Test each browser action individually.

- **State Manager**: Test state retrieval and interactable element extraction.

- **HTML Cleaner**: Test HTML cleanup functionality.

### **5.3 Integration Tests**

- Simulate full agent workflows in a controlled environment.

### **5.4 Test Data**

- Use mock HTML pages and predefined states for consistent testing.

---

## **6. Dependencies**

- **Python Packages**:

  - `selenium`

  - `beautifulsoup4`

  - `webdriver-manager`

  - `openai` (for testing purposes)

  - `requests`

  - `tokencost` (for calculating token costs)

- **Browser Drivers**:

  - Chrome WebDriver (managed via `webdriver-manager`)

---

## **7. Environment Setup**

- **Virtual Environment**: Use `env/` directory for virtual environment setup.

- **Installation**:

  ```bash
  python -m venv env
  source env/bin/activate  # On Windows: env\Scripts\activate
  pip install -r requirements.txt
  ```

- **Requirements File (`requirements.txt`)**:

  ```
  selenium
  beautifulsoup4
  webdriver-manager
  openai
  requests
  tokencost
  ```

---

## **8. Additional Notes**

- **Token Efficiency**: The HTML cleaner reduces the amount of data sent to the LLM, minimizing token usage and costs.

- **Automation Detection Evasion**: Selenium is configured to avoid detection by disabling automation flags.

- **State Representation**: Focuses on interactable elements to allow the agent to make informed decisions without being overwhelmed by unnecessary data.

---

## **9. Milestones**

1. **Project Setup**: Establish the project structure and environment.

2. **Implement HTML Cleaner**: Integrate the `cleanup_html` function.

3. **Develop Browser Actions**: Implement basic browser actions in `browser_actions.py`.

4. **Build State Manager**: Develop state retrieval and element extraction in `state.py`.

5. **Create Agent Interface**: Implement the agent logic in `agent.py` and define prompts.

6. **Implement Action Validator**: Develop validation logic in `action_validator.py`.

7. **Set Up Testing Layer**: Use OpenAI API for simulating agent decisions.

8. **Write Unit Tests**: Cover all modules with unit tests.

9. **Integration Testing**: Perform end-to-end testing of agent workflows.

10. **Documentation**: Update `README.md` with usage instructions and API documentation.

---

## **10. Future Enhancements**

- **Advanced Actions**: Implement handling for complex web elements like dropdowns, modals, and file uploads.

- **Concurrency**: Support multiple agents interacting simultaneously.

- **Error Handling**: Enhance exception handling and logging mechanisms.

- **User Interface**: Develop a GUI or web dashboard for monitoring agent activities.

---

## **11. Questions for Clarification**

- **Specific Actions**: Are there any additional actions that need to be implemented beyond the ones listed?

- **LLM Integration**: Should the system support real-time interaction with LLMs during operation, or is the OpenAI API solely for testing?

- **Security Measures**: Are there any specific security protocols or data handling policies that need to be followed?

- **Data Storage**: Will there be a need to store any data persistently (e.g., logs, states), and if so, what storage solutions are preferred?

- **Deployment**: Is there a preferred deployment platform or containerization strategy (e.g., Docker)?

