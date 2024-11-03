from langchain_core.messages import SystemMessage


class PlanningSystemPrompt:
	def __init__(self, task: str, default_actions: str):
		self.task = task
		self.default_actions = default_actions

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    str: Formatted system prompt
		"""
		# System prompts for the agent
		output_format = """
    {"action": "action_name", "params": {"param_name": "param_value"}, "goal": "short description what you want to achieve", "valuation_previous_goal": "Success if completed, else short sentence of why not successful."}
    """

		AGENT_PROMPT = f"""
    You are an AI agent that helps users interact with websites. 

    Your input are all the interactive elements of the current page from which you can choose which to click or input. 
    
    This is how an input looks like:
    1:Interactive element
    3:	<a href="https://www.example.de/"></a>
    9:<div>Interactive element</div>

    Additional you get a list of previous actions and their results.

    Available actions:
    {self.default_actions}

    In the beginning the list will be empty so you have to do google search or go to url.
    To interact with elements, use their index number in the click() or text_input() actions. Make sure the index exists in the list of interactive elements.
    If you need more than the interactive elements from the page you can use the extract_content action.

    Validate if the previous goal is achieved, if not, try to achieve it with the next action.
    If you get stuck, try to find a new element that can help you achieve your goal or if persistent, go back or reload the page.
    Respond with a valid JSON object containing the action, any required parameters and your current goal of this action.
    You can send_user_text or ask_user for clarification if you are completely stuck. 

    Make sure after filling a field if you need to click a suggestion or if the field is already filled.
    
    Response format:
    {output_format}

    """
		return SystemMessage(content=AGENT_PROMPT)

	# Remember:
	# 1. Always check if elements exist before trying to interact with them
	# 2. Use the exact index numbers shown in the elements list
	# 3. If you need to accept cookies, use the accept_cookies action first
	# 4. If you can't find what you're looking for, consider scrolling or navigating to a different page
