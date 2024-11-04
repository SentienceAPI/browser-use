from langchain_core.messages import HumanMessage, SystemMessage

from src.controller.views import ControllerPageState


class AgentSystemPrompt:
	def __init__(self, task: str, default_action_description: str):
		self.task = task
		self.default_action_description = default_action_description

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    str: Formatted system prompt
		"""
		# System prompts for the agent
		# 		output_format = """
		# {"valuation_previous_goal": "Success if completed, else short sentence of why not successful.", "goal": "short description what you want to achieve", "action": "action_name", "params": {"param_name": "param_value"}}
		#     """

		AGENT_PROMPT = f"""
    You are an AI agent that helps users interact with websites. 

    Your input are all the interactive elements of the current page from which you can choose which to click or input. 
    
    This is how an input looks like:
    1:Interactive element
    3:	<a href="https://www.ab.de/"></a>
    9:<div>Interactive element</div>

    Additional you get a list of previous actions and their results.

    Available actions (choose EXACTLY ONE, not 0 or 2):

    {self.default_action_description}

    In the beginning the list will be empty so you have to do google search or go to url.
    To interact with elements, use their index number in the click() or text_input() actions. Make sure the index exists in the list of interactive elements.
    If you need more than the interactive elements from the page you can use the extract_content action.
	At every step you HAVE to choose EXACTLY ONE action.

    Validate if the previous goal is achieved, if not, try to achieve it with the next action.
    If you get stuck, try to find a new element that can help you achieve your goal or if persistent, go back or reload the page.
    Respond with a valid JSON object containing the action, any required parameters and your current goal of this action.
    You can send_user_text or ask_user for clarification if you are completely stuck. 

    Make sure after filling a field if you need to click a suggestion or if the field is already filled.
    """
		return SystemMessage(content=AGENT_PROMPT)


class AgentMessagePrompt:
	def __init__(self, state: ControllerPageState):
		self.state = state

	def get_user_message(self) -> HumanMessage:
		state_description = f"""
Current url: {self.state.url}
		
Interactive elements:
{self.state.dom_items_to_string()}
        """

		if self.state.screenshot:
			# Format message for vision model
			return HumanMessage(
				content=[
					{'type': 'text', 'text': state_description},
					{
						'type': 'image_url',
						'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
					},
				]
			)

		return HumanMessage(content=state_description)

	def get_message_for_history(self) -> HumanMessage:
		return HumanMessage(content=f'Currently on url: {self.state.url}')
