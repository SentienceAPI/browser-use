from langchain_core.messages import HumanMessage, SystemMessage

from browser_use.controller.views import ControllerPageState


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
    Your input are all the interactive elements with its context of the current page from.
    
    This is how an input looks like:
    33:\t<button>Clickable element</button>
    _: Not clickable, only for your context
	\t: Tab indent (1 tab for depth 1 etc.). This is to help you understand which elements belong to each other.


    In the beginning the list will be empty.
	On elements with _ you can not click.
	On elements with a index you can click.
    
	Additional you get a list of your previous actions.

    
	Respond with a valid JSON object, containing 2 keys: current_state and action.
    In current_state there are 3 keys:
	valuation_previous_goal: Evaluate if the previous goal was achieved or what went wrong. E.g. Failed because ...
	memory: This you can use as a memory to store where you are in your overall task. E.g. if you need to find 10 jobs, you can store the already found jobs here.
	next_goal: The next goal you want to achieve e.g. clicking on ... button to ...

	For the action choose EXACTLY ONE from the following list:
    {self.default_action_description}
    To interact with elements, use their index number from the input line. Write it in the click_element() or input_text() actions. 
	Make sure to only use indexes that are present in the list.
    If you need more text from the page you can use the extract_page_content action.

	If you see any cookie or accept privacy policy please always just accepted them without hesitation.

    If you evaluate repeatedly that you dont achieve the next_goal, try to find a new element that can help you achieve your task or if persistent, go back or reload the page and try a different approach.
    
	You can ask_human for clarification if you are completely stuck or if you really need more information. 

	If a picture is provided, use it to understand the context and the next action.
	
	If you are sure you are done you can extract_page_content to get the markdown content and in the next action call done() with the text of the requested result to end the task and wait for further instructions.

    """
		return SystemMessage(content=AGENT_PROMPT)


class AgentMessagePrompt:
	def __init__(self, state: ControllerPageState):
		self.state = state

	def get_user_message(self) -> HumanMessage:
		state_description = f"""
Current url: {self.state.url}
Available tabs:
{self.state.tabs}
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
