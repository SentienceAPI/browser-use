from typing import Dict, List


class Agent:
    def __init__(self):
        self.current_state = None
        self.available_actions = None
        self.task = None
        self.context = None

    def receive_state(self, state: Dict, actions: List):
        """
        Receives the current state and possible actions.

        Args:
            state (Dict): Current browser state including URL and interactable elements
            actions (List): List of possible actions that can be taken
        """
        self.current_state = state
        self.available_actions = actions

    def decide_next_action(self) -> Dict:
        """
        Determines the next action based on the current state.

        Returns:
            Dict: Action to be executed with its parameters
        """
        # Implementation will use LLM to decide next action
        pass
