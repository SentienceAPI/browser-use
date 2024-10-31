from typing import Dict, List, Optional
import json


class Agent:
    def __init__(self):
        self.current_state = None
        self.available_actions = None
        self.task = None
        self.context = None
        self.task_progress = []

    def set_task(self, task: Dict):
        """
        Sets the task for the agent to complete.

        Args:
            task (Dict): Task description with type and parameters
        """
        self.task = task
        self.task_progress = []

    def receive_state(self, state: Dict, actions: List):
        """
        Receives the current state and possible actions.

        Args:
            state (Dict): Current browser state including URL and interactable elements
            actions (List): List of possible actions that can be taken
        """
        self.current_state = state
        self.available_actions = actions
        self.task_progress.append(state)
