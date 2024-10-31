import unittest
from src.agent_interface.agent import Agent


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()

    def test_receive_state(self):
        state = {"current_url": "https://example.com"}
        actions = [{"type": "click", "params": {}}]
        self.agent.receive_state(state, actions)
        self.assertEqual(self.agent.current_state, state)
        self.assertEqual(self.agent.available_actions, actions)

    def test_decide_next_action(self):
        # Test will be implemented when decide_next_action is completed
        pass
