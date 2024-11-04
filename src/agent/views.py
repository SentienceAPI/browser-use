from typing import Optional

from litellm import ConfigDict
from pydantic import BaseModel
from typing_extensions import Unpack

from src.controller.views import ControllerActions


class AskHumanAgentAction(BaseModel):
	question: str


class AgentOnlyAction(BaseModel):
	# TODO this is not really and action with function, but more an output only
	valuation_previous_goal: str
	memory: str
	next_goal: str

	ask_human: Optional[AskHumanAgentAction] = None

	@staticmethod
	def description() -> str:
		return """
- Ask human for help
  Example: {"ask_human": {"question": "To clarify ..."}}
"""


class AgentAction(ControllerActions, AgentOnlyAction):
	@staticmethod
	def description() -> str:
		return AgentOnlyAction.description() + ControllerActions.description()


if __name__ == '__main__':
	print(AgentAction(valuation_previous_goal='Failed', next_goal='Click', memory=''))
