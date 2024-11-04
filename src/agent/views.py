from typing import Optional

from pydantic import BaseModel

from src.controller.views import ControllerActions


class AskHumanAgentAction(BaseModel):
	question: str


class AgentOnlyAction(BaseModel):
	valuation_previous_goal: str
	goal: str

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
	print(AgentAction(valuation_previous_goal='Failed', goal='Click'))
