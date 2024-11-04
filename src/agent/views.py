from typing import Optional

from pydantic import BaseModel

from src.controller.views import ControllerActions


class AskHumanAgentAction(BaseModel):
	question: str


class AgentState(BaseModel):
	valuation_previous_goal: str
	memory: str
	next_goal: str


class AgentOnlyAction(BaseModel):
	ask_human: Optional[AskHumanAgentAction] = None

	@staticmethod
	def description() -> str:
		return """
- Ask human for help
  Example: {"ask_human": {"question": "To clarify ..."}}
"""


class AgentOutput(ControllerActions, AgentOnlyAction):
	@staticmethod
	def description() -> str:
		return AgentOnlyAction.description() + ControllerActions.description()
		#


class Output(BaseModel):
	current_state: AgentState
	action: AgentOutput


if __name__ == '__main__':
	print(
		Output(
			current_state=AgentState(
				valuation_previous_goal='Failed', next_goal='Click', memory=''
			),
			action=AgentOutput(),
		)
	)
