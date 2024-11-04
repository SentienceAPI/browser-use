from typing import Optional

from pydantic import BaseModel

from src.controller.views import ControllerActions


class AskHumanPlanningAgentAction(BaseModel):
	question: str


class PlanningControllerAction(ControllerActions):
	valuation_previous_goal: str
	goal: str

	ask_human: Optional[AskHumanPlanningAgentAction] = None

	@staticmethod
	def description() -> str:
		return (
			ControllerActions.description()
			+ '\n\nAsk human for help\nExample: {"ask_human": {"question": "To clarify ..."}}'
		)
