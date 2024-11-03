from typing import Optional

from pydantic import BaseModel

from src.agent.views import AgentActions


class AskHumanPlanningAgentAction(BaseModel):
	question: str


class PlanningAgentAction(AgentActions):
	valuation_previous_goal: str
	goal: str

	ask_human: Optional[AskHumanPlanningAgentAction] = None

	@staticmethod
	def description() -> str:
		return (
			AgentActions.description()
			+ '\n\nAsk human for help\nExample: {"ask_human": {"question": "To clarify ..."}}'
		)
