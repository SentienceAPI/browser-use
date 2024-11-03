from typing import Optional

from pydantic import BaseModel

from src.agent.views import AgentActions


class AskHumanPlanningAgentAction(BaseModel):
	question: str


class PlanningAgentAction(AgentActions):
	ask_human: Optional[AskHumanPlanningAgentAction] = None
