import asyncio
from typing import List, Optional
import os

from langchain_openai import ChatOpenAI


from browser_use.agent.service import Agent
from browser_use.browser.service import Browser
from browser_use.controller.service import Controller

from pydantic import BaseModel
import dotenv
dotenv.load_dotenv()


controller = Controller()


class WebpageInfo(BaseModel):
    link: str = "https://appointment.mfa.gr/en/reservations/aero/ireland-grcon-dub/"



@controller.action("Go to the webpage", param_model=WebpageInfo)
def go_to_webpage(webpage_info: WebpageInfo):
    return webpage_info.link



async def main():
    task = (
        'Go to the Greece MFA webpage via the link I provided you.'
        'Check the visa appointment dates. If there is no available date in this month, check the next month.'
        'If there is no available date in both months, tell me there is no available date.'
    )

    model = ChatOpenAI(model='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    agent = Agent(task, model, controller, use_vision=True)
    
    result = await agent.run()


if __name__ == '__main__':
    asyncio.run(main())