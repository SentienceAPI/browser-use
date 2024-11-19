import logging

from main_content_extractor import MainContentExtractor

from browser_use.agent.views import ActionModel, ActionResult
from browser_use.browser.service import Browser
from browser_use.browser.views import TabInfo
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import (
	ClickElementAction,
	DoneAction,
	ExtractPageContentAction,
	GoToUrlAction,
	InputTextAction,
	OpenTabAction,
	SearchGoogleAction,
	SwitchTabAction,
)
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


class Controller:
	def __init__(self, keep_open: bool = False):
		self.browser = Browser(keep_open=keep_open)
		self.registry = Registry()
		self._register_default_actions()

	def _register_default_actions(self):
		"""Register all default browser actions"""

		# Basic Navigation Actions
		@self.registry.action(
			'Search Google', param_model=SearchGoogleAction, requires_browser=True
		)
		async def search_google(params: SearchGoogleAction, browser: Browser):
			session = await browser.get_session()
			page = session.page
			await page.goto(f'https://www.google.com/search?q={params.query}')
			await browser.wait_for_page_load()

		@self.registry.action('Navigate to URL', param_model=GoToUrlAction, requires_browser=True)
		async def go_to_url(params: GoToUrlAction, browser: Browser):
			session = await browser.get_session()
			page = session.page
			await page.goto(params.url)
			await browser.wait_for_page_load()

		@self.registry.action('Go back', requires_browser=True)
		async def go_back(browser: Browser):
			session = await browser.get_session()
			page = session.page
			await page.go_back()
			await browser.wait_for_page_load()

		# Element Interaction Actions
		@self.registry.action(
			'Click element', param_model=ClickElementAction, requires_browser=True
		)
		async def click_element(params: ClickElementAction, browser: Browser):
			session = await browser.get_session()
			state = session.cached_state

			if params.index not in state.selector_map:
				print(state.selector_map)
				raise Exception(
					f'Element with index {params.index} does not exist - retry or use alternative actions'
				)

			xpath = state.selector_map[params.index]
			session = await browser.get_session()
			page = session.page
			initial_pages = len(page.context.pages)

			msg = None

			for _ in range(params.num_clicks):
				try:
					await browser._click_element_by_xpath(xpath)
					msg = f'🖱️  Clicked element {params.index}: {xpath}'
					if params.num_clicks > 1:
						msg += f' ({_ + 1}/{params.num_clicks} clicks)'
				except Exception as e:
					logger.warning(f'Element no longer available after {_ + 1} clicks: {str(e)}')
					break

			if len(page.context.pages) > initial_pages:
				await browser.handle_new_tab()

			return ActionResult(extracted_content=f'Clicked element {msg}')

		@self.registry.action('Input text', param_model=InputTextAction, requires_browser=True)
		async def input_text(params: InputTextAction, browser: Browser):
			session = await browser.get_session()
			state = session.cached_state

			if params.index not in state.selector_map:
				raise Exception(
					f'Element index {params.index} does not exist - retry or use alternative actions'
				)

			xpath = state.selector_map[params.index]
			await browser._input_text_by_xpath(xpath, params.text)
			msg = f'⌨️  Input text "{params.text}" into element {params.index}: {xpath}'
			return ActionResult(extracted_content=msg)

		# Tab Management Actions
		@self.registry.action('Switch tab', param_model=SwitchTabAction, requires_browser=True)
		async def switch_tab(params: SwitchTabAction, browser: Browser):
			session = await browser.get_session()
			page = session.page

			# Verify handle exists
			if params.page_id not in session.opened_tabs:
				raise ValueError(f'Tab {params.page_id} not found')

			# Only switch if we're not already on that tab
			if params.page_id != session.current_page_id:
				await browser.switch_to_tab(params.page_id)
				# Wait for tab to be ready
				await browser.wait_for_page_load()

			# Update and return tab info
			tab_info = TabInfo(page_id=params.page_id, url=page.url, title=await page.title())
			session.opened_tabs[params.page_id] = tab_info

		@self.registry.action('Open new tab', param_model=OpenTabAction, requires_browser=True)
		async def open_tab(params: OpenTabAction, browser: Browser):
			session = await browser.get_session()
			page = session.page
			await page.evaluate(f'window.open("{params.url}", "_blank");')
			await browser.wait_for_page_load()
			await browser.handle_new_tab()

		# Content Actions
		@self.registry.action(
			'Extract page content', param_model=ExtractPageContentAction, requires_browser=True
		)
		async def extract_content(params: ExtractPageContentAction, browser: Browser):
			session = await browser.get_session()
			page = session.page

			content = MainContentExtractor.extract(  # type: ignore
				html=await page.content(),
				output_format=params.value,
			)
			return ActionResult(extracted_content=content)

		@self.registry.action('Complete task', param_model=DoneAction, requires_browser=True)
		async def done(params: DoneAction, browser: Browser):
			session = await browser.get_session()
			state = session.cached_state
			logger.info(f'✅ Done on page {state.url}\n\n: {params.text}')
			return ActionResult(is_done=True, extracted_content=params.text)

	def action(self, description: str, **kwargs):
		"""Decorator for registering custom actions

		@param description: Describe the LLM what the function does (better description == better function calling)
		"""
		return self.registry.action(description, **kwargs)

	@time_execution_sync('--act')
	async def act(self, action: ActionModel) -> ActionResult:
		"""Execute an action"""
		try:
			for action_name, params in action.model_dump(exclude_unset=True).items():
				if params is not None:
					result = await self.registry.execute_action(
						action_name, params, browser=self.browser
					)
					if isinstance(result, str):
						return ActionResult(extracted_content=result)
					elif isinstance(result, ActionResult):
						return result
					elif result is None:
						return ActionResult()
					else:
						raise ValueError(f'Invalid action result type: {type(result)} of {result}')
			return ActionResult()
		except Exception as e:
			raise e
