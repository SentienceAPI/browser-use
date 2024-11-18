"""
Playwright browser on steroids.
"""

import base64
import logging
import time

from playwright.sync_api import Browser as PlaywrightBrowser
from playwright.sync_api import Page, sync_playwright

from browser_use.browser.views import BrowserState, TabInfo
from browser_use.dom.service import DomService
from browser_use.dom.views import SelectorMap
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


class Browser:
	def __init__(self, headless: bool = False):
		self.headless = headless
		self.MINIMUM_WAIT_TIME = 0.5
		self.MAXIMUM_WAIT_TIME = 5
		self._tab_cache: dict[str, TabInfo] = {}
		self._current_page_id = None

		# Initialize Playwright during construction
		self.playwright = sync_playwright().start()
		self.browser: PlaywrightBrowser = self._setup_browser()
		self.context = self._create_context()
		self.page: Page = self.context.new_page()
		self._current_page_id = str(id(self.page))
		self._cached_state = self._update_state()

	def get_browser(self) -> PlaywrightBrowser:
		if self.browser is None:
			self.browser = self._setup_browser()
		return self.browser

	def _setup_browser(self) -> PlaywrightBrowser:
		"""Sets up and returns a Playwright Browser instance with anti-detection measures."""
		try:
			chrome_args = [
				'--disable-blink-features=AutomationControlled',
				'--no-sandbox',
				'--window-size=1280,1024',
				'--disable-extensions',
				'--disable-infobars',
				'--disable-background-timer-throttling',
				'--disable-popup-blocking',
				'--disable-backgrounding-occluded-windows',
				'--disable-renderer-backgrounding',
			]

			browser = self.playwright.chromium.launch(
				headless=self.headless,
				args=chrome_args,
			)

			return browser
		except Exception as e:
			logger.error(f'Failed to initialize Playwright browser: {str(e)}')
			raise

	def _create_context(self):
		"""Creates a new browser context with anti-detection measures."""
		context = self.browser.new_context(
			viewport={'width': 1280, 'height': 1024},
			user_agent=(
				'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
				'(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
			),
			java_script_enabled=True,
		)

		# Expose anti-detection scripts
		context.add_init_script(
			"""
			// Webdriver property
			Object.defineProperty(navigator, 'webdriver', {
				get: () => undefined
			});

			// Languages
			Object.defineProperty(navigator, 'languages', {
				get: () => ['en-US', 'en']
			});

			// Plugins
			Object.defineProperty(navigator, 'plugins', {
				get: () => [1, 2, 3, 4, 5]
			});

			// Chrome runtime
			window.chrome = { runtime: {} };

			// Permissions
			const originalQuery = window.navigator.permissions.query;
			window.navigator.permissions.query = (parameters) => (
				parameters.name === 'notifications' ?
					Promise.resolve({ state: Notification.permission }) :
					originalQuery(parameters)
			);
			"""
		)

		return context

	def _get_page(self) -> Page:
		if self.page is None:
			self.context = self._create_context()
			self.page = self.context.new_page()
		return self.page

	def wait_for_page_load(self):
		"""
		Ensures page is fully loaded before continuing.
		Waits for either document.readyState to be complete or minimum WAIT_TIME, whichever is longer.
		"""
		page = self._get_page()

		# Start timing
		start_time = time.time()

		# Wait for page load
		try:
			page.wait_for_load_state('load', timeout=5000)
		except Exception:
			pass

		# Calculate remaining time to meet minimum WAIT_TIME
		elapsed = time.time() - start_time
		remaining = max(self.MINIMUM_WAIT_TIME - elapsed, 0)

		logger.debug(
			f'--Page loaded in {elapsed:.2f} seconds, waiting for additional {remaining:.2f} seconds'
		)

		# Sleep remaining time if needed
		if remaining > 0:
			time.sleep(remaining)

	def _update_state(self, use_vision: bool = False) -> BrowserState:
		"""
		Update and return state.
		"""
		page = self._get_page()
		dom_service = DomService(page)
		content = dom_service.get_clickable_elements()

		screenshot_b64 = None
		if use_vision:
			screenshot_b64 = self.take_screenshot(selector_map=content.selector_map)

		self.current_state = BrowserState(
			items=content.items,
			selector_map=content.selector_map,
			url=page.url,
			title=page.title(),
			current_page_id=self._current_page_id,
			tabs=self.get_tabs_info(),
			screenshot=screenshot_b64,
		)

		return self.current_state

	def close(self, force: bool = False):
		if force:
			if self.browser:
				self.browser.close()
				self.playwright.stop()
		else:
			input('Press Enter to close Browser...')
			self.keep_open = False
			self.close()

	def __del__(self):
		"""
		Close the browser when instance is destroyed.
		"""
		if self.browser is not None:
			self.close()

	# region - Browser Actions

	def take_screenshot(self, selector_map: SelectorMap | None, full_page: bool = False) -> str:
		"""
		Returns a base64 encoded screenshot of the current page.
		"""
		page = self._get_page()

		if selector_map:
			self.highlight_selector_map_elements(selector_map)

		screenshot = page.screenshot(full_page=full_page)
		screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')

		if selector_map:
			self.remove_highlights()

		return screenshot_b64

	def highlight_selector_map_elements(self, selector_map: SelectorMap):
		page = self._get_page()
		# First remove any existing highlights/labels
		self.remove_highlights()

		script = """
		const highlights = {
		"""

		# Build the highlights object with all selectors and indices
		for index, selector in selector_map.items():
			# Adjusting the JavaScript code to accept variables
			script += f'"{index}": "{selector}",\n'

		script += """
		};
		
		for (const [index, selector] of Object.entries(highlights)) {
			const el = document.evaluate(selector, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
			if (!el) continue;  // Skip if element not found
			el.style.outline = "2px solid red";
			el.setAttribute('browser-user-highlight-id', 'playwright-highlight');
			
			const label = document.createElement("div");
			label.className = 'playwright-highlight-label';
			label.style.position = "fixed";
			label.style.background = "red";
			label.style.color = "white";
			label.style.padding = "2px 6px";
			label.style.borderRadius = "10px";
			label.style.fontSize = "12px";
			label.style.zIndex = "9999999";
			label.textContent = index;
			const rect = el.getBoundingClientRect();
			label.style.top = (rect.top - 20) + "px";
			label.style.left = rect.left + "px";
			document.body.appendChild(label);
		}
		"""

		page.evaluate(script)

	def remove_highlights(self):
		"""
		Removes all highlight outlines and labels created by highlight_selector_map_elements

		"""
		page = self._get_page()
		page.evaluate(
			"""
			// Remove all highlight outlines
			const highlightedElements = document.querySelectorAll('[browser-user-highlight-id="playwright-highlight"]');
			highlightedElements.forEach(el => {
				el.style.outline = '';
				el.removeAttribute('browser-user-highlight-id');
			});
			

			// Remove all labels
			const labels = document.querySelectorAll('.playwright-highlight-label');
			labels.forEach(label => label.remove());
			"""
		)

	# endregion

	# region - User Actions

	def _input_text_by_xpath(self, xpath: str, text: str):
		page = self._get_page()

		try:
			# Wait for element to be both present and visible
			element = page.wait_for_selector(f'xpath={xpath}', timeout=10000, state='visible')

			if element is None:
				raise Exception(f'Element with xpath: {xpath} not found')

			# Scroll element into view
			element.scroll_into_view_if_needed()

			# Clear the input field
			element.fill('')

			# Then fill with text
			element.type(text)

			self.wait_for_page_load()

		except Exception as e:
			raise Exception(
				f'Failed to input text into element with xpath: {xpath}. Error: {str(e)}'
			)

	def _click_element_by_xpath(self, xpath: str):
		"""
		Optimized method to click an element using xpath.
		"""
		page = self._get_page()

		try:
			# Wait for element to be clickable
			element = page.wait_for_selector(f'xpath={xpath}', timeout=10000, state='visible')

			if element is None:
				raise Exception(f'Element with xpath: {xpath} not found')

			# Scroll into view if needed
			element.scroll_into_view_if_needed()

			# Try to click directly
			try:
				element.click()
				self.wait_for_page_load()
				return
			except Exception:
				pass

			# If direct click fails, try JavaScript click
			try:
				page.evaluate('(el) => el.click()', element)
				self.wait_for_page_load()
				return
			except Exception as e:
				raise Exception(f'Failed to click element: {str(e)}')

		except Exception as e:
			raise Exception(f'Failed to click element with xpath: {xpath}. Error: {str(e)}')

	def handle_new_tab(self) -> None:
		"""Handle newly opened tab and switch to it"""
		context = self.page.context
		pages = context.pages
		new_page = pages[-1]  # Get most recently opened page

		# Switch to new page
		self.page = new_page
		self._current_page_id = str(id(new_page))

		# Wait for page load
		self.wait_for_page_load()

		# Create and cache tab info
		tab_info = TabInfo(page_id=self._current_page_id, url=new_page.url, title=new_page.title())
		self._tab_cache[self._current_page_id] = tab_info

	def get_tabs_info(self) -> list[TabInfo]:
		"""Get information about all tabs"""
		context = self.page.context
		pages = context.pages
		current_page = self.page
		self._current_page_id = str(id(current_page))

		tabs_info = []
		for page in pages:
			page_id = str(id(page))
			# Use cached info if available, otherwise get new info
			if page_id in self._tab_cache:
				tab_info = self._tab_cache[page_id]
				# Update URL and title in case they changed
				tab_info.url = page.url
				tab_info.title = page.title()
			else:
				tab_info = TabInfo(page_id=page_id, url=page.url, title=page.title())
				self._tab_cache[page_id] = tab_info

			tabs_info.append(tab_info)

		return tabs_info

	def switch_to_tab(self, page_id: str) -> None:
		"""Switch to a specific tab by its page_id"""
		context = self.page.context
		pages = context.pages

		for page in pages:
			if str(id(page)) == page_id:
				page.bring_to_front()
				self.page = page
				self._current_page_id = page_id
				self.wait_for_page_load()
				return

		raise ValueError(f'No tab found with page_id: {page_id}')

	def create_new_tab(self, url: str = None) -> None:
		"""Create a new tab and optionally navigate to a URL"""
		new_page = self.context.new_page()
		self.page = new_page
		self._current_page_id = str(id(new_page))

		if url:
			new_page.goto(url)
			self.wait_for_page_load()

	# endregion

	@time_execution_sync('--get_state')
	def get_state(self, use_vision: bool = False) -> BrowserState:
		"""
		Get the current state of the browser including page content and tab information.
		"""
		self._cached_state = self._update_state(use_vision=use_vision)
		return self._cached_state
