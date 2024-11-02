"""
Driver Service
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class DriverService:
	def __init__(self, headless: bool = False):
		self.headless = headless

	def get_driver(self) -> webdriver.Chrome:
		"""
		Sets up and returns a Selenium WebDriver instance with anti-detection measures.

		Returns:
		    webdriver.Chrome: Configured Chrome WebDriver instance
		"""
		chrome_options = Options()
		if self.headless:
			chrome_options.add_argument('--headless')

		# Anti-detection measures
		chrome_options.add_argument('--disable-blink-features=AutomationControlled')
		chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
		chrome_options.add_experimental_option('useAutomationExtension', False)

		# Additional stealth settings
		chrome_options.add_argument('--start-maximized')
		chrome_options.add_argument('--disable-extensions')
		chrome_options.add_argument('--no-sandbox')
		chrome_options.add_argument('--disable-infobars')

		# Initialize the Chrome driver
		driver = webdriver.Chrome(
			service=Service(ChromeDriverManager().install()), options=chrome_options
		)

		# Execute stealth scripts
		driver.execute_cdp_cmd(
			'Page.addScriptToEvaluateOnNewDocument',
			{
				'source': """
				Object.defineProperty(navigator, 'webdriver', {
					get: () => undefined
				});
				
				Object.defineProperty(navigator, 'languages', {
					get: () => ['en-US', 'en']
				});
				
				Object.defineProperty(navigator, 'plugins', {
					get: () => [1, 2, 3, 4, 5]
				});
				
				window.chrome = {
					runtime: {}
				};
				
				Object.defineProperty(navigator, 'permissions', {
					get: () => ({
						query: Promise.resolve.bind(Promise)
					})
				});
			"""
			},
		)

		return driver
