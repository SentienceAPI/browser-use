import time

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def test_selenium():
	try:
		print('1. Setting up Chrome options...')
		chrome_options = Options()
		chrome_options.add_argument('--no-sandbox')
		# Uncomment to test headless mode
		# chrome_options.add_argument('--headless=new')

		print('2. Installing/finding ChromeDriver...')
		service = Service(ChromeDriverManager().install())

		print('3. Creating Chrome WebDriver...')
		driver = webdriver.Chrome(service=service, options=chrome_options)

		print('4. Navigating to Google...')
		driver.get('https://www.google.com')

		print('5. Getting page title...')
		title = driver.title
		print(f'Page title: {title}')

		time.sleep(2)  # Wait to see the page if not in headless mode

		print('6. Closing browser...')
		driver.quit()

		print('✅ Test completed successfully!')
		return True

	except Exception as e:
		print(f'❌ Test failed with error: {str(e)}')
		print(f'Error type: {type(e).__name__}')
		return False


# run with: pytest browser_use/browser/tests/test_selenium.py

#

if __name__ == '__main__':
	pytest.main([__file__, '-v'])
