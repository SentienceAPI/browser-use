import unittest
from bs4 import BeautifulSoup
from src.state_manager.utils import cleanup_html
import os


class TestHTMLCleanup(unittest.TestCase):
	def setUp(self):
		# Create test directory if it doesn't exist
		self.test_dir = 'test_files'
		if not os.path.exists(self.test_dir):
			os.makedirs(self.test_dir)

	def test_cleanup_sample_html(self):
		# Sample HTML with various elements to test
		test_html = """
        <html>
            <head>
                <script>Some script</script>
                <style>Some style</style>
            </head>
            <body>
                <div class="hidden">Hidden content</div>
                <div style="display: none">Display none content</div>
                <div aria-hidden="true">Aria hidden content</div>
                <button class="interactive" onclick="test()">Click me</button>
                <a href="/test" class="link">Link text</a>
                <div class="content">Visible content</div>
                <input type="text" placeholder="Enter text">
                <select>
                    <option>Option 1</option>
                </select>
                <div role="button">Custom button</div>
                <div class="cookie-banner">Accept cookies</div>
            </body>
        </html>
        """

		result = cleanup_html(test_html)
		cleaned_html = result['html']

		# Parse the cleaned HTML for testing
		soup = BeautifulSoup(cleaned_html, 'html.parser')

		# Test that script and style are removed
		self.assertEqual(len(soup.find_all('script')), 0)
		self.assertEqual(len(soup.find_all('style')), 0)

		# Test that hidden elements are removed
		self.assertEqual(len(soup.find_all(attrs={'class': 'hidden'})), 0)
		self.assertEqual(len(soup.find_all(attrs={'style': 'display: none'})), 0)
		self.assertEqual(len(soup.find_all(attrs={'aria-hidden': 'true'})), 0)

		# Test that interactive elements have 'c' attribute
		interactive_elements = soup.find_all(attrs={'c': True})
		self.assertGreater(len(interactive_elements), 0)

		# Test specific interactive elements
		self.assertIsNotNone(soup.find('button'))
		self.assertIsNotNone(soup.find('a'))
		self.assertIsNotNone(soup.find('input'))
		self.assertIsNotNone(soup.find('select'))

		# Print cleaned HTML for inspection
		print('\nCleaned HTML:')
		print(cleaned_html)

	def test_cleanup_real_page(self):
		# Load a real webpage HTML file
		test_file = os.path.join(self.test_dir, 'kayak_page.html')

		# First run: download and save the page if it doesn't exist
		if not os.path.exists(test_file):
			import requests

			response = requests.get('https://www.kayak.ch')
			with open(test_file, 'w', encoding='utf-8') as f:
				f.write(response.text)

		# Read the test file
		with open(test_file, 'r', encoding='utf-8') as f:
			html_content = f.read()

		# Clean up the HTML
		result = cleanup_html(html_content)
		cleaned_html = result['html']

		# Parse the cleaned HTML
		soup = BeautifulSoup(cleaned_html, 'html.parser')

		# Test basic cleanup
		self.assertEqual(len(soup.find_all('script')), 0)
		self.assertEqual(len(soup.find_all('style')), 0)

		# Test interactive elements
		interactive_elements = soup.find_all(attrs={'c': True})
		self.assertGreater(len(interactive_elements), 0)

		# Print statistics
		print('\nReal page cleanup statistics:')
		print(f'Total interactive elements found: {len(interactive_elements)}')
		print(f"Buttons: {len(soup.find_all('button'))}")
		print(f"Links: {len(soup.find_all('a'))}")
		print(f"Inputs: {len(soup.find_all('input'))}")
		print(f"Selects: {len(soup.find_all('select'))}")

		# Save cleaned HTML for manual inspection
		cleaned_file = os.path.join(self.test_dir, 'kayak_page_cleaned.html')
		with open(cleaned_file, 'w', encoding='utf-8') as f:
			f.write(cleaned_html)
		print(f'\nCleaned HTML saved to: {cleaned_file}')

	def test_main_content_extraction(self):
		test_html = """
        <html>
            <body>
                <header>Header content</header>
                <main>
                    <h1>Main Title</h1>
                    <p>Main content paragraph</p>
                    <div class="important">Important content</div>
                </main>
                <footer>Footer content</footer>
            </body>
        </html>
        """

		result = cleanup_html(test_html)
		main_content = result['main_content']

		self.assertIn('Main Title', main_content)
		self.assertIn('Main content paragraph', main_content)
		self.assertIn('Important content', main_content)
		self.assertNotIn('Header content', main_content)
		self.assertNotIn('Footer content', main_content)


if __name__ == '__main__':
	unittest.main()
