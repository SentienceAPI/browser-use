import pytest
from playwright.sync_api import Page


@pytest.fixture(scope='session')
def browser_type_launch_args():
	return {'headless': False}


def test_has_title(page: Page):
	page.goto('https://www.immobilienscout24.de')
	page.wait_for_timeout(5000)

	# Get all DOM content including all shadow roots recursively
	full_content = page.evaluate("""() => {
		function getAllContent(root) {
			let content = '';
			// Get all elements in the current root
			const elements = root.querySelectorAll('*');
			
			elements.forEach(element => {
				// Add the element's outer HTML
				content += element.outerHTML;
				// If element has shadow root, recursively get its content
				if (element.shadowRoot) {
					content += `\\n<!-- Shadow DOM for ${element.tagName} -->\\n`;
					content += getAllContent(element.shadowRoot);
					content += `\\n<!-- End Shadow DOM for ${element.tagName} -->\\n`;
				}
			});
			return content;
		}
		return getAllContent(document.body);
	}""")

	print(full_content)

	page.locator('#usercentrics-root').locator('[data-testid="uc-accept-all-button"]').click()

	input('Press Enter to continue...')
