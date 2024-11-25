import json
import os
import time

from browser_use.browser.service import Browser


async def test_process_dom():
	browser = Browser(headless=False)

	page = await browser.get_current_page()

	# dom_service = DomService(page)

	# await page.goto('https://kayak.com/flights')
	# browser.go_to_url('https://google.com/flights')
	await page.goto('https://immobilienscout24.de')

	time.sleep(1.5)

	with open('browser_use/dom/process_dom.js', 'r') as f:
		js_code = f.read()

	start = time.time()
	dom_tree = await page.evaluate(js_code)
	end = time.time()

	# print(dom_tree)
	print(f'Time: {end - start:.2f}s')

	os.makedirs('./tmp', exist_ok=True)
	with open('./tmp/dom.json', 'w') as f:
		json.dump(dom_tree, f, indent=1)

	input('Press Enter to continue...')
