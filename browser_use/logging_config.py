import logging
import os
import sys


def setup_logging():
	debug_logging = os.getenv('BROWSER_USE_DEBUG_LOGGING', 'false').lower() == 'true'

	# Check if handlers are already set up
	if logging.getLogger().hasHandlers():
		return

	# Clear existing handlers
	root = logging.getLogger()
	root.handlers = []

	class BrowserUseFormatter(logging.Formatter):
		def format(self, record):
			if record.name.startswith('browser_use.'):
				record.name = record.name.split('.')[-2]
			return super().format(record)

	# Setup single handler for all loggers
	console = logging.StreamHandler(sys.stdout)
	console.setFormatter(BrowserUseFormatter('%(levelname)-8s [%(name)s] %(message)s'))

	# Configure root logger only
	root.addHandler(console)

	if debug_logging:
		root.setLevel(logging.DEBUG)
	else:
		root.setLevel(logging.INFO)

	# Configure browser_use logger to prevent propagation
	browser_use_logger = logging.getLogger('browser_use')
	browser_use_logger.propagate = False
	browser_use_logger.addHandler(console)

	# Silence third-party loggers
	for logger in [
		'WDM',
		'httpx',
		'selenium',
		'urllib3',
		'asyncio',
		'langchain',
		'openai',
		'httpcore',
		'charset_normalizer',
	]:
		third_party = logging.getLogger(logger)
		third_party.setLevel(logging.ERROR)
		third_party.propagate = False
