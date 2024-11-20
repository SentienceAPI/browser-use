import os
import sys

from browser_use.logging_config import setup_logging

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

setup_logging()


# @pytest.fixture(autouse=True)
# async def event_loop():
# 	"""Create an instance of the default event loop for each test case."""
# 	loop = asyncio.get_event_loop_policy().new_event_loop()
# 	yield loop
# 	# Cleanup pending tasks
# 	pending = asyncio.all_tasks(loop)
# 	for task in pending:
# 		task.cancel()
# 	await asyncio.gather(*pending, return_exceptions=True)
# 	await loop.shutdown_asyncgens()
# 	loop.close()
