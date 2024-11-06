from browser_use.dom.views import ProcessedDomContent


# Exceptions
class BrowserException(Exception):
	pass


# Pydantic
class BrowserState(ProcessedDomContent):
	url: str
	title: str
