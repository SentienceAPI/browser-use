from bs4 import BeautifulSoup
import os


def save_formatted_html(html_content, output_file_name):
	"""
	Format HTML content using BeautifulSoup and save to file

	Args:
	    html_content (str): Raw HTML content to format
	    output_file_name (str): Name of the file where formatted HTML will be saved
	"""
	# Format HTML with BeautifulSoup for nice indentation
	soup = BeautifulSoup(html_content, 'html.parser')
	formatted_html = soup.prettify()

	# create temp folder if it doesn't exist
	if not os.path.exists('temp'):
		os.makedirs('temp')

	# Save formatted HTML to file
	with open('temp/' + output_file_name, 'w', encoding='utf-8') as f:
		f.write(formatted_html)


def save_markdown(markdown_content, output_file_name):
	"""Save markdown content to a file"""
	if not os.path.exists('temp'):
		os.makedirs('temp')

	with open('temp/' + output_file_name, 'w', encoding='utf-8') as f:
		f.write(markdown_content)
