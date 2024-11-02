import base64
import os

from bs4 import BeautifulSoup


def save_formatted_html(html_content, output_file_name):
	"""
	Format HTML content using BeautifulSoup and save to file

	Args:
	    html_content (str): Raw HTML content to format
	    output_file_name (str): Name of the file where formatted HTML will be saved
	    run_folder (str): Path to the run folder where the file will be saved
	"""
	# Format HTML with BeautifulSoup for nice indentation
	soup = BeautifulSoup(html_content, 'html.parser')
	formatted_html = soup.prettify()

	output_path = os.path.join(output_file_name)
	with open(output_path, 'w', encoding='utf-8') as f:
		f.write(formatted_html)


def save_conversation(input_messages, model_output, output_file_name):
	output_path = os.path.join(output_file_name)
	with open(output_path, 'w', encoding='utf-8') as f:
		for message in input_messages:
			f.write(f'{message["role"]}: {message["content"]}\n')
		f.write(f'\n\nModel output:\n{model_output}')


def save_markdown(markdown_content, output_file_name):
	"""Save markdown content to a file"""
	output_path = os.path.join(output_file_name)
	with open(output_path, 'w', encoding='utf-8') as f:
		f.write(markdown_content)


def encode_image(image_path):
	"""Encode image to base64 string"""
	with open(image_path, 'rb') as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')
