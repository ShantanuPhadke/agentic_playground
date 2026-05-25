from patterns.prompt_chaining import PromptChainer, PromptStep, PromptChainStepType

import re

class MyWikipediaProcessor:
	def __init__(self):
		self.description = "A simple Wikipedia based information processing workflow that takes in a specific Wikipedia URL and generates a structured report for it."
		self.construct_prompt_chain()

	def construct_prompt_chain(self):
		raw_prompts = [
			PromptStep(
				"Wikipedia Page Content Extraction",
				PromptChainStepType.DATA_PROCESSING_STEP,
				#prompt="Extract the text content from this Wikipedia URl using a curl command: {text_input}",
				output_type=(lambda msg: {"page_content": msg.content}),
				is_deterministic=True,
				processing_function=self.curl_processing_function
			),
			PromptStep(
				"Wikipedia Page Summarization",
				PromptChainStepType.LLM_PROMPT_STEP,
				prompt="Summarize the raw page content into cleaned text: {page_content}",
				output_type=(lambda msg: {"page_summary": msg.content})
			),
			PromptStep(
				"Extract Entities",
				PromptChainStepType.LLM_PROMPT_STEP,
				prompt="Extract entities from the summary you produced or the original text content. Page summary: {page_summary}",
				output_type=(lambda msg: {"page_entities": msg.content})
			),
			PromptStep(
				"Related Wikipedia URLs",
				PromptChainStepType.LLM_PROMPT_STEP,
				prompt="Use the entity names you extracted to come up with a list of other relevant Wikipedia page URLs to search: {page_entities}",
				output_type=(lambda msg: {"related_urls": msg.content})
			),
			PromptStep(
				"Get List of URLs",
				PromptChainStepType.DATA_PROCESSING_STEP,
				is_deterministic=True,
				processing_function=self.extract_urls,
				output_type=(lambda msg: {"urls": msg.content})
			),
			PromptStep(
				"Get all URLs text",
				PromptChainStepType.DATA_PROCESSING_STEP,
				is_deterministic=True,
				processing_function=self.process_multiple_pages_contents,
				output_type=(lambda msg: {"overall_content": msg.content})
			),
			PromptStep(
				"Overall Wikipedia Related URLs text",
				PromptChainStepType.LLM_PROMPT_STEP,
				prompt="Generate a structured report combining your summary, the data on entities you extracted, and other relevant Wikipedia data: {overall_content}",
				output_type=(lambda msg: {"related_urls": msg.content})
			)
		]

		self.promptchainer = PromptChainer(raw_prompts)

	def curl_processing_function(self, info):
		import requests
		HEADERS = {
		    "User-Agent": "MyResearchBot/1.0 (contact: myemail@example.com)"
		}
		r = requests.get(info['text_input'], headers=HEADERS)
		return { "page_content": r.text }

	def extract_urls(self, raw_urls_text):
		pattern = r'(?:https?://|www\.)[^\s]+'
		urls = re.findall(pattern, raw_urls_text['related_urls'])
		return urls

	def process_multiple_pages_contents(self, page_urls):
		overall_content = ''
		for page_url in page_urls:
			import requests
			HEADERS = {
			    "User-Agent": "MyResearchBot/1.0 (contact: myemail@example.com)"
			}
			r = requests.get(page_url, headers=HEADERS)
			overall_content += ", " + r.text
		return overall_content

	def run(self, url: str) -> str:
		return self.promptchainer.run(url)


my_wikipedia_processor = MyWikipediaProcessor()
print(my_wikipedia_processor.run('https://en.wikipedia.org/wiki/Cricket'))

#my_wikipedia_processor.curl_processing_function('https://en.wikipedia.org/wiki/Cricket')
