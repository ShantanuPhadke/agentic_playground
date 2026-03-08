import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# For loading environment variables from a .env file
from dotenv import load_dotenv

class PromptChainer:
	def __init__(self, raw_prompts: List[str]):
		self.llm = ChatOpenAI(temperature=0)
		self.chat_prompts = [
			ChatPromptTemplate.from_template(raw_prompt) for raw_prompt in raw_prompts
		]
		# This should be a dynamic LangChain representation
		self.promptchain = None
		self.initialize_prompt_chain()

	def initialize_prompt_chain(self):
		for chat_prompt in self.chat_prompts:
			if self.promptchain == None:
				self.promptchain = chat_prompt | self.llm | StrOutputParser()
			else:
				self.promptchain = self.promptchain | chat_prompt | self.llm | StrOutputParser()


	def run(self, input_text: str) -> str:
		return self.promptchain.invoke({"text_input": input_text})

# Sanity Checking the Implementation Below Here
json_transformer_promptchain = PromptChainer(raw_prompts=[
	"Extract the technical specifications from the following text:\n\n{text_input}",
	"Transform the following specifications into a JSON object with 'cpu','memory', and 'storage' as keys:\n\n{specifications}",
])

input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."
print(json_transformer_promptchain.run("The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."))
