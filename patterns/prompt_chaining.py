import os
from typing import List, Callable, Any
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# For loading environment variables from a .env file
from dotenv import load_dotenv

class PromptChainStepType(Enum):
	LLM_PROMPT_STEP = 1
	# For any custom data processing tools or code calling we may need
	DATA_PROCESSING_STEP = 2

class PromptStep:
	def __init__(self, name: str, type: PromptChainStepType, processing_function: Callable[Any, Any] = None, prompt: str = None, output_type = None, is_deterministic: bool = None):
		self.name = name
		self.type = type
		# Optional, only applicable when the type is DATA_PROCESSING_STEP
		self.processing_function = processing_function
		# Optional, only applicable when the type is LLM_PROMPT_STEP
		self.prompt = prompt
		# Optional, specifies the type of output we expect from this specific step
		self.output_type = output_type
		# Optional, for the data processing steps, we give the option of whether to
		# mark the step as being determinstic or not
		self.is_deterministic = is_deterministic

class PromptChainer:
	def __init__(self, raw_prompt_steps: List[PromptStep]):
		print('Inside PromptChainer, raw_prompt_steps = ' + str(raw_prompt_steps))
		print()
		self.llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
		self.raw_prompt_steps = raw_prompt_steps
		'''self.chat_prompts = [
			ChatPromptTemplate.from_template(raw_prompt_step.prompt) for raw_prompt_step in raw_prompt_steps if not raw_prompt_step.is_deterministic
		]'''
		# This should be a dynamic LangChain representation
		self.promptchain = None
		self.initialize_prompt_chain()

	def configure_step(self, raw_prompt_step):
		if self.promptchain == None:
			if raw_prompt_step.is_deterministic:
				self.promptchain = RunnableLambda(raw_prompt_step.processing_function)
			else:
				self.promptchain = ChatPromptTemplate.from_template(raw_prompt_step.prompt) | self.llm | raw_prompt_step.output_type
		else:
			if raw_prompt_step.is_deterministic:
				self.promptchain = self.promptchain | RunnableLambda(raw_prompt_step.processing_function)
			else:
				self.promptchain = self.promptchain | ChatPromptTemplate.from_template(raw_prompt_step.prompt) | self.llm | raw_prompt_step.output_type

	def initialize_prompt_chain(self):
		for prompt_step in self.raw_prompt_steps:
			self.configure_step(prompt_step)

	def run(self, input_text: str) -> str:
		return self.promptchain.invoke({"text_input": input_text})

'''
# Sanity Checking the Implementation Below Here
json_transformer_promptchain = PromptChainer(raw_prompts=[
	"Extract the technical specifications from the following text:\n\n{text_input}",
	"Transform the following specifications into a JSON object with 'cpu','memory', and 'storage' as keys:\n\n{specifications}",
])

input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."
print(json_transformer_promptchain.run("The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."))
'''