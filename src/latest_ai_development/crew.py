# import os
# from langtrace_python_sdk import langtrace

# langtrace.init(api_key='')

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, after_kickoff, before_kickoff

@CrewBase
class AiAssistanceCrew():
	"""AI Assistant with overseer and specialist agents"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self):
		self.knowledge_base = self.load_knowledge_base()
		self.chat_history = []  # Add chat history list

	def load_knowledge_base(self):
		# Implement knowledge base loading from provided docs
		pass



	@agent
	def overseer(self) -> Agent:
		return Agent(
			config=self.agents_config['overseer'],
			verbose=True,
			allow_delegation=True,
		)
	
	@agent
	def prompt_expert(self) -> Agent:
		return Agent(
			config=self.agents_config['prompt_expert'],
			verbose=True
		)

	def format_response(self, response: str) -> str:
		"""Format the response in a clean markdown format"""
		if not response:
			return "I apologize, but I couldn't generate a response."
			
		# Remove any system messages or prefixes
		if "Final Answer:" in response:
			response = response.split("Final Answer:")[-1].strip()
		
		# Don't strip markdown characters, only clean up extra whitespace
		response = response.strip()
		
		return response

	def create_crew(self, user_query: str) -> Crew:
		"""Creates a crew instance with the given query"""
		# Format chat history for context
		chat_context = "\n".join([
			f"{'User' if i%2==0 else 'Assistant'}: {msg}"
			for i, msg in enumerate(self.chat_history[-6:])  # Last 3 exchanges
		])

		# Task for direct response
		response_task = Task(
			description=f"""
			Previous conversation:
			{chat_context if self.chat_history else "No previous conversation"}

			Current query: {user_query}

			Provide a direct response to this query. You are an expert in general AI topics and conversation.
			Format your response using markdown:
			- Use **bold** for emphasis
			- Use headings where appropriate (# for main headings)
			- Use bullet points for lists
			- Use code blocks with ```language for code
			- Use > for quotes or important notes
			
			If you need prompt engineering expertise, respond with 'DELEGATE: <reason>'.
			Otherwise, provide a helpful response directly.
			""",
			expected_output="Markdown-formatted response",
			agent=self.prompt_expert()
		)

		tasks = [response_task]
		agents = [self.prompt_expert()]

		return Crew(
			agents=agents,
			tasks=tasks,
			manager_agent=self.overseer(),
			process=Process.hierarchical,
			verbose=True,
			memory=True
		)

	def process_query(self, user_query: str) -> str:
		"""Process a query and return the formatted response"""
		print(f"\n🤖 Processing query: {user_query}")
		
		# Add user query to history
		self.chat_history.append(user_query)
		
		crew = self.create_crew(user_query)
		result = crew.kickoff()
		
		print("\n✨ Processing complete")
		
		# Get the response
		response = str(result.output) if hasattr(result, 'output') else str(result)
		
		formatted_response = self.format_response(response)
		
		# Add assistant response to history
		self.chat_history.append(formatted_response)
		
		return formatted_response
