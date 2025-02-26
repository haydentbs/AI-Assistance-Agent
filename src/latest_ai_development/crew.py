# import os
# from langtrace_python_sdk import langtrace

# langtrace.init(api_key='')

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class AiAssistanceCrew():
	"""Simple AI Assistant with a single agent"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self):
		self.knowledge_base = self.load_knowledge_base()

	def load_knowledge_base(self):
		# Implement knowledge base loading from provided docs
		pass

	@agent
	def overseer(self) -> Agent:
		return Agent(
			config=self.agents_config['overseer'],
			verbose=True  # Show agent's thought process in terminal
		)

	@task
	def handle_query(self) -> Task:
		return Task(
			description="Provide a comprehensive response to the user query: {user_query}",
			expected_output="A clear, well-structured response that addresses the user's question",
			agent=self.overseer()
		)

	def format_response(self, response: str) -> str:
		"""Format the response in a clean chat-like format"""
		if not response:
			return "I apologize, but I couldn't generate a response."
			
		# Remove any system messages or prefixes
		if "Final Answer:" in response:
			response = response.split("Final Answer:")[-1].strip()
		
		# Clean up any markdown or special characters
		response = response.replace("#", "").strip()
		
		return response

	def create_crew(self, user_query: str) -> Crew:
		"""Creates a crew instance with the given query"""
		task = Task(
			description=f"Answer this question concisely and helpfully: {user_query}",
			expected_output="A clear and direct response",
			agent=self.overseer()
		)

		return Crew(
			agents=[self.overseer()],
			tasks=[task],
			process=Process.sequential,
			verbose=True  # Show crew execution details in terminal
		)

	def process_query(self, user_query: str) -> str:
		"""Process a query and return the formatted response"""
		print(f"\n🤖 Processing query: {user_query}")  # Terminal feedback
		
		crew = self.create_crew(user_query)
		result = crew.kickoff()
		
		print("\n✨ Processing complete")  # Terminal feedback
		
		# Handle the CrewOutput object directly
		if hasattr(result, 'output'):
			return self.format_response(str(result.output))
		elif isinstance(result, dict):
			return self.format_response(result.get('output', ''))
		else:
			return self.format_response(str(result))
