# import os
# from langtrace_python_sdk import langtrace

# langtrace.init(api_key='')

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

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

	def load_knowledge_base(self):
		# Implement knowledge base loading from provided docs
		pass

	@agent
	def overseer(self) -> Agent:
		return Agent(
			config=self.agents_config['overseer'],
			verbose=True,
			allow_delegation=True,
			tools_enabled=False
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
		# Task for the overseer to analyze and route the query
		analysis_task = Task(
			description=f"""
			Analyze this query and respond appropriately: {user_query}

			You are capable of direct conversation and should:
			- Respond directly to greetings (like "hi", "hello", etc.)
			- Answer general questions about AI and technology
			- Engage in casual conversation
			- Provide helpful responses to basic queries
			- Handle follow-up questions

			Format your responses using markdown:
			- Use **bold** for emphasis
			- Use headings where appropriate (# for main headings)
			- Use bullet points for lists
			- Use code blocks with ```language for code
			- Use > for quotes or important notes

			Only delegate to the prompt expert (using 'DELEGATE: <reason>') if the query specifically requires
			expertise in:
			- Prompt engineering techniques
			- LLM chain design
			- Prompt optimization strategies
			- Complex LLM interaction patterns

			For everything else, provide a friendly and helpful response yourself in markdown format.
			""",
			expected_output="Either a markdown-formatted response or 'DELEGATE: <reason>'",
			agent=self.overseer()
		)

		# Task for the prompt expert to handle relevant queries
		expert_task = Task(
			description=f"""
			Only respond if the overseer has delegated this query to you.
			If the overseer did not delegate (no 'DELEGATE:' prefix), remain silent.
			
			Format your response using markdown:
			- Use **bold** for emphasis
			- Use headings where appropriate (# for main headings)
			- Use bullet points for lists
			- Use code blocks with ```language for code
			- Use > for quotes or important notes
			
			Query: {user_query}
			""",
			expected_output="Markdown-formatted expert advice on prompt engineering (only if delegated)",
			agent=self.prompt_expert()
		)

		return Crew(
			agents=[self.prompt_expert()],
			tasks=[analysis_task, expert_task],
			manager_agent=self.overseer(),
			process=Process.hierarchical,
			verbose=True
		)

	def process_query(self, user_query: str) -> str:
		"""Process a query and return the formatted response"""
		print(f"\n🤖 Processing query: {user_query}")
		
		crew = self.create_crew(user_query)
		result = crew.kickoff()
		
		print("\n✨ Processing complete")
		
		# Get the overseer's analysis
		response = str(result.output) if hasattr(result, 'output') else str(result)

		# If the overseer delegated, use the expert's response
		if response.startswith('DELEGATE:'):
			if hasattr(result, 'task_outputs') and len(result.task_outputs) > 1:
				response = str(result.task_outputs[1])
		
		return self.format_response(response)
