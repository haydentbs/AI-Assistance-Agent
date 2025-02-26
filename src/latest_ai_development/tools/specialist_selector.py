from crewai.tools import BaseTool
from typing import Type, Dict
from pydantic import BaseModel, Field

class SpecialistSelectorInput(BaseModel):
    """Input schema for SpecialistSelector tool."""
    query: str = Field(..., description="The user query to analyze for specialist selection.")

class SpecialistSelector(BaseTool):
    name: str = "select_specialist"
    description: str = (
        "Analyzes a user query and determines the most appropriate specialist to handle it. "
        "Returns one of: 'prompt_engineer', 'compliance_expert', 'model_selector', or 'faq_specialist'."
    )
    args_schema: Type[BaseModel] = SpecialistSelectorInput
    specialist_options: Dict[str, str] = {}

    def __init__(self, specialist_options: Dict[str, str]):
        super().__init__()
        self.specialist_options = specialist_options or {}

    def _run(self, query: str) -> str:
        """
        Determines the most appropriate specialist based on the query content.
        
        Args:
            query: The user's question or request
            
        Returns:
            str: The name of the selected specialist
        """
        # Handle empty or None query
        if not query:
            return 'faq_specialist'
        
        # Check for direct mentions of specialists
        for specialist in self.specialist_options:
            if specialist in query.lower():
                return specialist
                
        # Analyze query content for relevant keywords
        if any(keyword in query.lower() for keyword in ['prompt', 'instruction', 'asking']):
            return 'prompt_engineer'
        elif any(keyword in query.lower() for keyword in ['compliance', 'regulation', 'ethics', 'legal']):
            return 'compliance_expert'
        elif any(keyword in query.lower() for keyword in ['model', 'choose', 'select', 'which ai']):
            return 'model_selector'
        
        # Default to FAQ specialist for general queries
        return 'faq_specialist' 