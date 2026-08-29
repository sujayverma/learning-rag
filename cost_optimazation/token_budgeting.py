from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class TokenBudgeting:
    """Track and limit Token usage for LLM Calls."""

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count based on word count."""
        return int(len(text.split()) * 1.5)  # Approximation: 1.5 tokens per word

    def check_tokens(self, input_text: str) -> tuple[bool, int]:
        """Check if the request is within the token budget."""
        input_tokens = self.estimate_tokens(input_text)
        return input_tokens <= self.max_per_request, input_tokens