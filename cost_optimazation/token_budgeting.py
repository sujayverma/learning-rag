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

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record the total token usage for input and output for a request."""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_usage(self) -> dict:
        """Return the current token usage statistics."""
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "average_tokens_per_request": (
                self.usage["total_input"] + self.usage["total_output"] /
                max(1, self.usage["requests"])  # Avoid division by zero
            ),
        }


class BudgetLLM:
    """LLM With Token Budgeting."""

    def __init__(self, max_tokens: int = 4000):
        self.budget = TokenBudgeting(max_tokens_per_request=max_tokens)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=max_tokens)


    def invoke(self, query: str) -> str:
        """Invoke the LLM with Token Budgeting."""
        within_budget, input_tokens = self.budget.check_tokens(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds the maximum token limit of {self.budget.max_per_request}. "
            )

        response = self.llm.invoke(query)
        result = response.content

        output_tokens = self.budget.estimate_tokens(result)
        self.budget.record_usage(input_tokens, output_tokens)

        return result

    def get_usage(self) -> dict:
        """Return the current token usage statistics."""
        return self.budget.get_usage()


def demo_token_budgeting():
    """Demonstrate the Token Buudgeting with LLM."""
    budget_llm = BudgetLLM(max_tokens=100)

    queries = [
        "What is the Capital of France?",
        "Explain a "+ "Very "* 100 + " long text to test the token budgeting system."
    ]

    print("\nToken Budgeting Demo:\n")

    for query in queries:
        try:
            result = budget_llm.invoke(query)
            print(f"Query: {query}\nResult: {result}\n")
        except ValueError as e:
            print(f"Query: {query}\nError: {e}\n")

    print("Token Usage Statistics:")
    usage_stats = budget_llm.get_usage()
    for key, value in usage_stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    demo_token_budgeting()