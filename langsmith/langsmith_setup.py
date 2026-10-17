"""
LangSmith Setup observability and tracing configuration.
Production Monitoring and observability for LangChain applications using LangSmith.
"""
import os
from langsmith import traceable, get_current_run_tree
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.run_trees import RunTree

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"


# @traceable(name="demo_basic_tracing") 
# When using a method without any input or output, them use with_config in chain to add input and output tracing.
def demo_basic_tracing():
    """
    Demonstrates basic tracing with langsmith.
    """
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_template(
        "Explain the significance of {topic} in modern technology."
    )

    chain = (prompt | llm | StrOutputParser()).with_config(
        run_name="demo_basic_tracing"
    )

    print("Running the chain with tracing enabled...")

    result = chain.invoke({"topic": "Artificial Intelligence"})

    print("Result:", result)
    print("\nTraceable run completed. Check LangSmith dashboard for details.")
    # run = get_current_run_tree()
    # run.add_metadata({"topic": "Artificial Intelligence", "user": "Sujay"})
    # run.add_tags(["observability", "tracing", "production"])
    # # run.add_notes("This run demonstrates basic tracing with LangSmith.")
    # run.add_outputs({"result": result})


# @traceable(name="demo_basic_tracing") 

# def demo_basic_tracing():
#     """
#     Demonstrates basic tracing with langsmith.
#     """
#     llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

#     prompt = ChatPromptTemplate.from_template(
#         "Explain the significance of {topic} in modern technology."
#     )

#     chain = prompt | llm | StrOutputParser()

#     print("Running the chain with tracing enabled...")

#     result = chain.invoke({"topic": "Artificial Intelligence"})

#     print("Result:", result)
#     print("\nTraceable run completed. Check LangSmith dashboard for details.")
#     run = get_current_run_tree()
#     run.add_metadata({"topic": "Artificial Intelligence", "user": "Sujay"})
#     run.add_tags(["observability", "tracing", "production"])
#     # run.add_notes("This run demonstrates basic tracing with LangSmith.")
#     run.add_outputs({"result": result})

@traceable(name="demo_named_runs", tags=["observability", "tracing", "production"])
def demo_name_runs():
    """
    Demonstrates naming runs in LangSmith for better observability.
    """

    llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_template(
        "Describe the impact of {topic} on society."
    )

    chain  = prompt | llm | StrOutputParser()

    print("Running the chain with named runs...")

    result = chain.invoke({"topic": "Human Evolution"})

    print("Result:", result)
    print("\nNamed run completed. Check LangSmith dashboard for details.")

def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata to traces for filtering and observability."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    result = llm.invoke(f"Hello the user {user_id}")
    return result.content


if __name__ == "__main__":
    demo_basic_tracing()
    # demo_name_runs()
    demo_trace_with_metadata(user_id="Sujay", request_type="greetings")
    