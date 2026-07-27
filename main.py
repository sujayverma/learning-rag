from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from importlib.metadata import version 
from langchain_openai import __version__ as openai_version
from langchain_openai import ChatOpenAI

lg_version = version("langgraph")
print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")
print(f"langchain-openai version: {openai_version}")

def main():
    # print("Hello from langc-course!")
    llm = ChatOpenAI(model_name="gpt-5.4-nano", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word.")
    print(f"OpenAI response: {response}")


if __name__ == "__main__":
    main()
