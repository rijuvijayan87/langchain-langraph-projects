from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2

load_dotenv()


class MessageState(TypedDict):
    messages: Annotated[list, add_messages]


class ManagerOutput(BaseModel):
    context: str
    handoff_to: str
    handoff_reason: str


def chat_with_agent(question: str):
    """Ask question to the Customer service agent. This does the triaging of the question
    then categorize which sub-agents should handle the requests. It then routes the question
    to the appropriate agent
    """

    if not question:
        raise ValueError("error: question cannot be empty")

    llm = init_chat_model(model=MODEL, temperature=TEMPERATURE).with_structured_output(
        ManagerOutput
    )

    def manager(state: MessageState) -> dict:
        """manager agent responsible for the following tasks:
        1. triage the questions asked by the user
        2. identify and categorize the appropriate agent who has the maximum context on this question
        3. hand-offs the control with the prompt to that agent to handle the request
        """

        prompt = """
        You are a Customer Service manager of a travel management portal (like makemytrip). You are an expert in handling requests from the users and route the request to the appropriate
        agents from your team to handle that request
        You should also provide enough context as much as you possibily can, for the next agent to process the request. Keep it meticulous and true, do not manufacture information that you do not have. keep the context fact filled.
        You do not come up with the answers to the questions being asked. You rely purely on the team member who has the maximum context on the
        related question. Following is your team. Route to these appropriate agents
        - Billing - if the question is related. e.g. "i have a problem with billing, what do do?" or "i have been charged twice, how can i get a refund?"
        - Sales - if the question is related to sales and marketing. e.g. "i want to purchase a subscription of your product" or "what do i need to do to use your product"
        - Support - if question is related to engineering or issues with product. e.g. "i am getting an error while accessing the application" or "trying to book a ticket but i have received email of booking" or "i want to update the email on booking"
        - default - if the answer does not need a dedicated agent's skillset to solve the problem. i.e. if it is related to some static information or general information of the product. e.g. "what time does your call center open" or "do you have an office to visit or is it only online
        """

        message = [SystemMessage(content=prompt)] + state["messages"]

        response = llm.invoke(message)

        return {"messages": [AIMessage(content=str(response))]}

    graph = StateGraph(MessageState)
    graph.add_node("manager", manager)
    graph.add_edge(START, "manager")
    graph.add_edge("manager", END)

    app = graph.compile()

    response = app.invoke({"messages": [question]})

    if response:
        return response

    raise ValueError("error: response from app is not valid")


if __name__ == "__main__":
    print("=" * 60)
    question = input("\n ask your question: ")
    print(f"\n Operator: {question}")
    answer = chat_with_agent(question)

    print(f"AI: {answer["messages"][-1].content}")
