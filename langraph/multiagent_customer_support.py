from email.policy import default
from typing import Annotated, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

load_dotenv()

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2


class ManagerOutput(BaseModel):
    handoff_to_agent: str
    handoff_reason: str
    context: str


class MessageState(TypedDict):
    messages: Annotated[list, add_messages]
    handoff_to_agent: NotRequired[str]
    handoff_reason: NotRequired[str]
    context: NotRequired[str]


def customer_support_agent(question: str) -> dict:
    """Ask question to the Customer service agent. This does the triaging of the question
    then categorize which sub-agents should handle the requests. It then routes the question
    to the appropriate agent
    """

    llm_with_structured_output = init_chat_model(
        model=MODEL, temperature=TEMPERATURE
    ).with_structured_output(ManagerOutput)

    if not question:
        raise ValueError(f"question should be non empty value")

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

        decision = llm_with_structured_output.invoke(
            [SystemMessage(content=prompt)] + state["messages"]
        )

        if not isinstance(decision, ManagerOutput):
            raise ValueError(
                "error: decision llm response does not conform to ManagerOutput schema"
            )

        return {
            "messages": AIMessage(
                content=str(
                    f"[Triage] transfer control to {decision.handoff_to_agent}: {decision.handoff_reason}"
                )
            ),
            "context": decision.context,
            "handoff_to_agent": decision.handoff_to_agent,
            "handoff_reason": decision.handoff_reason,
        }

    def billing(state: MessageState) -> dict:
        """Billing agent who gets routed to when the handoff_to_agent=billing"""
        print(f"billing agent activated")
        return {}

    def sales(state: MessageState) -> dict:
        """sales agent who gets routed to when the handoff_to_agent=sales"""
        print(f"sales agent activated")
        return {}

    def support(state: MessageState) -> dict:
        """support agent who gets routed to when the handoff_to_agent=support"""
        print(f"support agent activated")
        return {}

    def agent_routing_logic(
        state: MessageState,
    ) -> str | None:
        """Logic to route the control to appropriate agents"""
        agent = state.get("handoff_to_agent", "not available")

        if not agent:
            return "default"

        match agent.lower():
            case "billing":
                return "billing"
            case "sales":
                return "sales"
            case "support":
                return "support"
            case _:
                return "default"

    graph = StateGraph(MessageState)

    graph.add_node("manager", manager)
    graph.add_node("billing", billing)
    graph.add_node("sales", sales)
    graph.add_node("support", support)

    graph.add_edge(START, "manager")
    graph.add_conditional_edges(
        source="manager",
        path=agent_routing_logic,
        path_map={
            "billing": "billing",
            "sales": "sales",
            "support": "support",
            "default": END,
        },
    )

    graph.add_edge("billing", END)
    graph.add_edge("sales", END)
    graph.add_edge("support", END)

    app = graph.compile()

    image_bytes = app.get_graph().draw_mermaid_png()
    with open("customer_support.png", "wb") as f:
        f.write(image_bytes)

    return app.invoke({"messages": [question]})


if __name__ == "__main__":
    question = input("\n ask your question: ")

    answer = customer_support_agent(question)
    print(f"AI: {answer}")
