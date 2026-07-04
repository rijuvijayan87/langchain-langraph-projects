import operator
import stat
import typing
from email import message
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph, add_messages

load_dotenv(override=True)


class SimpleGraph(TypedDict):
    input: str
    output: str
    index: int


def simple_graph_demo():

    def process(state: SimpleGraph) -> dict[str, typing.Any]:
        return {
            "output": state["input"].upper() + "_data_formatting",
            "index": state["index"] + 1,
        }

    graph = StateGraph(SimpleGraph)

    graph.add_node("process", process)

    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    app = graph.compile()

    print("\n print mermaid graph")

    print(app.get_graph().draw_mermaid())

    png_bytes = app.get_graph().draw_mermaid_png()
    with open("image.png", "wb") as f:
        f.write(png_bytes)

    print("mermaid graph drawn\n")
    result = app.invoke({"input": "test data", "output": "", "index": 0})

    print(f"simple graph result {result}")

    print(
        f"input value : {result['input']}, ouptut: {result['output']}, index: {result['index']}"
    )


class MessageSchema(TypedDict):
    messages: Annotated[list[str], operator.add]
    count: Annotated[int, operator.add]


def annotated_graph_demo():
    """Annotated graph demo"""

    def process_node_1(state: MessageSchema) -> dict:
        """Process message that comes in"""
        return {"messages": ["processed by node 1"], "count": 1}

    def process_node_2(state: MessageSchema) -> dict:
        """Process message that comes in"""
        return {"messages": ["processed by node 2"], "count": 9}

    graph = StateGraph(MessageSchema)

    graph.add_node("node_1", process_node_1)
    graph.add_node("node_2", process_node_2)
    graph.add_edge(START, "node_1")
    graph.add_edge("node_1", "node_2")
    graph.add_edge("node_2", END)

    app = graph.compile()

    response = app.invoke({"messages": ["initial message"], "count": 1})

    print(response)


class MessageState(TypedDict):
    messages: Annotated[list, add_messages]


def annotated_ai_example():
    """Annotated AI Example"""
    llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)

    def process(state: MessageState) -> dict:
        """Send requests to llm and get response"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessageState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    app = graph.compile()

    response = MessageState(
        **app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="how do you explain anything in a way a 5 year old would understand. my daughter is 5 and asks a lot of questions. e.g.'what is risk' or 'how are waves formed' or 'how does rain happen?'"
                    )
                ]
            }
        )
    )

    print(f"\n printing response list ...")

    for msg in response["messages"]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f" {role}: {msg.content}")


if __name__ == "__main__":
    # simple_graph_demo()
    # annotated_graph_demo()
    annotated_ai_example()
