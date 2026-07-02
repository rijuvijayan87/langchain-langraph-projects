from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()


class GraphStateSchema(TypedDict):
    start: str
    index: int
    output: str


def demo_graph():
    """Calling demo graph function"""

    def process(state: GraphStateSchema) -> dict:
        return {
            "start": state["start"],
            "output": state["start"].casefold() + "_processed_output",
            "index": state["index"] + 1,
        }

    def casefold_input(state: GraphStateSchema) -> dict:
        return {"output": state["start"].casefold() + "_processed_output"}

    def increment_index(state: GraphStateSchema) -> dict:
        return {"index": state["index"] + 1}

    graph = StateGraph(GraphStateSchema)

    graph.add_node("casefold_input", casefold_input)
    graph.add_node("increment_index", increment_index)
    graph.add_edge(START, "casefold_input")
    graph.add_edge("casefold_input", "increment_index")
    graph.add_edge("increment_index", END)
    app = graph.compile()

    # Save PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("langraph_core_graph.png", "wb") as f:
        f.write(png_bytes)
    print("Graph saved to langraph_core_graph.png")

    response = app.invoke(
        {"start": "THIS IS A PROCESSED DATA", "output": "", "index": 0}
    )

    print(f"response {response}")


if __name__ == "__main__":
    demo_graph()
