from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model


def demo_basic_chain():
    """Basic LCEL and runnables"""

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. now answer this {question} in a short, crisp, no aifluff fashion"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    response = chain.invoke(input={"question": "tell me more about langgraph"})

    print(f"response : {response}")


def demo_batch_chain():
    """Demonstrate batch chain with LCEL"""

    prompt = ChatPromptTemplate.from_template(
        "Translate the {text} to Spanish. just respond with the translation and no fillers"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    parser = StrOutputParser()

    chain = prompt | llm | parser

    inputs = [
        {"text": "how are you?"},
        {"text": "my name is riju?"},
        {"text": "i am a software engineer?"},
    ]

    responses = chain.batch(inputs=inputs)

    for text, translation in zip(inputs, responses):
        print(f"text: {text['text']} -> {translation}")


def demo_stream_chain():
    """Demonstrate response streaming"""

    prompt = ChatPromptTemplate.from_template("explain about {topic} in 2 sentences")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    parser = StrOutputParser()

    chain = prompt | llm | parser

    print("streaming output:")
    for chunk in chain.stream({"topic": "langgraph and langchain"}):
        print(chunk, end="", flush=True)
    print()


def excercise_first_chain():
    """Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as string

    Test with: product="AI Course", "audience=developers"
    """

    prompt = ChatPromptTemplate.from_template(
        "You are an expert marketing person. Create me a marketing tagline for the {product} and the {audience}. It has to be catchy but compelling. No ai-fluff"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    parser = StrOutputParser()

    chain = prompt | llm | parser

    response = chain.invoke(
        {"product": "browser test automation", "audience": "software engineers"}
    )

    print(response)


def new_way_of_chain():
    """Demonstrate init_chat_model — swap providers without changing chain logic."""

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence. No fluff."
    )
    parser = StrOutputParser()

    model = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=0)

    chain = prompt | model | parser

    response = chain.invoke({"topic": "LangGraph"})
    print(f"response: {response}")


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_chain()
    # demo_stream_chain()
    # excercise_first_chain()
    new_way_of_chain()
