from dotenv import load_dotenv
from typing import List

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


def demo():
    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0,
            streaming=True,
            max_retries=3,
        ),
        "gpt-4": init_chat_model(
            model="gpt-4",
            temperature=0,
            streaming=True,
            max_retries=3,
        ),
    }

    for model_name, model in models.items():
        response = model.invoke("explain langchain in 1 sentence only")
        print(f"response from {model_name} : {response.content}")


def exercise_multi_model(question: str, models: List[str]):
    """Create a function that
    1. Takes a question and a list of models
    2. Gets response from all models
    3. Returns dict of {model_name: response}

    Test with question="what is ai?", models = ["gpt-4o-mini", "gpt-4o"]
    """
    response_from_ai = {}
    for model_name in models:
        model = init_chat_model(
            model=model_name,
            temperature=0,
            max_tokens=5000,
            max_retries=3,
            streaming=True,
        )

        response = model.invoke(question)

        response_from_ai[model_name] = response.content

    return response_from_ai


def test_prompt_templates():

    # prompt_template = ChatPromptTemplate.from_template(
    #     "tell me a {theme} story about {topic} in a single sentence"
    # )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are assistant who is {mood}"),
            ("human", "tell me a joke about {topic}"),
        ]
    )

    message = prompt_template.format_prompt(mood="angry", topic="software engineering")

    print(message)

    model = init_chat_model(model="gpt-4o-mini", temperature=0, max_tokens=500)
    parser = StrOutputParser()

    chain = model | parser

    response = chain.invoke(message)
    print(response)


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_chain()
    # demo_stream_chain()
    # excercise_first_chain()
    # new_way_of_chain()
    # demo()

    ########
    # models = ["gpt-4o-mini", "gpt-4o"]
    # response = exercise_multi_model("what is ai?", models=models)
    # print(response)
    ########

    test_prompt_templates()
