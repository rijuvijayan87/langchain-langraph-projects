import os
import pathlib

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableWithMessageHistory

load_dotenv()


def example_persistent_memory():
    """
    EXERCISE: Build a chatbot with
    1. Persistent memory (SQLite)
    2. Automatic Summarization after 10 messages
    3. User Preference Tracking

    Hint: Combine RunnableMessageWithHistory with SQLChatMessageHistory
    """
    print("=" * 60)
    print("PERSISTENT MEMORY CHATBOT")
    print("=" * 60)

    DB_PATH = "./chat_history.db"

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=session_id, connection=f"sqlite:///{DB_PATH}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Remember user preferences."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    llm = init_chat_model(model="gpt-4o-mini")

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config: RunnableConfig = {"configurable": {"session_id": "persistent_user"}}

    print("\nPersistent memory chatbot:")
    print("(Messages saved to SQLite database)\n")

    test_messages = [
        "Remember that I prefer dark mode themes",
        "What theme do I prefer?",
    ]

    for message in test_messages:
        print(f"User: {message}")
        response = chain_with_history.invoke({"input": message}, config=config)
        print(f"AI: {response}\n")

    print(f"Database created: {DB_PATH}")
    print("Messages persist across restarts!")


if __name__ == "__main__":
    example_persistent_memory()
