import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableWithMessageHistory

load_dotenv()

DB_PATH = "./chat_history_summarized.db"
MAX_MESSAGES = 10


class SummarizingChatMessageHistory(BaseChatMessageHistory):
    """
    Wraps SQLChatMessageHistory. After every write, if the total message count
    hits MAX_MESSAGES, it summarizes the older messages with the LLM and replaces
    them in SQLite with a single SystemMessage — keeping only the 2 most recent
    messages fresh.
    """

    def __init__(self, session_id: str, llm, max_messages: int = MAX_MESSAGES):
        self.store = SQLChatMessageHistory(
            session_id=session_id, connection=f"sqlite:///{DB_PATH}"
        )
        self.llm = llm
        self.max_messages = max_messages

    @property
    def messages(self) -> list[BaseMessage]:
        return self.store.messages

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.store.add_messages(messages)
        all_messages = self.store.messages
        if len(all_messages) >= self.max_messages:
            self._summarize_and_replace(all_messages)

    def clear(self) -> None:
        self.store.clear()

    def _summarize_and_replace(self, messages: list[BaseMessage]) -> None:
        to_summarize = messages[:-2]  # everything except the most recent exchange
        recent = messages[-2:]        # keep the last human + AI message fresh

        summarization_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Summarize the following conversation concisely. "
                "Preserve all user preferences, key facts, and decisions mentioned.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

        summary = (summarization_prompt | self.llm | StrOutputParser()).invoke(
            {"messages": to_summarize}
        )

        self.store.clear()
        self.store.add_messages([
            SystemMessage(content=f"Summary of earlier conversation: {summary}"),
            *recent,
        ])
        print(f"\n[Summarized {len(to_summarize)} messages into 1 summary message]")


def run():
    llm = init_chat_model(model="gpt-4o-mini")

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return SummarizingChatMessageHistory(
            session_id=session_id, llm=llm, max_messages=MAX_MESSAGES
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember user preferences."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config: RunnableConfig = {"configurable": {"session_id": "summarizing_user"}}

    # 12 messages to trigger summarization at message 10
    test_messages = [
        "I prefer dark mode in all apps",
        "My favourite language is Python",
        "I work in the data engineering space",
        "I prefer concise answers, no fluff",
        "I use VS Code as my editor",
        "What editor do I use?",
        "What language do I prefer?",
        "What is my job domain?",
        "Do I prefer dark or light mode?",
        "What are all my preferences so far?",  # message 10 — triggers summarization
        "Add one more: I prefer tabs over spaces",
        "List all my preferences including the latest one",
    ]

    print("=" * 60)
    print("SUMMARIZING MEMORY CHATBOT")
    print(f"Summarization triggers at {MAX_MESSAGES} messages")
    print("=" * 60)

    for message in test_messages:
        print(f"\nUser: {message}")
        response = chain_with_history.invoke({"input": message}, config=config)
        print(f"AI: {response}")
        print(f"[Messages in DB: {len(get_session_history('summarizing_user').messages)}]")


if __name__ == "__main__":
    run()
