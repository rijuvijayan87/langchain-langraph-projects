# Steps to manage memory
# --------------------------
# 1. add a llm chat interface that get {historical_message} and {current_user_message}
# 2. Use MessagesPlaceHolder to store the response for history
# 3. Manage a list of all history and response as a register. Take the oldest 4 messages (2 pairs) and summarise it
# 4. Format the 4 messages to a summariser llm flow and get the conversations summarised.
# 5. Reset the register to only have the latest messages

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


def summarize_conversations():
    chat_llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)
    summarize_llm = init_chat_model(model="gpt-4o-mini`", temperature=0)

    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a meticulous assistant. you are good at responding to queries in a concise and precise manner. You excel at 
                    - Giving answers to questions that you know and call out things you do not know. 
                    - For unknowns, you politely let the user know that you do not know the answer to the given question.
                    You have access to the {summarized_conversation_history} to make sure you do not lose any context,
                """,
            ),
            MessagesPlaceholder(variable_name="response_history"),
            ("human", "{user_question}"),
        ]
    )

    chat_chain = chat_prompt_template | chat_llm | StrOutputParser()

    summarized_prompt_template = ChatPromptTemplate.from_template("""
        You are an amazing summarizer of content. You will be provided with a list of Human and AI messages. Your role is to 
            - Summarise the whole converstation in 2 - 3 sentences
            - do not change the meaning of the convesration
            - special care must be taken to not dilute the meaning of the conversation
            - prepare summary so that when this summary is passed over to another the agent / llm, they get sufficient context 
            about the ongoing conversation
                                                                  
        summarized context from memory : {summarized_context}
        context from current memory: {current_message}
        """)

    summarize_chain = summarized_prompt_template | summarize_llm | StrOutputParser()

    messages = [
        "Hi! my name is Riju!",
        "I am a software engineer",
        "I want to learn more about Langchain and how it is used to build AI applications",
        "what are some of the practical usecases of langchain in my area of work specifically. do not give me usecases from any other field",
        "how would you introduce me to another person?",
    ]

    conversation_history: list[BaseMessage] = []
    CONVERSATION_QUEUE_SIZE = 4
    summarized_message = ""

    for index, msg in enumerate(messages):
        print("*" * 60)
        print(f"{index+1}. Question: {msg}")
        summarized_context = ""

        chat_response = chat_chain.invoke(
            {
                "user_question": msg,
                "response_history": conversation_history,
                "summarized_conversation_history": (
                    summarized_message
                    if summarized_message
                    else "No conversation history yet."
                ),
            }
        )
        conversation_history.append(HumanMessage(content=msg))
        conversation_history.append(AIMessage(content=chat_response))

        print(f"\nResponse from llm received <<< : {chat_response}")
        print(f"\nConversation history length: {len(conversation_history)}")

        def format_old_conversations(conversation: BaseMessage):
            role = "Human" if type(conversation) is HumanMessage else "AI"
            return f"{role}: {conversation.content}"

        # strip of the old message and summarise if the queue size is greater than CONVERSATION_QUEUE_SIZE
        if len(conversation_history) > CONVERSATION_QUEUE_SIZE:
            print("#" * 60)
            print(
                f"\n --- Conversation history should be summarised as the conversation length is {len(conversation_history)} --- "
            )

            old_conversations = conversation_history[:-CONVERSATION_QUEUE_SIZE]
            print(
                f"\n length of old conversations to be summarised {len(old_conversations)}"
            )

            for old_conv in old_conversations:
                summarized_context += format_old_conversations(old_conv)

            print(f"\n --- Message to be summarized --- \n")
            print(summarized_context)

            summarized_message = summarize_chain.invoke(
                {
                    "summarized_context": summarized_context,
                    "current_message": summarized_message,
                }
            )

            print(f"\n --- Summarised message: --- ")
            print(summarized_message)

            conversation_history = conversation_history[-CONVERSATION_QUEUE_SIZE:]
            print(
                f" --- length of conversation history after summarization : {len(conversation_history)}"
            )
            print("#" * 60)


if __name__ == "__main__":
    summarize_conversations()
