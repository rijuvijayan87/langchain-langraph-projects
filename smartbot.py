from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


class QAResponse(BaseModel):
    question: str = Field(description="The original question asked")
    answer: str = Field(description="The concise answer to the question")
    sources: list[str] = Field(default_factory=list, description="List of sources used")
    followup_questions: list[str] = Field(
        default_factory=list,
        description="follow up questions based on the answer of the LLM",
    )


class QABot:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.5) -> None:
        self.model = model
        self.temperature = temperature
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an experienced helpful assistant. You respond very meticulously to serve the user with the answers to the question being asked

                    Your Rules:
                    - You need to be courteous and welcoming to the question being asked
                    - if you do not know the answer, say so. do not beat around the bush
                    - keep it straight and no AI-fluff required
                    - most of the users of the system might be non-technical users. so keep it simple and explain in layman terms
                    - You should provide sources where you are deriving your answers from
                    - You should also prepare follow up question for the users, so they can benefit (if they want to ask follow up questions)
                    - keep your responses to 5 sentences
            """,
                ),
                ("human", "{question}"),
            ]
        )
        self.llm = init_chat_model(
            model=self.model,
            temperature=self.temperature,
            max_retries=3,
            max_tokens=1000,
        ).with_structured_output(QAResponse)
        self.chain = self.prompt_template | self.llm

    def ask(self, question: str) -> QAResponse:
        try:
            result = self.chain.invoke({"question": question})
            if not isinstance(result, QAResponse):
                raise ValueError(f"unexpected response type: {type(result)}")
            return result
        except Exception as e:
            print(f"error occured in llm call {e}")
            return QAResponse(
                question=question,
                answer=f"error occured in llm call {e}",
                sources=[],
                followup_questions=[],
            )


if __name__ == "__main__":
    bot = QABot(model="gpt-4o-mini")
    result = bot.ask(
        "what are some of usecases of using langraph and langchain in software engineering"
    )

    print(f"question asked -> {result.question}")
    print(f"answer -> {result.answer}")
    print(f"sources -> {result.sources}")
    print(f"follow up questions -> {result.followup_questions}")
