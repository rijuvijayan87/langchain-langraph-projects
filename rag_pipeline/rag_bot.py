from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    langsmith_tracing: bool = False
    langsmith_endpoint: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None
    tavily_api_key: str | None = None


class LLMSettings(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=0, le=5)
    max_tokens: int = Field(default=1024, gt=0)


settings = Settings()


class LLMConfig:

    def __init__(self, config: LLMSettings, model: str = "gpt-4o"):
        self.model = model
        self.llm_config = init_chat_model(
            model=self.model,
            model_provider="openai",
            temperature=config.temperature,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def llm(self) -> BaseChatModel:
        if self.llm:
            return self.llm_config
        raise ValueError("llm not initiated")

    def get_prompt(self):
        prompt_template = ChatPromptTemplate.from_template("test")


class QABot:

    def ask(self, question: str) -> str:
        print(f"question -> {question}")
        return "answer is Riju"


if __name__ == "__main__":

    # load the md files from location
    # use a embedding model to
    # Embed context

    # add data to rag

    # retrieve and llm calls
    # Bootstrap LLM configs
    # llm base url and configuration
    config = LLMSettings(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=0.1,
        max_retries=1,
        max_tokens=2000,
    )

    llm_config = LLMConfig(config=config, model="claude-sonnet-4-6")
    response = llm_config.llm().invoke("what is langchain?")
    print(response.content)
