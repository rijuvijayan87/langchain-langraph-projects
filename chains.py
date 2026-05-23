from itertools import chain

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

load_dotenv()


class Sentiment(BaseModel):
    sentiment: str = Field("sentiment analysis result")
    keywords: str = Field("keyword from sentiment analysis")
    summary: str = Field("summary of analysis")


def combine_results(r: dict[str, Sentiment]) -> Sentiment:
    return Sentiment(
        sentiment=r["sentiment"].sentiment,
        keywords=r["keywords"].keywords,
        summary=r["summary"].summary,
    )


def call_chain():
    llm = init_chat_model(
        model="gpt-4o-mini", temperature=0, max_tokens=500, max_retries=3
    ).with_structured_output(Sentiment)

    sentiment_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "you are a sentiment analyser. you are expected to analysis the {context} and tell me the sentiment of the given context",
            ),
            ("human", "{context}"),
        ]
    )
    keyword_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "you are a keyword analyser. you are expected to analysis the {context} and give me comma seperated keywords",
            ),
            ("human", "{context}"),
        ]
    )
    summarise_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "you are a summariser. you are expected to analysis the {context} and summarise the whole context with details, action items and bullet points",
            ),
            ("human", "{context}"),
        ]
    )

    analysis_chain = RunnableParallel(
        sentiment=sentiment_analysis_prompt | llm,
        keywords=keyword_analysis_prompt | llm,
        summary=summarise_prompt | llm,
    ) | RunnableLambda(combine_results)

    statement = """
       Subject: Re: Quick feedback on your experience with TestPilot Pro — would love your thoughts

Daniel,

I've been waiting for an excuse to write this email. Adopting TestPilot Pro has been,
without exaggeration, the single best engineering decision Northwind Logistics has made
in the last two years. I want to make sure you, your team, and your leadership hear that
loud and clear.

Going through your questions:

1. Onboarding & Setup:
   Flawless. Absolutely flawless. The documentation is the gold standard — clear,
   accurate, and beautifully organized. Both the Jenkins and GitHub Actions integrations
   worked perfectly on the very first attempt. Meera, your onboarding engineer, was
   extraordinary; she felt like a member of our own team. We finished onboarding two full
   weeks ahead of schedule, which has never happened with any vendor we've worked with.

2. Measurable Improvements:
   The results have exceeded every expectation we set:
     - Regression suite runtime: 3h 40m → 38 minutes. A nearly 6x improvement.
     - Flaky test rate on the payments module: down 92%.
     - Order-tracking flakiness: down 88%.
     - Release cadence: from biweekly to daily deployments, with higher confidence than
       we've ever had.
     - Production incidents tied to regressions: down to zero over the last 90 days.
   Our CFO asked me last week what changed — I told him it was TestPilot Pro, full stop.
   It's now a line item he proactively defends in budget reviews.

3. AI Test Generation:
   This feature is genuinely magical. We're seeing 95%+ usable output across the board,
   even for our most complex multi-step workflows. The assertion quality is razor-sharp,
   the generated tests read like something a senior engineer wrote, and Priya's team has
   reclaimed roughly 20 engineer-hours per sprint. Two of our junior QAs have told me it
   has accelerated their learning more than any training program we've offered. Whatever
   you're doing on the AI side is, frankly, industry-leading.

4. Frustrations:
   None. Truly none. I sat with this question for ten minutes trying to find something
   constructive to add, and I came up empty. Every rough edge we've encountered has
   already been addressed in a release before we even had to ask.

5. NPS:
   11 out of 10. I've personally recommended TestPilot Pro to seven engineering leaders
   in my network, four of whom have already signed contracts. Consider me your unpaid
   sales rep.

A few more things I want on the record:
   - Your support team is the best I have ever worked with — period. Every ticket
     answered in under an hour, every fix correct on the first try. Arjun deserves a
     raise, a promotion, and a parade.
   - The Slack failure alerts are a work of art. Our on-call engineers have stopped
     dreading pages.
   - The ROI is, conservatively, 9x annual cost. My finance team double-checked the math
     because they didn't believe it the first time.
   - The product roadmap you shared last quarter is exactly what we'd want — we feel
     genuinely heard as customers.

Looking ahead, here's what we'd love to do:
   - Expand TestPilot Pro to all five engineering orgs at Northwind by end of next
     quarter.
   - Upgrade to Enterprise immediately — SSO, audit logs, and dedicated TAM, please send
     the paperwork this week.
   - Sign a multi-year renewal at favorable terms; I'd like to lock this in before
     anyone else tries to poach you.
   - Have Priya keynote your next customer summit. She's already drafting the talk.
   - Co-author a public case study with your marketing team. We want to be on record as
     advocates.

Riju, working with you has restored my faith in vendor relationships. You follow up, you
listen, you ship what you promise, and your team treats us like partners rather than
accounts. That is rarer than it should be, and I want you to know it does not go
unnoticed.

Let's absolutely take that 15-minute call — though I suspect it'll run longer because I
have a lot more good things to say.

With genuine appreciation,
Daniel Okafor
Director of Engineering, Northwind Logistics
daniel.okafor@northwindlogistics.example

    """

    results = analysis_chain.invoke({"context": statement})
    print(f"sentiment : {results.sentiment}")
    print(f"keywords : {results.keywords}")
    print(f"summary : {results.summary}")


if __name__ == "__main__":
    call_chain()
