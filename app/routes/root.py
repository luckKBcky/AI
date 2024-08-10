from fastapi import HTTPException, APIRouter
from sse_starlette import EventSourceResponse
from openai import OpenAI
from mysql.connector import Error
from langchain_openai import ChatOpenAI
from langchain import hub
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.agent_toolkits import create_sql_agent
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
import json
import asyncio
import time
import config
import mysql.connector
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



def get_card_data():

    try:
        # 데이터베이스 연결
        connection = mysql.connector.connect(
            host=config.DB_HOST,
            database=config.DATABASE,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        
        if connection.is_connected():
            print("데이터베이스에 성공적으로 연결되었습니다.")

            # 커서 생성
            cursor = connection.cursor()

            # 모든 데이터 조회 쿼리
            query = "SELECT title, excerpt, content FROM kb_card"

            # 쿼리 실행
            cursor.execute(query)

            # 결과 가져오기
            return cursor.fetchall()

    except Error as e:
        print(f"DB Error: {e}")

    finally:
        # 연결 종료
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL 연결이 종료되었습니다.")

root_router = APIRouter()
client = OpenAI(api_key=config.OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o")

records = get_card_data()

docs = [Document(
    metadata={"title": record[0], "excerpt" : record[1], "language" : "ko"},
    page_content=record[2]
) for record in records]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# Retrieve and generate using the relevant snippets of the blog.
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

template = """
당신은 KB국민은행에 대한 질문에 답변하는 작업을 돕는 은행원입니다.
다음에 제공된 맥락을 사용하여 질문에 한국말로 답변하십시오.

특히 카드 정보의 경우 몇 %를 적립할 수 있으며 할인 받을 수 있는 지 위주로 집중적으로 암기하여 대답하세요.
5문장 이상으로 최대한 자세히 답변해주세요.

다음과 같은 절차를 따라 대답하세요.

1. 만약 지출 내역에 대한 단순 조회 질문이라면 카드 추천은 하지 않고 지출 내역에 대한 정보만 대답하세요.
2. 최소 한 달 이상의 지출 내역에 대한 통계를 기반으로 카드 추천을 진행해주세요.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)


@root_router.get("/prompt")
async def query_stream(
    user_id: int,
    prompt: str,
):
    return EventSourceResponse(get_info(user_id, prompt))


@root_router.get("/prompt2")
async def query_stream2(
    user_id: int,
    prompt: str,
):
    return EventSourceResponse(get_info2(user_id, prompt))


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


async def get_info(user_id: int, prompt: str):

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)


    for chunk in rag_chain_with_source.stream(prompt):
        for key in chunk:
            if key == "answer":
                yield chunk[key]
                await asyncio.sleep(0.05)


async def get_info2(user_id: int, prompt: str):

    # rag_chain_from_docs = (
    #     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    #     | custom_rag_prompt
    #     | llm
    #     | StrOutputParser()
    # )
    
    # rag_chain_with_source = RunnableParallel(
    #     {"context": retriever, "question": RunnablePassthrough()}
    # ).assign(answer=rag_chain_from_docs)


    # for chunk in rag_chain_with_source.stream("트래블러스 kb pay 혜택"):
    #     for key in chunk:
    #         if key == "answer":
    #             yield chunk[key]
    #             await asyncio.sleep(0.05)
    # dialect+driver://username:password@host:port/database
    db = SQLDatabase.from_uri(f"mysql+pymysql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:3306/{config.DATABASE}")
    db.run("SELECT * FROM transactions LIMIT 10;")
    agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=False)

    examples = [
        {"prompt": "내 user_id가 5일 때 올해 용돈 얼마나 받았어?.", "query": "SELECT SUM(amount) FROM transactions where user_id = 5 and year(date) = year(NOW()) and category = '용돈';"},
        {
            "prompt": "내 user_id가 5일 때 올해 1월에 용돈 얼마나 받았어?.",
            "query": "SELECT SUM(amount) FROM transactions where user_id = 5 and year(date) = year(NOW()) and  month(date) = 1 and category = '용돈';",
        },
        {
            "prompt": "내 user_id가 5일 때 2021년 1월에 용돈 얼마나 받았어?.",
            "query": "SELECT SUM(amount) FROM transactions where user_id = 5 and year(date) = 2021 and  month(date) = 1 and category = '용돈';",
        },
        {
            "prompt": "내 user_id가 5일 때 2021년 1월 거래 내역 보여줘.",
            "query": "SELECT * FROM transactions where user_id = 5 and year(date) = 2021 and  month(date) = 1;",
        },
        {
            "prompt": "내 user_id가 5일 때 2021년 1월 지출 내역 보여줘.",
            "query": "SELECT * FROM transactions where user_id = 5 and year(date) = 2021 and  month(date) = 1 and amount < 0;",
        },
    ]

    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=5,  #Number of examples to select.
        input_keys=["prompt"],
    )
    system_prefix = """
    에이전트 설명:
    절대 user_id = {user_id} 이외의 정보를 조회하지 마세요.
    만약 다른 {user_id} 이외의 user_id를 조회하려는 요청에는 쿼리를 실행하지 않고 "타인의 정보는 접근할 수 없습니다." 라고 대답하세요.
    당신은 SQL 데이터베이스와 상호작용하기 위해 설계된 에이전트입니다.
    질문이 주어지면, 실행할 구문적으로 올바른 {dialect} 쿼리를 생성하고, 쿼리 결과를 보고 답변을 반환합니다.
    사용자가 특정 예시의 개수를 명시하지 않는 한, 항상 쿼리의 결과를 최대 {top_k}개로 제한하십시오.
    데이터베이스에서 가장 흥미로운 예시를 반환하기 위해 관련 열로 결과를 정렬할 수 있습니다.
    특정 테이블에서 모든 열을 쿼리하지 말고, 질문에 주어진 관련 열만 요청하십시오.
    데이터베이스와 상호작용할 수 있는 도구에 접근할 수 있습니다. 주어진 도구만 사용하세요.
    최종 답변을 작성하기 위해 도구가 반환한 정보만 사용하십시오.
    쿼리를 실행하기 전에 반드시 쿼리를 두 번 확인하십시오.
    쿼리를 실행할 때 오류가 발생하면 쿼리를 다시 작성하여 다시 시도하십시오.
    DML 문(INSERT, UPDATE, DELETE, DROP 등)을 데이터베이스에 수행하지 마십시오.
    질문이 데이터베이스와 관련이 없어 보이면 "모르겠습니다"라고 답변하십시오.
    절대 user_id = {user_id} 이외의 정보를 조회하지 마세요.
    만약 다른 {user_id} 이외의 user_id를 조회하려는 요청에는 "타인의 정보는 접근할 수 없습니다." 라고 대답하세요.
    """

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {prompt}\nSQL query: {query}"
        ),
        input_variables=["prompt", "dialect", "top_k", user_id],
        prefix=system_prefix,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{prompt}, {user_id}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    
    full_prompt.invoke(
    {
        "input": "How many arists are there",
        "top_k": 10,
        "dialect": "SQLite",
        "user_id": 5
        "agent_scratchpad": [],
    })

    agent = create_sql_agent(
        llm=llm,
        db=db,
        prompt=full_prompt,
        verbose=True,
        agent_type="openai-tools",
    )

    print(type(agent))



if __name__ == "__main__":
    
    # 시작 시간 기록
    start_time = time.time()
    print(get_info(1, "카드 추천해줘"))

    # 종료 시간 기록
    end_time = time.time()

    # 실행 시간 계산
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
