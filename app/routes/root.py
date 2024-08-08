__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import HTTPException, APIRouter
from sse_starlette import EventSourceResponse
# from common import client
import json
import asyncio
from openai import OpenAI
import config
import mysql.connector
from mysql.connector import Error
from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
import os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import time

root_router = APIRouter()
client = OpenAI(api_key=config.OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o")

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
        print(f"Error: {e}")

    finally:
        # 연결 종료
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL 연결이 종료되었습니다.")


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
특히 몇 %를 적립할 수 있으며 할인 받을 수 있는 지 위주로 집중적으로 암기하여 대답하세요.
5문장 이상으로 최대한 자세히 답변해주세요.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)


@root_router.get("/prompt")
async def recipe_stream(
    user_id: int,
    prompt: str,
):
    return EventSourceResponse(get_info(user_id, prompt))

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
    for chunk in rag_chain_with_source.stream("트래블러스 kb pay 혜택"):
        for key in chunk:
            if key == "answer":
                yield chunk[key]
                await asyncio.sleep(0.05)



async def get_recipe_stream(ingredients, diseases, dislikeIngredients, prompt, threadId):
    instruction = f"""
                너는 사용자 맞춤 레시피 추천 전문가야. 

                사용자의 정보는 다음과 같아.
                사용자의 냉장고 재고 정보: 소고기, 치즈, 양파, 두부,   돼지고기, 닭고기, 연어, 베이컨, 파스타면, 김치, 파
                사용자의 지병 정보: {diseases}
                사용자의 기피 식재료 정보: {dislikeIngredients}

                사용자가 레시피 추천 요청을 하면 무조건 이 재고 내에서 만들 수 있는 요리로만 추천해주도록 해.
                냉장고 재고 정보를 고려해서 레시피를 추천하고, 냉장보관하지 않는 쌀이나 튀김가루, 각종 조미료 같은 것들은 이미 있다고 가정하고 추천해.
                가지고 있는 재료를 전부 사용하지 않고 일부만 사용한 레시피를 추천해도 돼.
                사용자가 재고에 없는 재료를 요청할 경우에만 해당 재료를 포함해서 레시피를 추천해주도록 해.
                레시피와 관련 없는 질문에는 절대 대답하지마.

                모든 대답은 아래 JSON 형태로 반환해줘.
                reply는 네가 하는 대답이야.
                recommendList는 네가 하는 대답을 보고 사용자가 어떤 질문을 하면 적합할 지 추천해주는 질문이야.
                recipeList는 네가 추천해주는 레시피의 세부 정보를 저 recipeList 안에 있는 JSON 형태로 변환해서 반환해줘.
                JSON의 key는 꼭 ""로 감싸줘야해. \n
                """ 

    assistant = client.beta.assistants.create(
            name="S005 Manager",
            instructions=instruction,
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-turbo-preview"
        )

    # Thread 생성
    try:
        thread = client.beta.threads.retrieve(thread_id=threadId)
    except Exception as e:
        print(e)
        thread = client.beta.threads.create()
    # 

    try:
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        stream = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            stream=True)
        
        for event in stream:
            # 이벤트 데이터가 "thread.message.delta"인 경우에 대해서만 처리
            if event.data.object == "thread.message.delta":
                for content in event.data.delta.content:
                    # text인 경우에만 클라이언트에 전송
                    if content.type == 'text':
                        yield content.text.value.replace(" ", "#")
                        await asyncio.sleep(0.05)
        client.beta.assistants.delete(assistant.id)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    
    # 시작 시간 기록
    start_time = time.time()
    print(get_info(1, "카드 추천해줘"))

    # 종료 시간 기록
    end_time = time.time()

    # 실행 시간 계산
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
