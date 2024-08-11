from fastapi import HTTPException, APIRouter
from mysql.connector import Error
from langchain_openai import ChatOpenAI
from langchain import hub
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
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
from routes.common import llm, vectorstore, retriever, korean_to_english_map
import json
import asyncio
import time
import config
import mysql.connector
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

card_router = APIRouter()


def get_card_data_last_half_year(user_id: int):

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
            query = f"""
            SELECT 
                category, 
                SUM(amount) AS total_amount 
            FROM 
                transactions 
            WHERE 
                user_id = {user_id}
                AND date >= DATE_ADD(CURRENT_DATE, INTERVAL -6 MONTH) 
                AND amount < 0 
                AND category != "기타"
            GROUP BY 
                category 
            ORDER BY 
                total_amount ASC 
            LIMIT 3;

            """

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@card_router.get("/recommendation")
def get_recommendation(user_id: int):
    return card_recommend(user_id)


def card_recommend(user_id: int):
    import ast

    records = get_card_data_last_half_year(user_id)
    template = """
    당신은 KB국민은행에 대한 질문에 답변하는 작업을 돕는 은행원입니다.
    다음에 제공된 맥락을 사용하여 질문에 한국말로 답변하십시오.

    다음은 사용자의 지난 6개월 간 가장 지출이 많았던 카테고리 TOP3입니다:\n
    """ + str(records) + """

    각 TOP3의 카테고리에 적합한 카드를 추천해주세요.
    특히 몇 %를 적립할 수 있으며 할인 받을 수 있는 지를 계산하여 아래 예시와 같은 tuple 배열 형식으로 대답하세요.
    무조건 아래 예시와 같은 tuple 배열 형식으로 대답하세요. 절대 주어진 예시 외 다른 tuple index는 추가하지 마세요.
    
    [
        [('category1', 식비),
            ("card_title", "쿠팡 와우 카드"),
            ("summary_card_benefit", "쿠팡, 쿠팡이츠, 쿠팡플레이 2% 쿠팡캐시 적립"),
            ("money_benefit", "연 9,700원의 혜택을 받을 수 있어요!")],
        [("category2", 여가)
            ("card_title", "트래블러스 체크카드"),
            ("summary_card_benefit", "해외 이용 수수료 1.25% 면제, 철도 할인 5,000원"),
            ("money_benefit", "연 10,000원의 혜택을 받을 수 있어요!)]
        ,
            [("category3", "편의"),
            ("card_title", "데일리 WE:SH 카드"),
            ("summary_card_benefit", "국내 가맹점 0.5%, 편의점,커피,올리브영 10%"),
            ("money_benefit", "연 3,200원의 혜택을 받을 수 있어요!)]
    ]

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    rag_response = rag_chain.invoke("각 TOP3 카테고리에 대해 나에게 적합한 카드 추천해줘.")
    tuple_array = ast.literal_eval(rag_response)

    result_dict = {}

    for item in tuple_array:
        korean_category = item[0][1]
        english_category = korean_to_english_map.get(korean_category, korean_category)  # 매칭이 없을 경우 한국어 그대로 사용
        card_info = {k: v for k, v in item[1:]}
        result_dict[english_category] = card_info

    # 결과 출력
    return result_dict
    
    


if __name__ == "__main__":
    
    # 시작 시간 기록
    start_time = time.time()
    card_recommend(5)
    # 종료 시간 기록
    end_time = time.time()

    # 실행 시간 계산
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
