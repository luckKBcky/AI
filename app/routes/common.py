from openai import OpenAI
from mysql.connector import Error
from langchain_openai import ChatOpenAI
import os
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import json
import config
import mysql.connector

english_to_korean_map = {
    'food': '식비',
    'etc': '기타',
    'allowance': '용돈',
    'leisure': '여가',
    'clothing': '옷',
    'dating': '데이트',
    'savings': '저축',
    'beauty': '미용',
    'studies': '학업',
    'convenience': '편의'
}
korean_to_english_map = {v: k for k, v in english_to_korean_map.items()}


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

client = OpenAI(api_key=config.OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o")

records = get_card_data()

docs = [Document(
    metadata={"input": "", "query" : "","title": record[0], "excerpt" : record[1], "language" : "ko"},
    page_content=record[2]
) for record in records]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma(collection_name="card").from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


if __name__ == "__main__":
    
    # 시작 시간 기록
    print("!")
