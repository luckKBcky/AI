from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE = os.getenv("DATABASE")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
