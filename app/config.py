from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HOST = os.getenv("HOST")
DB_HOST = os.getenv("DB_HOST")
PORT = int(os.getenv("PORT"))
USER = os.getenv("USER")
PASSWORD = os.getenv("PW")
DATABASE = os.getenv("DATABASE")
