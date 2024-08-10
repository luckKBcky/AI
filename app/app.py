from fastapi import FastAPI
from routes.root import root_router
from fastapi.middleware.cors import CORSMiddleware
import config
import uvicorn


app = FastAPI()
app.include_router(root_router,  prefix="/ai")
# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)
if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)

    