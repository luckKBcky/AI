from pydantic import BaseModel


class PromptRequest(BaseModel):
    threadId: str
    assistantId: str
    prompt: str

    class Config:
        json_schema_extra = {
            "example": {
                "threadId": "thread_sRT1we1w421",
                "assistantId": "asst_sRT1we1w421",
                "prompt" : "저녁 메뉴 추천해줘"
            }
        }
