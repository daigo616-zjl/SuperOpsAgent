"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    id: str = Field(..., description="会话 ID", alias="Id")
    question: str = Field(..., description="用户问题", alias="Question")
    user_id: str | None = Field(
        default=None,
        description="用户 ID（长期记忆主体）；缺省时以会话 ID 作为记忆主体",
        alias="UserId",
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Id": "session-123",
                "Question": "什么是向量数据库？",
                "UserId": "3f1c9a2e-0000-0000-0000-000000000000",
            }
        }


class ClearRequest(BaseModel):
    """清空会话请求"""

    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True
