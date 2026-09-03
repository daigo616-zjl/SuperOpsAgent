"""用户接口

提供最小用户注册能力，作为长期记忆的隔离主体（user_id）来源。
"""

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.core.postgres import postgres_manager

router = APIRouter()


class CreateUserRequest(BaseModel):
    display_name: str = Field(default="", max_length=64)


@router.post("/users")
async def create_user(request: CreateUserRequest):
    """注册用户，返回作为长期记忆主体的 user_id"""
    try:
        with postgres_manager.engine.begin() as connection:
            row = connection.execute(
                text(
                    "insert into users (display_name) "
                    "values (:display_name) "
                    "returning id"
                ),
                {"display_name": request.display_name},
            ).scalar_one()

        user_id = str(row)
        logger.info(f"用户注册成功: {user_id}")
        return {
            "code": 200,
            "message": "success",
            "data": {"user_id": user_id, "display_name": request.display_name},
        }
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {"user_id": None, "display_name": None, "errorMessage": str(e)},
        }
