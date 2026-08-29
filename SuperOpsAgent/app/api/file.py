"""PostgreSQL 权威知识文档与索引任务 API。"""

from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.models.knowledge import KnowledgeDocumentCreate, KnowledgeDocumentUpdate
from app.services.knowledge_repository import (
    DocumentConflictError,
    DocumentNotFoundError,
    knowledge_repository,
)

router = APIRouter()
ALLOWED_EXTENSIONS = {"txt", "md", "markdown"}
MAX_FILE_SIZE = 10 * 1024 * 1024


def _response(data: object, *, code: int = 200, message: str = "success") -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=jsonable_encoder({"code": code, "message": message, "data": data}),
    )


def _translate_repository_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DocumentNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, DocumentConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status_code=500, detail="知识库操作失败")


@router.post("/knowledge/documents", status_code=status.HTTP_201_CREATED)
async def create_document(payload: KnowledgeDocumentCreate):
    try:
        document = await run_in_threadpool(
            knowledge_repository.create_document,
            payload.title,
            payload.source_path,
            payload.content,
        )
        return _response(
            {**document, "index_status": "queued"},
            code=status.HTTP_201_CREATED,
        )
    except (DocumentNotFoundError, DocumentConflictError) as exc:
        raise _translate_repository_error(exc) from exc


@router.get("/knowledge/documents")
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    data = await run_in_threadpool(knowledge_repository.list_documents, limit, offset)
    return _response(data)


@router.get("/knowledge/documents/{document_id}")
async def get_document(document_id: UUID):
    try:
        document = await run_in_threadpool(knowledge_repository.get_document, str(document_id))
        return _response(document)
    except (DocumentNotFoundError, DocumentConflictError) as exc:
        raise _translate_repository_error(exc) from exc


@router.put("/knowledge/documents/{document_id}")
async def update_document(document_id: UUID, payload: KnowledgeDocumentUpdate):
    try:
        identifier = str(document_id)
        before = await run_in_threadpool(knowledge_repository.get_document, identifier)
        document = await run_in_threadpool(
            knowledge_repository.update_document,
            identifier,
            payload.expected_version,
            payload.title,
            payload.source_path,
            payload.content,
        )
        changed = int(document["version"]) != int(before["version"])
        return _response({**document, "index_status": "queued" if changed else "unchanged"})
    except (DocumentNotFoundError, DocumentConflictError) as exc:
        raise _translate_repository_error(exc) from exc


@router.delete("/knowledge/documents/{document_id}")
async def delete_document(document_id: UUID, expected_version: int = Query(gt=0)):
    try:
        document = await run_in_threadpool(
            knowledge_repository.delete_document, str(document_id), expected_version,
        )
        return _response({"document_id": document["public_id"], "index_status": "queued"})
    except (DocumentNotFoundError, DocumentConflictError) as exc:
        raise _translate_repository_error(exc) from exc


@router.post("/knowledge/documents/{document_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_document(document_id: UUID):
    try:
        document = await run_in_threadpool(
            knowledge_repository.enqueue_repair, str(document_id), True,
        )
        return _response(
            {"document_id": document["public_id"], "index_status": "queued"},
            code=status.HTTP_202_ACCEPTED,
        )
    except (DocumentNotFoundError, DocumentConflictError) as exc:
        raise _translate_repository_error(exc) from exc


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """兼容文件上传；内容直接写入 PostgreSQL，不创建本地副本。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    safe_filename = _sanitize_filename(file.filename)
    if _get_file_extension(safe_filename) not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 UTF-8 的 .txt、.md、.markdown 文件")
    raw = await file.read(MAX_FILE_SIZE + 1)
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过 10MB 限制")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="文件必须使用 UTF-8 编码") from exc
    try:
        document, index_status = await run_in_threadpool(
            knowledge_repository.upsert_uploaded_document,
            safe_filename,
            safe_filename,
            content,
        )
        return _response({
            "filename": safe_filename,
            "document_id": document["public_id"],
            "size": len(raw),
            "index_status": index_status,
        })
    except (DocumentNotFoundError, DocumentConflictError) as exc:
        raise _translate_repository_error(exc) from exc


@router.get("/index/tasks")
async def list_index_tasks(limit: int = Query(default=100, ge=1, le=500)):
    tasks = await run_in_threadpool(knowledge_repository.list_jobs, limit)
    return _response(tasks)


@router.post("/index/retry/{task_id}")
async def retry_index_task(task_id: UUID):
    try:
        task = await run_in_threadpool(knowledge_repository.retry_job, str(task_id))
        return _response(task)
    except (DocumentNotFoundError, DocumentConflictError) as exc:
        raise _translate_repository_error(exc) from exc


def _get_file_extension(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return parts[1].lower() if len(parts) == 2 else ""


def _sanitize_filename(filename: str) -> str:
    sanitized = filename.strip().replace(" ", "_")
    for character in '\\/:*?"<>|':
        sanitized = sanitized.replace(character, "_")
    return sanitized
