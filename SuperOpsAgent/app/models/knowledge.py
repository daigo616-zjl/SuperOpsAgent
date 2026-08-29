"""PostgreSQL 知识文档 API 模型。"""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

DocumentTitle = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
SourcePath = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class KnowledgeDocumentCreate(BaseModel):
    title: DocumentTitle
    source_path: SourcePath
    content: str


class KnowledgeDocumentUpdate(KnowledgeDocumentCreate):
    expected_version: int = Field(gt=0)
