"""PostgreSQL 连接池与 schema 初始化。"""

from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import config

# 有序迁移清单：新增 migration 必须登记于此，否则不会被执行
MIGRATIONS: list[str] = [
    "001_postgres_knowledge.sql",
    "002_aiops_evidence.sql",
]


class PostgresManager:
    def __init__(self) -> None:
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("PostgreSQL 尚未连接")
        return self._engine

    def connect(self) -> None:
        if self._engine is not None:
            return
        self._engine = create_engine(
            config.database_url,
            pool_pre_ping=True,
            pool_size=config.database_pool_size,
            max_overflow=config.database_max_overflow,
            pool_timeout=config.database_pool_timeout,
        )
        try:
            with self._engine.connect() as connection:
                connection.execute(text("select 1"))
            self._initialize_schema()
        except Exception:
            self.close()
            raise
        logger.info("PostgreSQL 权威知识库连接成功")

    def _initialize_schema(self) -> None:
        migrations_dir = Path(__file__).resolve().parents[2] / "migrations"
        with self.engine.begin() as connection:
            for migration in MIGRATIONS:
                schema_sql = (migrations_dir / migration).read_text(encoding="utf-8")
                connection.exec_driver_sql(schema_sql)

    def health_check(self) -> bool:
        try:
            with self.engine.connect() as connection:
                return connection.execute(text("select 1")).scalar_one() == 1
        except Exception:
            return False

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


postgres_manager = PostgresManager()
