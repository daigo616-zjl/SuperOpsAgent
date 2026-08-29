"""将旧 Markdown/TXT 目录一次性导入 PostgreSQL；服务启动不会调用本脚本。"""

import argparse
from pathlib import Path

from app.core.postgres import postgres_manager
from app.services.knowledge_repository import knowledge_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Import legacy knowledge files into PostgreSQL")
    parser.add_argument("--directory", default="aiops-docs")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    queued = unchanged = failed = 0
    postgres_manager.connect()
    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".markdown"}:
                continue
            relative_path = path.relative_to(root).as_posix()
            try:
                _, result = knowledge_repository.upsert_uploaded_document(
                    path.name, relative_path, path.read_text(encoding="utf-8"),
                )
                if result == "unchanged":
                    unchanged += 1
                else:
                    queued += 1
            except Exception as exc:
                failed += 1
                print(f"FAILED {relative_path}: {exc}")
    finally:
        postgres_manager.close()
    print(f"queued={queued} unchanged={unchanged} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
