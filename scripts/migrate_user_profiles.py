"""Bulk-import legacy data/profiles JSON files into the configured L3 backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from travel_agent.settings import get_settings
from travel_agent.storage.user_memory import (
    JsonUserMemoryRepository,
    UserMemoryService,
    get_user_memory_service,
)


def _build_service(profile_dir: Path | None) -> tuple[UserMemoryService, bool]:
    settings = get_settings()
    if profile_dir is None:
        return get_user_memory_service(settings.memory), False
    if settings.memory.backend == "postgres":
        if not settings.memory.database_url:
            raise RuntimeError("TRAVEL_AGENT_DATABASE_URL is required for postgres memory")
        from travel_agent.storage.postgres_user_memory import PostgresUserMemoryRepository

        repository = PostgresUserMemoryRepository(
            settings.memory.database_url,
            legacy_profile_dir=profile_dir,
        )
    else:
        repository = JsonUserMemoryRepository(profile_dir)
    return UserMemoryService(repository), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    profile_dir = args.profile_dir or settings.memory.profile_dir
    service, owns_service = _build_service(args.profile_dir)
    imported = skipped = failed = 0
    try:
        for path in sorted(profile_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                user_id = str(data.get("user_id") or path.stem)
                memory = service.repository.load_memory(user_id)
                if memory.preference_events or memory.recent_trips:
                    imported += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"failed {path.name}: {exc}")
    finally:
        if owns_service:
            service.close()
    print(f"imported={imported} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
