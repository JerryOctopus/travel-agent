from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os

import pytest


@pytest.mark.postgres
def test_postgres_repository_idempotency_migration_and_cascade(monkeypatch, tmp_path):
    if os.environ.get("RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set RUN_POSTGRES_TESTS=1 to run Docker PostgreSQL integration")
    pytest.importorskip("sqlalchemy")
    postgres = pytest.importorskip("testcontainers.postgres")
    from alembic import command
    from alembic.config import Config

    from travel_agent.storage.postgres_user_memory import PostgresUserMemoryRepository
    from travel_agent.storage.user_memory import PreferenceObservation, UserMemoryService

    with postgres.PostgresContainer("postgres:16-alpine") as container:
        url = container.get_connection_url().replace("postgresql+psycopg2", "postgresql+psycopg")
        monkeypatch.setenv("TRAVEL_AGENT_DATABASE_URL", url)
        command.upgrade(Config("alembic.ini"), "head")
        (tmp_path / "legacy-user.json").write_text(
            json.dumps(
                {
                    "user_id": "legacy-user",
                    "profile": {
                        "destination": "杭州",
                        "days": 3,
                        "interests": ["food"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        repo = PostgresUserMemoryRepository(url, legacy_profile_dir=tmp_path)
        observation = [PreferenceObservation("interests", "food")]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(
                executor.map(
                    lambda _: repo.record_preference_events("u1", "s1", 1, observation),
                    range(4),
                )
            )
        assert repo.load_memory("u1").preference_evidence[0].positive_count == 1

        with ThreadPoolExecutor(max_workers=2) as executor:
            migrated = list(executor.map(lambda _: repo.load_memory("legacy-user"), range(2)))
        assert all(item.preference_evidence[0].positive_count == 1 for item in migrated)
        assert len(repo.list_recent_trips("legacy-user")) == 1

        repo.delete_user_memory("u1")
        assert UserMemoryService(repo).load_stable_profile("u1").interests == []
        repo.close()
