from __future__ import annotations

import re
from pathlib import Path

from travel_agent.schemas import KnowledgeChunk

DEFAULT_GUIDE_DIR = Path(__file__).resolve().parents[2] / "data" / "guides"


CITY_TO_FILE = {
    "北京": "beijing.md",
    "杭州": "hangzhou.md",
    "上海": "shanghai.md",
    "成都": "chengdu.md",
    "西安": "xian.md",
}


def retrieve_destination_knowledge(
    city: str,
    query: str,
    guide_dir: Path | str = DEFAULT_GUIDE_DIR,
    top_k: int = 2,
) -> list[KnowledgeChunk]:
    filename = CITY_TO_FILE.get(city)
    if not filename:
        return []
    path = Path(guide_dir) / filename
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    chunks = _split_markdown(text)
    query_tokens = _tokenize(query)
    scored = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = _tokenize(chunk)
        score = _overlap_score(query_tokens, chunk_tokens)
        scored.append(
            KnowledgeChunk(
                chunk_id=f"{city}_{index}",
                city=city,
                text=chunk,
                score=score,
                source=str(path),
            )
        )
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def _split_markdown(text: str) -> list[str]:
    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", text)
        if chunk.strip() and not chunk.strip().startswith("#")
    ]
    return chunks


def _tokenize(text: str) -> set[str]:
    lower = text.lower()
    tokens = set(re.findall(r"[a-zA-Z_]+", lower))
    chinese_keywords = [
        "历史",
        "文化",
        "美食",
        "自然",
        "亲子",
        "情侣",
        "雨天",
        "博物馆",
        "夜景",
        "购物",
        "轻松",
        "城市漫步",
    ]
    tokens.update(keyword for keyword in chinese_keywords if keyword in text)
    return tokens


def _overlap_score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return round(len(query_tokens & chunk_tokens) / len(query_tokens), 4)
