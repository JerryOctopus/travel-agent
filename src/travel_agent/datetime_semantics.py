"""Shared, auditable datetime normalization for deterministic contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
import re
from typing import Any, Mapping


_BARE_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?$")


@dataclass(frozen=True)
class DateTimeNormalization:
    raw_value: Any
    canonical_value: str | None
    comparable: bool
    reason: str | None = None
    resolution_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DateTimeComparison:
    equivalent: bool | None
    expected: DateTimeNormalization
    actual: DateTimeNormalization
    reason_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "expected": self.expected.to_dict(),
            "actual": self.actual.to_dict(),
            "reason_code": self.reason_code,
        }


def normalize_datetime_value(
    value: Any,
    *,
    context: Mapping[str, Any] | None = None,
    timezone_hint: tzinfo | None = None,
    prefer_trip_end: bool = False,
) -> DateTimeNormalization:
    """Normalize a timestamp or bare clock without inventing date/timezone context."""
    raw = value
    text = str(value or "").strip()
    if not text:
        return DateTimeNormalization(raw, None, False, "datetime_value_missing")

    parsed = _parse_datetime(text)
    if parsed is not None:
        zone = parsed.tzinfo or timezone_hint or _context_timezone(context)
        if zone is None:
            return DateTimeNormalization(
                raw, None, False, "datetime_timezone_ambiguous", "full_timestamp"
            )
        aware = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=zone)
        canonical = aware.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        return DateTimeNormalization(
            raw,
            canonical,
            True,
            resolution_source=(
                "full_timestamp" if parsed.tzinfo is not None else "full_local_with_context_timezone"
            ),
        )

    if not _BARE_TIME.fullmatch(text):
        return DateTimeNormalization(raw, None, False, "datetime_format_invalid")
    try:
        clock = time.fromisoformat(text)
    except ValueError:
        return DateTimeNormalization(raw, None, False, "datetime_format_invalid")
    final_date, date_source = _context_date(context, prefer_trip_end=prefer_trip_end)
    if final_date is None:
        return DateTimeNormalization(raw, None, False, "datetime_date_ambiguous")
    zone = timezone_hint or _context_timezone(context)
    if zone is None:
        return DateTimeNormalization(raw, None, False, "datetime_timezone_ambiguous")
    aware = datetime.combine(final_date, clock, tzinfo=zone)
    canonical = aware.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    return DateTimeNormalization(raw, canonical, True, resolution_source=date_source)


def compare_datetime_values(
    expected: Any,
    actual: Any,
    *,
    context: Mapping[str, Any] | None = None,
    prefer_trip_end: bool = False,
) -> DateTimeComparison:
    """Compare datetime semantics; ``None`` means the pair is not safely comparable."""
    expected_parsed = _parse_datetime(str(expected or "").strip())
    actual_parsed = _parse_datetime(str(actual or "").strip())
    timezone_hint = (
        (expected_parsed.tzinfo if expected_parsed is not None else None)
        or (actual_parsed.tzinfo if actual_parsed is not None else None)
    )
    expected_text = str(expected or "").strip()
    actual_text = str(actual or "").strip()
    left = normalize_datetime_value(
        expected,
        context=context,
        timezone_hint=timezone_hint if _BARE_TIME.fullmatch(expected_text) else None,
        prefer_trip_end=prefer_trip_end,
    )
    right = normalize_datetime_value(
        actual,
        context=context,
        timezone_hint=timezone_hint if _BARE_TIME.fullmatch(actual_text) else None,
        prefer_trip_end=prefer_trip_end,
    )
    if not left.comparable or not right.comparable:
        return DateTimeComparison(False if left.comparable != right.comparable else None, left, right, "datetime_not_comparable")
    return DateTimeComparison(
        left.canonical_value == right.canonical_value,
        left,
        right,
        None if left.canonical_value == right.canonical_value else "datetime_semantic_mismatch",
    )


def is_datetime_like(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(_BARE_TIME.fullmatch(text) or _parse_datetime(text) is not None)


def _parse_datetime(text: str) -> datetime | None:
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}"
        r"(?::\d{2}(?:\.\d{1,6})?)?(?:[Zz]|[+-]\d{2}:?\d{2})?",
        text,
    ):
        return None
    try:
        normalized = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
        return datetime.fromisoformat(
            normalized.replace("Z", "+00:00").replace("z", "+00:00")
        )
    except ValueError:
        return None


def _context_date(
    context: Mapping[str, Any] | None, *, prefer_trip_end: bool
) -> tuple[date | None, str | None]:
    values = context or {}
    end = values.get("date_end") or values.get("end_date")
    if end:
        try:
            return date.fromisoformat(str(end)), "date_end"
        except ValueError:
            return None, None
    start = values.get("date_start") or values.get("start_date")
    days = values.get("duration_days") or values.get("days")
    if start and days:
        try:
            return (
                date.fromisoformat(str(start)) + timedelta(days=max(1, int(days)) - 1),
                "date_start_plus_duration",
            )
        except (TypeError, ValueError):
            return None, None
    if start and (not prefer_trip_end or str(days or "1") == "1"):
        try:
            return date.fromisoformat(str(start)), "date_start_one_day"
        except ValueError:
            return None, None
    return None, None


def _context_timezone(context: Mapping[str, Any] | None) -> tzinfo | None:
    values = context or {}
    for key in ("timezone", "tz", "reference_datetime"):
        raw = values.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, tzinfo):
            return raw
        text = str(raw).strip()
        parsed = _parse_datetime(text)
        if parsed is not None and parsed.tzinfo is not None:
            return parsed.tzinfo
        match = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", text)
        if match:
            sign = 1 if match.group(1) == "+" else -1
            return timezone(sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))
    return None
