"""Create stable preference evidence and recent trip tables."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260804_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_memories",
        sa.Column("user_id", sa.String(255), primary_key=True),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_table(
        "user_preference_events",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(255), sa.ForeignKey("user_memories.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("polarity", sa.String(16), nullable=False),
        sa.Column("observed_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_preference_events_user_time", "user_preference_events", ["user_id", "observed_at"])
    op.create_table(
        "user_preference_evidence",
        sa.Column("user_id", sa.String(255), sa.ForeignKey("user_memories.user_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("category", sa.String(64), primary_key=True),
        sa.Column("value", sa.Text(), primary_key=True),
        sa.Column("positive_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("negative_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_polarity", sa.String(16), nullable=False),
        sa.Column("last_seen_at", sa.Float(), nullable=False),
    )
    op.create_table(
        "user_recent_trips",
        sa.Column("user_id", sa.String(255), sa.ForeignKey("user_memories.user_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("session_id", sa.String(255), primary_key=True),
        sa.Column("destination", sa.Text()),
        sa.Column("days", sa.Integer()),
        sa.Column("start_date", sa.String(64)),
        sa.Column("companions", sa.Text()),
        sa.Column("budget_level", sa.String(32)),
        sa.Column("interests", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("pace", sa.String(32), nullable=False, server_default="standard"),
        sa.Column("hotel_area", sa.Text()),
        sa.Column("food_preference", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("must_visit", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("avoid", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("transport_mode", sa.String(32), nullable=False, server_default="public_transport"),
        sa.Column("itinerary_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("poi_names", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("critic_passed", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_recent_trips_user_time", "user_recent_trips", ["user_id", "updated_at"])


def downgrade() -> None:
    op.drop_table("user_recent_trips")
    op.drop_table("user_preference_evidence")
    op.drop_table("user_preference_events")
    op.drop_table("user_memories")
