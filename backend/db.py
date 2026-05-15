import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from config import SQLITE_DB_PATH


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                subject_id TEXT NOT NULL,
                status TEXT NOT NULL,
                prediction_label TEXT,
                confidence REAL,
                result_json TEXT,
                report_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analyses_subject_id
            ON analyses(subject_id, completed_at DESC)
            """
        )


def save_analysis(
    job_id: str,
    subject_id: str,
    status: str,
    result: dict | None = None,
    report: dict | None = None,
    error: str | None = None,
) -> None:
    prediction_label = None
    confidence = None
    if result and result.get("prediction"):
        prediction_label = result["prediction"].get("label")
        confidence = result["prediction"].get("confidence")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO analyses (
                job_id,
                subject_id,
                status,
                prediction_label,
                confidence,
                result_json,
                report_json,
                error,
                completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(job_id) DO UPDATE SET
                subject_id = excluded.subject_id,
                status = excluded.status,
                prediction_label = excluded.prediction_label,
                confidence = excluded.confidence,
                result_json = excluded.result_json,
                report_json = excluded.report_json,
                error = excluded.error,
                completed_at = CURRENT_TIMESTAMP
            """,
            (
                job_id,
                subject_id,
                status,
                prediction_label,
                confidence,
                json.dumps(result) if result else None,
                json.dumps(report) if report else None,
                error,
            ),
        )


def get_subject_history(subject_id: str, limit: int = 20) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                job_id,
                subject_id,
                status,
                prediction_label,
                confidence,
                result_json,
                report_json,
                error,
                created_at,
                completed_at
            FROM analyses
            WHERE subject_id = ?
            ORDER BY completed_at DESC, id DESC
            LIMIT ?
            """,
            (subject_id, limit),
        ).fetchall()

    history = []
    for row in rows:
        history.append(
            {
                "job_id": row["job_id"],
                "subject_id": row["subject_id"],
                "status": row["status"],
                "prediction_label": row["prediction_label"],
                "confidence": row["confidence"],
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "report": json.loads(row["report_json"]) if row["report_json"] else None,
                "error": row["error"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            }
        )
    return history
