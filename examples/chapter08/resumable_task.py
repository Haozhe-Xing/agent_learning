"""Resume a staged task with a SQLite checkpoint and a local, audited effect."""

import argparse
import sqlite3
from contextlib import closing

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    proposal TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    job_id TEXT PRIMARY KEY,
    content TEXT NOT NULL
);
"""


def resume(db: sqlite3.Connection, job_id: str, stop_after_apply: bool) -> None:
    # In this demo the effect and its checkpoint share one transaction.
    with db:
        apply = db.execute(
            "UPDATE jobs SET stage = 'applied' WHERE id = ? AND stage = 'approved'",
            (job_id,),
        )
        if apply.rowcount:
            db.execute(
                "INSERT INTO artifacts SELECT id, proposal FROM jobs WHERE id = ?",
                (job_id,),
            )

    if stop_after_apply:
        return

    # A resumed process can verify the recorded output without applying it again.
    with db:
        db.execute(
            """
            UPDATE jobs SET stage = 'verified'
            WHERE id = ? AND stage = 'applied'
            AND EXISTS (
                SELECT 1 FROM artifacts
                WHERE job_id = jobs.id AND content = jobs.proposal
            )
            """,
            (job_id,),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "approve", "run", "status"])
    parser.add_argument("--db", required=True, help="Path to the SQLite database")
    parser.add_argument("--job", default="report-001")
    parser.add_argument("--stop-after-apply", action="store_true")
    args = parser.parse_args()

    with closing(sqlite3.connect(args.db)) as db:
        db.executescript(SCHEMA)
        with db:
            if args.command == "start":
                db.execute(
                    "INSERT OR IGNORE INTO jobs VALUES (?, 'proposed', ?)",
                    (args.job, "Publish the reviewed example report."),
                )
            elif args.command == "approve":
                db.execute(
                    "UPDATE jobs SET stage = 'approved' WHERE id = ? AND stage = 'proposed'",
                    (args.job,),
                )

        if args.command == "run":
            resume(db, args.job, args.stop_after_apply)

        job = db.execute("SELECT stage FROM jobs WHERE id = ?", (args.job,)).fetchone()
        if job is None:
            parser.error(f"Unknown job: {args.job}")
        print(f"{args.job}: {job[0]}")


if __name__ == "__main__":
    main()
