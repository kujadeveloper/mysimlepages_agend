"""SQLite tabanlı kalıcı iş kuyruğu.

Üretim saniyeler-dakikalar sürer; HTTP isteği beklemez, iş kuyruğa alınır ve durum sorgulanır. Ek altyapı (Redis/Celery)
gerektirmez. MSP_WORKERS (varsayılan 1) kadar iş parçacığı çalışır: tek GPU'da aynı anda tek model çağrısı bellek
taşmasını önler. Servis yeniden başlarsa yarım kalan işler kuyruğa geri döner.
"""
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,              -- queued | running | done | failed
    brief TEXT NOT NULL,
    user_ref TEXT,
    model_version TEXT,
    stage TEXT, progress REAL DEFAULT 0,
    html TEXT, plan TEXT, metrics TEXT, trace TEXT, error TEXT,
    rating INTEGER, comment TEXT,
    created_at TEXT NOT NULL, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
"""
# Sonradan eklenen sütunlar (mevcut veritabanlarına ALTER TABLE ile eklenir)
MIGRATIONS = {
    "parent_id": "TEXT",      # düzeltme ise üzerine çalışılan tasarım
    "instruction": "TEXT",    # kullanıcının düzeltme talimatı
    "sections": "TEXT",       # {bölüm_id: html}: düzeltmede değişmeyen bölümler aynen korunur
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, path: Path):
        self._path = path
        self._local = threading.local()
        self._claim_lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            have = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
            for col, typ in MIGRATIONS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            c.execute("CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent_id)")
            c.execute("UPDATE jobs SET status='queued', stage=NULL, progress=0 WHERE status='running'")

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path, timeout=30, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return c

    def create(self, brief: str, user_ref: str | None, parent_id: str | None = None,
               instruction: str | None = None) -> str:
        job_id = uuid.uuid4().hex
        self._conn().execute(
            "INSERT INTO jobs(id, status, brief, user_ref, parent_id, instruction, created_at) VALUES (?,?,?,?,?,?,?)",
            (job_id, "queued", brief, user_ref, parent_id, instruction, _now()))
        return job_id

    def get(self, job_id: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("plan", "metrics", "trace", "sections"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def list_for(self, user_ref: str, limit: int = 20) -> list[dict]:
        rows = self._conn().execute(
            "SELECT id, status, substr(brief, 1, 200) AS brief, parent_id, substr(instruction, 1, 200) AS instruction, "
            "rating, created_at, finished_at FROM jobs WHERE user_ref=? ORDER BY created_at DESC LIMIT ?",
            (user_ref, limit)).fetchall()
        return [dict(r) for r in rows]

    def count_since(self, user_ref: str, since_iso: str) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM jobs WHERE user_ref=? AND created_at >= ?",
                                    (user_ref, since_iso)).fetchone()[0]

    def queue_position(self, job_id: str) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) FROM jobs WHERE status='queued' AND created_at <= (SELECT created_at FROM jobs WHERE id=?)",
            (job_id,)).fetchone()
        return row[0]

    def claim(self) -> dict | None:
        with self._claim_lock:
            c = self._conn()
            row = c.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            c.execute("UPDATE jobs SET status='running', stage='start' WHERE id=?", (row["id"],))
            return self.get(row["id"])

    def progress(self, job_id: str, stage: str, value: float):
        self._conn().execute("UPDATE jobs SET stage=?, progress=? WHERE id=?", (stage, round(value, 3), job_id))

    def finish(self, job_id: str, *, html, plan, metrics, trace, model_version, sections=None):
        self._conn().execute(
            "UPDATE jobs SET status='done', stage='done', progress=1, html=?, plan=?, metrics=?, trace=?, "
            "model_version=?, sections=?, finished_at=? WHERE id=?",
            (html, json.dumps(plan, ensure_ascii=False), json.dumps(metrics), json.dumps(trace, ensure_ascii=False),
             model_version, json.dumps(sections, ensure_ascii=False) if sections is not None else None, _now(), job_id))

    def fail(self, job_id: str, error: str):
        self._conn().execute("UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                             (error[:1000], _now(), job_id))

    def rate(self, job_id: str, rating: int, comment: str | None):
        self._conn().execute("UPDATE jobs SET rating=?, comment=? WHERE id=?", (rating, comment, job_id))


class WorkerPool:
    def __init__(self, store: JobStore, handler, workers: int = 1):
        self._store = store
        self._handler = handler
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._loop, name=f"agent-worker-{i}", daemon=True)
                         for i in range(max(1, workers))]

    def start(self):
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def notify(self):
        self._wake.set()

    def _loop(self):
        while not self._stop.is_set():
            job = self._store.claim()
            if job is None:
                self._wake.wait(timeout=5)
                self._wake.clear()
                continue
            try:
                self._handler(job)
            except Exception as e:  # iş başarısız olur, işçi çalışmaya devam eder
                log.exception("İş başarısız: %s", job["id"])
                self._store.fail(job["id"], f"{type(e).__name__}: {e}")
