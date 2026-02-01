#셀2
%%writefile app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sqlite3
import math
import os

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get(
    "RUNRANK_DB_PATH",
    str(BASE_DIR / "runrank.db")
))

KST = timezone(timedelta(hours=9))

def load_tracks():
    path = BASE_DIR / "tracks.json"
    if not path.exists():
        raise FileNotFoundError("tracks.json 파일이 없습니다. tracks.json 셀을 먼저 실행하세요.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

TRACKS = load_tracks()
TRACK_IDS = {t["id"] for t in TRACKS}

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

    # ✅ vote_day_kst: 유저별 "KST 날짜" 기준 1일 1표 제한에 사용
    cur.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            user TEXT NOT NULL,
            ts_epoch INTEGER NOT NULL,
            vote_day_kst TEXT NOT NULL
        )
    """)

    # ✅ "유저-곡-날짜(KST)" 유니크 → 하루 1곡 1표 강제
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_vote_per_day
        ON submissions(track_id, user, vote_day_kst)
    """)

    # 조회 최적화
    cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_track ON submissions(track_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_ts ON submissions(ts_epoch)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_day ON submissions(vote_day_kst)")

    conn.commit()
    conn.close()

init_db()

class SubmitBody(BaseModel):
    track_id: int
    user: Optional[str] = None

def now_utc_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def kst_day_str_from_epoch(ts_epoch: int) -> str:
    dt_kst = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).astimezone(KST)
    return dt_kst.strftime("%Y-%m-%d")

def parse_iso_to_epoch(s: str) -> int:
    """
    ISO 문자열을 epoch로 변환.
    - timezone 포함이면 그대로 사용
    - timezone 없으면 UTC로 간주(안전한 기본값)
    """
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

@app.get("/api/health")
def health():
    return {"ok": True, "tracks_count": len(TRACKS)}

@app.get("/api/tracks")
def get_tracks():
    return TRACKS

@app.post("/api/submit")
def submit_vote(body: SubmitBody):
    if body.track_id not in TRACK_IDS:
        return {"ok": False, "error": "Invalid track id"}

    user = (body.user or "").strip()
    if not user:
        return {"ok": False, "error": "user is required (to prevent duplicate votes per day)"}

    ts_epoch = now_utc_epoch()
    vote_day_kst = kst_day_str_from_epoch(ts_epoch)

    conn = db()
    cur = conn.cursor()

    # ✅ 유저-곡-오늘(KST) 이미 투표했는지 체크
    cur.execute(
        "SELECT 1 FROM submissions WHERE track_id=? AND user=? AND vote_day_kst=? LIMIT 1",
        (body.track_id, user, vote_day_kst),
    )
    if cur.fetchone():
        conn.close()
        return {"ok": False, "error": f"already voted today (KST {vote_day_kst})"}

    # ✅ 저장
    cur.execute(
        "INSERT INTO submissions (track_id, user, ts_epoch, vote_day_kst) VALUES (?, ?, ?, ?)",
        (body.track_id, user, ts_epoch, vote_day_kst),
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) AS c FROM submissions WHERE track_id=?", (body.track_id,))
    current_votes = cur.fetchone()["c"]
    conn.close()

    return {"ok": True, "track_id": body.track_id, "current_votes": current_votes, "vote_day_kst": vote_day_kst}

@app.get("/api/ranking")
def ranking(from_ts: Optional[str] = None, to_ts: Optional[str] = None):
    where = []
    params = []

    if from_ts:
        try:
            where.append("ts_epoch >= ?")
            params.append(parse_iso_to_epoch(from_ts))
        except Exception:
            return {"ok": False, "error": "from_ts must be ISO format like 2026-02-01T00:00:00"}

    if to_ts:
        try:
            where.append("ts_epoch < ?")
            params.append(parse_iso_to_epoch(to_ts))
        except Exception:
            return {"ok": False, "error": "to_ts must be ISO format like 2026-02-08T00:00:00"}

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = db()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT track_id, COUNT(*) AS votes
        FROM submissions
        {where_sql}
        GROUP BY track_id
    """, params)
    rows = cur.fetchall()
    conn.close()

    counts: Dict[int, int] = {r["track_id"]: r["votes"] for r in rows}
    ranked = sorted(TRACKS, key=lambda t: counts.get(t["id"], 0), reverse=True)

    return [
        {"id": t["id"], "title": t["title"], "artist": t["artist"], "votes": counts.get(t["id"], 0)}
        for t in ranked
    ]

@app.get("/api/hot_ranking")
def hot_ranking(tau_hours: float = 24.0):
    """
    🔥 핫 랭킹(최근 투표 가중치)
    score = Σ exp(-(now - vote_time)/tau)
    - tau_hours 작을수록 '최근'에 더 민감
    """
    if tau_hours <= 0:
        return {"ok": False, "error": "tau_hours must be > 0"}

    now_epoch = now_utc_epoch()
    tau = tau_hours * 3600.0

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT track_id, ts_epoch FROM submissions")
    rows = cur.fetchall()
    conn.close()

    scores: Dict[int, float] = {t["id"]: 0.0 for t in TRACKS}

    for r in rows:
        tid = r["track_id"]
        age = max(0, now_epoch - r["ts_epoch"])
        w = math.exp(-age / tau)
        if tid in scores:
            scores[tid] += w

    ranked = sorted(TRACKS, key=lambda t: scores.get(t["id"], 0.0), reverse=True)

    return [
        {
            "id": t["id"],
            "title": t["title"],
            "artist": t["artist"],
            "score": round(scores.get(t["id"], 0.0), 4)
        }
        for t in ranked
    ]

# ✅ static을 BASE_DIR 기준으로 고정
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")