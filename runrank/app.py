from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sqlite3
import math
import os
import re
import uuid
import urllib.parse
import urllib.request

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get(
    "RUNRANK_DB_PATH",
    str(BASE_DIR / "runrank.db")
))

KST = timezone(timedelta(hours=9))

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def load_tracks() -> List[Dict[str, Any]]:
    path = BASE_DIR / "tracks.json"
    if not path.exists():
        raise FileNotFoundError("tracks.json 파일이 없습니다. tracks.json 셀을 먼저 실행하세요.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

TRACKS = load_tracks()
TRACK_IDS = {t["id"] for t in TRACKS}
TRACK_BY_ID = {t["id"]: t for t in TRACKS}

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

    # votes
    cur.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            user TEXT NOT NULL,
            ts_epoch INTEGER NOT NULL,
            vote_day_kst TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_vote_per_day
        ON submissions(track_id, user, vote_day_kst)
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_track ON submissions(track_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_ts ON submissions(ts_epoch)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_day ON submissions(vote_day_kst)")

    # ✅ cover cache
    cur.execute("""
        CREATE TABLE IF NOT EXISTS track_covers (
            track_id INTEGER PRIMARY KEY,
            cover_url TEXT NOT NULL,
            source TEXT NOT NULL,
            updated_ts_epoch INTEGER NOT NULL
        )
    """)

    # ✅ running records (photo optional)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            ts_epoch INTEGER NOT NULL,
            day_kst TEXT NOT NULL,
            date_label TEXT NOT NULL,
            distance_km REAL NOT NULL,
            duration_sec INTEGER NOT NULL,
            track_id INTEGER,
            photo_url TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts_epoch)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_user ON runs(user)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_day ON runs(day_kst)")

    conn.commit()
    conn.close()

init_db()

class SubmitBody(BaseModel):
    track_id: int
    user: Optional[str] = None

class CoverResolveBody(BaseModel):
    track_id: int

def now_utc_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def kst_day_str_from_epoch(ts_epoch: int) -> str:
    dt_kst = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).astimezone(KST)
    return dt_kst.strftime("%Y-%m-%d")

def parse_iso_to_epoch(s: str) -> int:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def get_cover_from_cache(track_id: int) -> Optional[str]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT cover_url FROM track_covers WHERE track_id=?", (track_id,))
    row = cur.fetchone()
    conn.close()
    return row["cover_url"] if row else None

def upsert_cover(track_id: int, cover_url: str, source: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO track_covers(track_id, cover_url, source, updated_ts_epoch)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
          cover_url=excluded.cover_url,
          source=excluded.source,
          updated_ts_epoch=excluded.updated_ts_epoch
    """, (track_id, cover_url, source, now_utc_epoch()))
    conn.commit()
    conn.close()

def itunes_search_cover(title: str, artist: str) -> Optional[str]:
    """
    iTunes Search API로 아트워크 URL(100x100)을 가져옴.
    - 첫 결과만 사용
    - 실패 시 None
    """
    q = f"{title} {artist}".strip()
    term = urllib.parse.quote(q)
    url = f"https://itunes.apple.com/search?term={term}&entity=song&limit=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RunRank/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        obj = json.loads(data)
        results = obj.get("results") or []
        if not results:
            return None
        r0 = results[0]
        # artworkUrl100 is common. Sometimes artworkUrl60 etc.
        cover = r0.get("artworkUrl100") or r0.get("artworkUrl60")
        if not cover:
            return None
        return str(cover)
    except Exception:
        return None

def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:120] if name else "file"

def calc_pace_str(distance_km: float, duration_sec: int) -> str:
    if distance_km <= 0:
        return "-"
    sec_per_km = duration_sec / distance_km
    m = int(sec_per_km // 60)
    s = int(round(sec_per_km - m * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{str(s).zfill(2)}/km"

@app.get("/api/health")
def health():
    return {"ok": True, "tracks_count": len(TRACKS)}

@app.get("/api/tracks")
def get_tracks(resolve_missing: int = 0, resolve_limit: int = 8):
    """
    tracks.json + cover cache를 합쳐서 반환.
    resolve_missing=1 이면 cover_url 없는 트랙 중 일부를 iTunes로 자동 채움(캐시 저장).
    """
    out = []
    missing = []
    for t in TRACKS:
        tt = dict(t)
        if not tt.get("cover_url"):
            cached = get_cover_from_cache(tt["id"])
            if cached:
                tt["cover_url"] = cached
            else:
                missing.append(tt)
        out.append(tt)

    if resolve_missing == 1 and missing:
        # 너무 느려지지 않게 limit만 처리
        n = max(0, min(int(resolve_limit), 20))
        for tt in missing[:n]:
            cover = itunes_search_cover(tt.get("title", ""), tt.get("artist", ""))
            if cover:
                upsert_cover(tt["id"], cover, "itunes")
                # out에도 반영
                for o in out:
                    if o["id"] == tt["id"]:
                        o["cover_url"] = cover
                        break

    return out

@app.post("/api/cover/resolve")
def resolve_cover(body: CoverResolveBody):
    """
    특정 track_id의 커버를 iTunes에서 찾아 캐시에 저장.
    """
    tid = body.track_id
    if tid not in TRACK_IDS:
        return {"ok": False, "error": "Invalid track id"}

    cached = get_cover_from_cache(tid)
    if cached:
        return {"ok": True, "track_id": tid, "cover_url": cached, "cached": True}

    t = TRACK_BY_ID.get(tid)
    cover = itunes_search_cover(t.get("title", ""), t.get("artist", "")) if t else None
    if not cover:
        return {"ok": False, "error": "cover not found"}
    upsert_cover(tid, cover, "itunes")
    return {"ok": True, "track_id": tid, "cover_url": cover, "cached": False}

@app.post("/api/covers/resolve_all")
def resolve_all_covers(limit: int = 10):
    """
    cover_url 없는 트랙들 중 limit개를 iTunes에서 찾아 캐시에 저장.
    """
    n = max(0, min(int(limit), 30))
    updated = 0
    tried = 0
    for t in TRACKS:
        if updated >= n:
            break
        tid = t["id"]
        if t.get("cover_url"):
            continue
        if get_cover_from_cache(tid):
            continue
        tried += 1
        cover = itunes_search_cover(t.get("title", ""), t.get("artist", ""))
        if cover:
            upsert_cover(tid, cover, "itunes")
            updated += 1
    return {"ok": True, "tried": tried, "updated": updated, "limit": n}

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

    cur.execute(
        "SELECT 1 FROM submissions WHERE track_id=? AND user=? AND vote_day_kst=? LIMIT 1",
        (body.track_id, user, vote_day_kst),
    )
    if cur.fetchone():
        conn.close()
        return {"ok": False, "error": f"already voted today (KST {vote_day_kst})"}

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

    # cover merge
    out = []
    for t in ranked:
        tid = t["id"]
        cover = t.get("cover_url") or get_cover_from_cache(tid)
        out.append({"id": tid, "title": t["title"], "artist": t["artist"], "votes": counts.get(tid, 0), "cover_url": cover})
    return out

@app.get("/api/hot_ranking")
def hot_ranking(tau_hours: float = 24.0):
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

    out = []
    for t in ranked:
        tid = t["id"]
        cover = t.get("cover_url") or get_cover_from_cache(tid)
        out.append({
            "id": tid,
            "title": t["title"],
            "artist": t["artist"],
            "score": round(scores.get(tid, 0.0), 4),
            "cover_url": cover
        })
    return out

# ----------------------
# ✅ Runs API (photo upload)
# ----------------------

@app.post("/api/runs")
async def create_run(
    user: str = Form(...),
    distance_km: float = Form(...),
    duration_sec: int = Form(...),
    date_label: str = Form(...),  # 예: 2026-02-05 or "오늘"
    track_id: Optional[int] = Form(None),
    photo: Optional[UploadFile] = File(None),
):
    user = (user or "").strip()
    if not user:
        return {"ok": False, "error": "user is required"}

    if distance_km <= 0:
        return {"ok": False, "error": "distance_km must be > 0"}
    if duration_sec <= 0:
        return {"ok": False, "error": "duration_sec must be > 0"}

    if track_id is not None and track_id not in TRACK_IDS:
        return {"ok": False, "error": "Invalid track id"}

    ts_epoch = now_utc_epoch()
    day_kst = kst_day_str_from_epoch(ts_epoch)

    photo_url = None
    if photo is not None:
        orig = sanitize_filename(photo.filename or "photo")
        ext = Path(orig).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic"]:
            # jpg/png가 대부분. 그래도 저장은 해주되 확장자는 안전하게 .jpg로 강제
            ext = ".jpg"
        fname = f"run_{ts_epoch}_{uuid.uuid4().hex[:8]}{ext}"
        out_path = UPLOAD_DIR / fname
        content = await photo.read()
        with open(out_path, "wb") as f:
            f.write(content)
        photo_url = f"/uploads/{fname}"

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs(user, ts_epoch, day_kst, date_label, distance_km, duration_sec, track_id, photo_url)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
    """, (user, ts_epoch, day_kst, date_label, float(distance_km), int(duration_sec), track_id, photo_url))
    conn.commit()
    run_id = cur.lastrowid
    conn.close()

    pace = calc_pace_str(float(distance_km), int(duration_sec))
    return {"ok": True, "id": run_id, "day_kst": day_kst, "pace": pace, "photo_url": photo_url}

@app.get("/api/runs")
def list_runs(user: Optional[str] = None, limit: int = 30):
    n = max(1, min(int(limit), 100))
    conn = db()
    cur = conn.cursor()

    if user and user.strip():
        cur.execute("SELECT * FROM runs WHERE user=? ORDER BY ts_epoch DESC LIMIT ?", (user.strip(), n))
    else:
        cur.execute("SELECT * FROM runs ORDER BY ts_epoch DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    conn.close()

    out = []
    for r in rows:
        tid = r["track_id"]
        t = TRACK_BY_ID.get(tid) if tid is not None else None
        out.append({
            "id": r["id"],
            "user": r["user"],
            "ts_epoch": r["ts_epoch"],
            "day_kst": r["day_kst"],
            "date_label": r["date_label"],
            "distance_km": r["distance_km"],
            "duration_sec": r["duration_sec"],
            "pace": calc_pace_str(r["distance_km"], r["duration_sec"]),
            "track": ({"id": t["id"], "title": t["title"], "artist": t["artist"], "cover_url": (t.get("cover_url") or get_cover_from_cache(t["id"]))} if t else None),
            "photo_url": r["photo_url"]
        })
    return out

# ✅ static mount (index.html 포함)
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
