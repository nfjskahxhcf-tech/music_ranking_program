from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import math
import os
import re
import uuid
import urllib.parse
import urllib.request
import csv 
import io
import shutil

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

# Share image (Pillow)
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except Exception:
    Image = ImageDraw = ImageFont = ImageFilter = None

# ----------------------
# DB setup (Postgres if DATABASE_URL is set, else SQLite fallback)
# ----------------------
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:
    psycopg = None
    dict_row = None

import sqlite3

# ----------------------
# App / Paths
# ----------------------
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"  # 남겨둬도 됨

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# SQLite path (fallback only)
DB_PATH = Path(os.environ.get("RUNRANK_DB_PATH", str(BASE_DIR / "runrank.db")))

KST = timezone(timedelta(hours=9))

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------
# Helpers
# ----------------------
def is_postgres() -> bool:
    return bool(DATABASE_URL)


def db():
    """
    Returns a DB connection:
    - Postgres (psycopg) if DATABASE_URL exists
    - SQLite otherwise (local dev)
    """
    if is_postgres():
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is set but psycopg is not installed.")
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ph() -> str:
    """SQL placeholder for a single parameter."""
    return "%s" if is_postgres() else "?"


def make_in_clause(n: int) -> str:
    """Return '(?,?,?)' or '(%s,%s,...)' depending on DB."""
    if n <= 0:
        return "(NULL)"
    return "(" + ",".join([ph()] * n) + ")"


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


def fmt_duration_hms(sec: int) -> str:
    sec = max(0, int(sec or 0))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}:{str(m).zfill(2)}:{str(s).zfill(2)}"
    return f"{m}:{str(s).zfill(2)}"


def _font(path: str, size: int, index: int = 0):
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype(path, size=size, index=index)
    except Exception:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            return ImageFont.load_default()


def _safe_open_local_image(path: Path) -> "Image.Image":
    """Open image and handle common edge cases. Returns PIL Image."""
    img = Image.open(str(path))
    # Some uploads may include EXIF orientation
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def _cover_resize(img: "Image.Image", w: int, h: int) -> "Image.Image":
    """Resize by covering the given box (center crop)."""
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return Image.new("RGB", (w, h), (15, 18, 25))
    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img2 = img.resize((nw, nh), Image.LANCZOS)
    left = max(0, (nw - w) // 2)
    top = max(0, (nh - h) // 2)
    return img2.crop((left, top, left + w, top + h))


def _hex_color_from_int(x: int) -> Tuple[int, int, int]:
    # deterministic pleasant-ish color
    x = int(x) & 0xFFFFFFFF
    r = 40 + (x & 0x7F)
    g = 50 + ((x >> 8) & 0x7F)
    b = 70 + ((x >> 16) & 0x7F)
    return (r, g, b)


def _render_run_share_image(
    *,
    run: Dict[str, Any],
    tracks: List[Dict[str, Any]],
    aspect: str = "story",
    theme: str = "dark",
) -> bytes:
    """Return PNG bytes for a share image (Instagram/Kakao)."""
    if Image is None:
        raise RuntimeError("Pillow is required for share image generation.")

    aspect = (aspect or "story").lower().strip()
    if aspect == "feed":
        W, H = 1080, 1350
    elif aspect == "square":
        W, H = 1080, 1080
    else:
        W, H = 1080, 1920

    # --- Background
    bg = None
    photo_url = run.get("photo_url")
    if photo_url:
        # Only allow local uploads
        try:
            if str(photo_url).startswith("/static/uploads/"):
                p = UPLOAD_DIR / Path(str(photo_url)).name
                if p.exists():
                    im = _safe_open_local_image(p)
                    bg = _cover_resize(im, W, H)
        except Exception:
            bg = None

    if bg is None:
        c = _hex_color_from_int(int(run.get("id", 0)) * 2654435761)
        bg = Image.new("RGB", (W, H), c)

    # Blur a copy for readability, keep a subtle texture
    try:
        bg_blur = bg.filter(ImageFilter.GaussianBlur(radius=14))
        bg = Image.blend(bg_blur, bg, 0.35)
    except Exception:
        pass

    # Dark overlay
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    if theme == "light":
        od.rectangle([0, 0, W, H], fill=(255, 255, 255, 60))
    else:
        od.rectangle([0, 0, W, H], fill=(0, 0, 0, 90))

    # Bottom gradient for text
    grad = Image.new("L", (1, H))
    for y in range(H):
        # stronger at bottom
        a = int(255 * min(1.0, max(0.0, (y - H * 0.45) / (H * 0.55))))
        grad.putpixel((0, y), a)
    grad = grad.resize((W, H))
    grad_rgba = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    grad_rgba.putalpha(grad)
    overlay = Image.alpha_composite(overlay, grad_rgba)

    canvas = bg.convert("RGBA")
    canvas = Image.alpha_composite(canvas, overlay)

    d = ImageDraw.Draw(canvas)

        # --- Fonts (Korean-safe)
    # NOTE: keep these inside the share-image render function (same indentation)
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    local_font_reg = base_dir / "assets" / "fonts" / "NotoSansKR-Regular.ttf"

    def _pick_font_path(candidates):
        for p in candidates:
            try:
                if p and Path(p).exists():
                    return str(p)
            except Exception:
                pass
        return None

    font_reg_path = _pick_font_path([
        str(local_font_reg),  # 1st priority: repo font
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ])

    font_bold_path = _pick_font_path([
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        str(local_font_reg),  # fallback
    ])

    f_brand = _font(font_bold_path, 44)
    f_big = _font(font_bold_path, 120)
    f_mid = _font(font_bold_path, 60)
    f_label = _font(font_reg_path, 34)
    f_small = _font(font_reg_path, 30)
    f_tiny = _font(font_reg_path, 26)

    # --- Content
    date_label = str(run.get("date_label") or "")
    distance_km = float(run.get("distance_km") or 0.0)
    duration_sec = int(run.get("duration_sec") or 0)
    pace = calc_pace_str(distance_km, duration_sec)
    duration = fmt_duration_hms(duration_sec)
    user = str(run.get("user") or "")

    # colors
    white = (245, 248, 255, 255)
    muted = (190, 205, 235, 220)
    accent = (123, 220, 255, 255)

    pad = 72
    top_y = 72

    # Brand
    d.text((pad, top_y), "RS music app", font=f_brand, fill=white)
    d.text((pad, top_y + 54), "RUN", font=f_label, fill=muted)

    # Big distance
    dist_str = f"{distance_km:.2f}".rstrip("0").rstrip(".")
    d.text((pad, top_y + 130), dist_str, font=f_big, fill=white)
    d.text((pad + 10 + int(d.textlength(dist_str, font=f_big)), top_y + 200), "km", font=f_mid, fill=muted)

    # Stats block
    block_y = H - 420 if H >= 1350 else H - 360
    if H == 1080:
        block_y = H - 330
    if H == 1350:
        block_y = H - 360

    # Left column: time
    d.text((pad, block_y), "TIME", font=f_tiny, fill=muted)
    d.text((pad, block_y + 48), duration, font=f_mid, fill=white)

    # Right column: pace
    col2_x = W // 2 + 30
    d.text((col2_x, block_y), "PACE", font=f_tiny, fill=muted)
    d.text((col2_x, block_y + 48), pace, font=f_mid, fill=white)

    # Date/user line
    meta_y = block_y + 150
    meta = date_label.strip()
    if user:
        meta = f"{meta} · {user}" if meta else user
    if meta:
        d.text((pad, meta_y), meta, font=f_small, fill=muted)

    # Music line (first track)
    if tracks:
        t0 = tracks[0]
        music = f"🎵 {t0.get('title','').strip()} — {t0.get('artist','').strip()}".strip()
        # clamp length by rough chars
        if len(music) > 60:
            music = music[:57] + "…"
        d.text((pad, meta_y + 52), music, font=f_tiny, fill=accent)

    # Footer
    d.text((pad, H - 92), "Share to Instagram / Kakao", font=f_tiny, fill=(255, 255, 255, 170))

    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


# ----------------------
# Tracks
# ----------------------
def load_tracks() -> List[Dict[str, Any]]:
    path = BASE_DIR / "tracks.json"
    if not path.exists():
        raise FileNotFoundError("tracks.json 파일이 없습니다. tracks.json 셀을 먼저 실행하세요.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


TRACKS = load_tracks()
TRACK_IDS = {t["id"] for t in TRACKS}
TRACK_BY_ID = {t["id"]: t for t in TRACKS}


# ----------------------
# PWA / Static routes
# ----------------------
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)


@app.get("/index.html", include_in_schema=False)
def index_html():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/service-worker.js", include_in_schema=False)
def service_worker():
    return FileResponse(
        str(STATIC_DIR / "service-worker.js"),
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return FileResponse(
        str(STATIC_DIR / "manifest.webmanifest"),
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(str(STATIC_DIR / "favicon.ico"))


@app.get("/icon-192.png", include_in_schema=False)
def icon_192():
    return FileResponse(str(STATIC_DIR / "icon-192.png"))


@app.get("/icon-512.png", include_in_schema=False)
def icon_512():
    return FileResponse(str(STATIC_DIR / "icon-512.png"))


@app.get("/apple-touch-icon.png", include_in_schema=False)
def apple_touch_icon():
    return FileResponse(str(STATIC_DIR / "apple-touch-icon.png"))


@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon_precomposed():
    return FileResponse(str(STATIC_DIR / "apple-touch-icon-precomposed.png"))


@app.get("/apple-touch-icon-120x120.png", include_in_schema=False)
def apple_touch_icon_120():
    return FileResponse(str(STATIC_DIR / "apple-touch-icon-120x120.png"))


@app.get("/apple-touch-icon-120x120-precomposed.png", include_in_schema=False)
def apple_touch_icon_120_precomposed():
    return FileResponse(str(STATIC_DIR / "apple-touch-icon-120x120-precomposed.png"))


# ----------------------
# DB schema init
# ----------------------
def init_db():
    conn = db()
    cur = conn.cursor()

    if is_postgres():
        # Postgres schema
        cur.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id BIGSERIAL PRIMARY KEY,
                track_id INTEGER NOT NULL,
                user_name TEXT NOT NULL,
                ts_epoch BIGINT NOT NULL,
                vote_day_kst TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_vote_per_day
            ON submissions(track_id, user_name, vote_day_kst)
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_track ON submissions(track_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_ts ON submissions(ts_epoch)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submissions_day ON submissions(vote_day_kst)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS vote_batches (
                id BIGSERIAL PRIMARY KEY,
                user_name TEXT NOT NULL,
                vote_day_kst TEXT NOT NULL,
                ts_epoch BIGINT NOT NULL,
                run_id BIGINT
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_vote_batch_user_day
            ON vote_batches(user_name, vote_day_kst)
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vote_batches_day ON vote_batches(vote_day_kst)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS track_covers (
                track_id INTEGER PRIMARY KEY,
                cover_url TEXT NOT NULL,
                source TEXT NOT NULL,
                updated_ts_epoch BIGINT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id BIGSERIAL PRIMARY KEY,
                user_name TEXT NOT NULL,
                ts_epoch BIGINT NOT NULL,
                day_kst TEXT NOT NULL,
                date_label TEXT NOT NULL,
                distance_km DOUBLE PRECISION NOT NULL,
                duration_sec INTEGER NOT NULL,
                track_id INTEGER,
                photo_url TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts_epoch)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_user ON runs(user_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_day ON runs(day_kst)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS run_tracks (
                run_id BIGINT NOT NULL,
                track_id INTEGER NOT NULL,
                PRIMARY KEY(run_id, track_id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_run_tracks_run ON run_tracks(run_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_run_tracks_track ON run_tracks(track_id)")

    else:
        # SQLite schema (fallback)
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

        cur.execute("""
            CREATE TABLE IF NOT EXISTS vote_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                vote_day_kst TEXT NOT NULL,
                ts_epoch INTEGER NOT NULL,
                run_id INTEGER
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_vote_batch_user_day
            ON vote_batches(user, vote_day_kst)
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vote_batches_day ON vote_batches(vote_day_kst)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS track_covers (
                track_id INTEGER PRIMARY KEY,
                cover_url TEXT NOT NULL,
                source TEXT NOT NULL,
                updated_ts_epoch INTEGER NOT NULL
            )
        """)

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

        cur.execute("""
            CREATE TABLE IF NOT EXISTS run_tracks (
                run_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                PRIMARY KEY(run_id, track_id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_run_tracks_run ON run_tracks(run_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_run_tracks_track ON run_tracks(track_id)")

    conn.commit()
    conn.close()


init_db()


# ----------------------
# Models
# ----------------------
class SubmitBody(BaseModel):
    track_id: Optional[int] = None
    track_ids: Optional[List[int]] = None
    user: Optional[str] = None


class CoverResolveBody(BaseModel):
    track_id: int


# ----------------------
# Covers
# ----------------------
def get_cover_from_cache(track_id: int) -> Optional[str]:
    conn = db()
    cur = conn.cursor()
    if is_postgres():
        cur.execute(f"SELECT cover_url FROM track_covers WHERE track_id={ph()}", (track_id,))
    else:
        cur.execute("SELECT cover_url FROM track_covers WHERE track_id=?", (track_id,))
    row = cur.fetchone()
    conn.close()
    return row["cover_url"] if row else None


def upsert_cover(track_id: int, cover_url: str, source: str):
    conn = db()
    cur = conn.cursor()
    ts = now_utc_epoch()

    if is_postgres():
        cur.execute(f"""
            INSERT INTO track_covers(track_id, cover_url, source, updated_ts_epoch)
            VALUES({ph()}, {ph()}, {ph()}, {ph()})
            ON CONFLICT(track_id) DO UPDATE SET
              cover_url=EXCLUDED.cover_url,
              source=EXCLUDED.source,
              updated_ts_epoch=EXCLUDED.updated_ts_epoch
        """, (track_id, cover_url, source, ts))
    else:
        cur.execute("""
            INSERT INTO track_covers(track_id, cover_url, source, updated_ts_epoch)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET
              cover_url=excluded.cover_url,
              source=excluded.source,
              updated_ts_epoch=excluded.updated_ts_epoch
        """, (track_id, cover_url, source, ts))

    conn.commit()
    conn.close()


def itunes_search_cover(title: str, artist: str) -> Optional[str]:
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
        cover = r0.get("artworkUrl100") or r0.get("artworkUrl60")
        return str(cover) if cover else None
    except Exception:
        return None


# ----------------------
# Basic APIs
# ----------------------
@app.get("/api/health")
def health():
    return {"ok": True, "tracks_count": len(TRACKS), "db": ("postgres" if is_postgres() else "sqlite")}


@app.get("/api/tracks")
def get_tracks(resolve_missing: int = 0, resolve_limit: int = 8):
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
        n = max(0, min(int(resolve_limit), 20))
        for tt in missing[:n]:
            cover = itunes_search_cover(tt.get("title", ""), tt.get("artist", ""))
            if cover:
                upsert_cover(tt["id"], cover, "itunes")
                for o in out:
                    if o["id"] == tt["id"]:
                        o["cover_url"] = cover
                        break
    return out


@app.post("/api/cover/resolve")
def resolve_cover(body: CoverResolveBody):
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


# ----------------------
# Voting
# ----------------------
def _select_vote_batch(cur, user: str, vote_day_kst: str):
    if is_postgres():
        cur.execute(
            f"SELECT id, run_id FROM vote_batches WHERE user_name={ph()} AND vote_day_kst={ph()} LIMIT 1",
            (user, vote_day_kst),
        )
    else:
        cur.execute(
            "SELECT id, run_id FROM vote_batches WHERE user=? AND vote_day_kst=? LIMIT 1",
            (user, vote_day_kst),
        )
    return cur.fetchone()


def _insert_vote_batch(cur, user: str, vote_day_kst: str, ts_epoch: int, run_id: Optional[int]):
    if is_postgres():
        cur.execute(
            f"INSERT INTO vote_batches(user_name, vote_day_kst, ts_epoch, run_id) VALUES({ph()}, {ph()}, {ph()}, {ph()}) RETURNING id",
            (user, vote_day_kst, ts_epoch, run_id),
        )
        return cur.fetchone()["id"]
    else:
        cur.execute(
            "INSERT INTO vote_batches(user, vote_day_kst, ts_epoch, run_id) VALUES(?, ?, ?, ?)",
            (user, vote_day_kst, ts_epoch, run_id),
        )
        return cur.lastrowid


def _insert_submission_ignore(cur, tid: int, user: str, ts_epoch: int, vote_day_kst: str) -> int:
    if is_postgres():
        cur.execute(
            f"INSERT INTO submissions(track_id, user_name, ts_epoch, vote_day_kst) VALUES({ph()}, {ph()}, {ph()}, {ph()}) "
            f"ON CONFLICT(track_id, user_name, vote_day_kst) DO NOTHING",
            (tid, user, ts_epoch, vote_day_kst),
        )
        # psycopg rowcount works for DO NOTHING (0 if conflict, 1 if inserted)
        return 1 if cur.rowcount == 1 else 0
    else:
        cur.execute(
            "INSERT OR IGNORE INTO submissions(track_id, user, ts_epoch, vote_day_kst) VALUES(?, ?, ?, ?)",
            (tid, user, ts_epoch, vote_day_kst),
        )
        return 1 if cur.rowcount == 1 else 0


@app.post("/api/submit")
def submit_vote(body: SubmitBody):
    user = (body.user or "").strip()
    if not user:
        return {"ok": False, "error": "user is required"}

    track_ids: List[int] = []
    if body.track_ids:
        track_ids = [int(x) for x in body.track_ids]
    elif body.track_id is not None:
        track_ids = [int(body.track_id)]

    track_ids = [tid for tid in dict.fromkeys(track_ids) if tid in TRACK_IDS]
    if not track_ids:
        return {"ok": False, "error": "track_ids is required"}

    ts_epoch = now_utc_epoch()
    vote_day_kst = kst_day_str_from_epoch(ts_epoch)

    conn = db()
    cur = conn.cursor()

    already_batch = _select_vote_batch(cur, user, vote_day_kst)
    if already_batch:
        conn.close()
        return {
            "ok": False,
            "error": f"already voted today (KST {vote_day_kst})",
            "vote_day_kst": vote_day_kst,
            "batch_id": already_batch["id"],
            "run_id": already_batch["run_id"],
        }

    batch_id = _insert_vote_batch(cur, user, vote_day_kst, ts_epoch, None)

    inserted = 0
    for tid in track_ids:
        inserted += _insert_submission_ignore(cur, tid, user, ts_epoch, vote_day_kst)

    conn.commit()
    conn.close()

    return {
        "ok": True,
        "vote_day_kst": vote_day_kst,
        "batch_id": batch_id,
        "tracks_voted": track_ids,
        "inserted": inserted,
    }


def try_auto_vote_tracks(user: str, track_ids: List[int], run_id: Optional[int] = None) -> Dict[str, Any]:
    if not track_ids:
        return {"did_vote": False}

    track_ids = [tid for tid in dict.fromkeys([int(x) for x in track_ids]) if tid in TRACK_IDS]
    if not track_ids:
        return {"did_vote": False, "error": "Invalid track ids"}

    ts_epoch = now_utc_epoch()
    vote_day_kst = kst_day_str_from_epoch(ts_epoch)

    conn = db()
    cur = conn.cursor()

    already_batch = _select_vote_batch(cur, user, vote_day_kst)
    if already_batch:
        conn.close()
        return {
            "did_vote": False,
            "already_voted": True,
            "vote_day_kst": vote_day_kst,
            "batch_id": already_batch["id"],
            "run_id": already_batch["run_id"],
        }

    batch_id = _insert_vote_batch(cur, user, vote_day_kst, ts_epoch, run_id)

    inserted = 0
    for tid in track_ids:
        inserted += _insert_submission_ignore(cur, tid, user, ts_epoch, vote_day_kst)

    conn.commit()
    conn.close()

    return {
        "did_vote": True,
        "vote_day_kst": vote_day_kst,
        "batch_id": batch_id,
        "tracks_voted": track_ids,
        "inserted": inserted,
    }


# ----------------------
# Ranking APIs
# ----------------------
@app.get("/api/ranking")
def ranking(from_ts: Optional[str] = None, to_ts: Optional[str] = None):
    where = []
    params = []

    if from_ts:
        try:
            where.append(f"ts_epoch >= {ph()}")
            params.append(parse_iso_to_epoch(from_ts))
        except Exception:
            return {"ok": False, "error": "from_ts must be ISO format like 2026-02-01T00:00:00"}

    if to_ts:
        try:
            where.append(f"ts_epoch < {ph()}")
            params.append(parse_iso_to_epoch(to_ts))
        except Exception:
            return {"ok": False, "error": "to_ts must be ISO format like 2026-02-08T00:00:00"}

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = db()
    cur = conn.cursor()

    if is_postgres():
        cur.execute(f"""
            SELECT track_id, COUNT(*) AS votes
            FROM submissions
            {where_sql}
            GROUP BY track_id
        """, params)
    else:
        # SQLite uses same constructed where_sql already with '?', so ok
        cur.execute(f"""
            SELECT track_id, COUNT(*) AS votes
            FROM submissions
            {where_sql}
            GROUP BY track_id
        """, params)

    rows = cur.fetchall()
    conn.close()

    counts: Dict[int, int] = {r["track_id"]: int(r["votes"]) for r in rows}
    ranked = sorted(TRACKS, key=lambda t: counts.get(t["id"], 0), reverse=True)

    out = []
    for t in ranked:
        tid = t["id"]
        cover = t.get("cover_url") or get_cover_from_cache(tid)
        out.append({
            "id": tid,
            "title": t["title"],
            "artist": t["artist"],
            "votes": counts.get(tid, 0),
            "cover_url": cover
        })
    return out

@app.get("/api/export/ranking.json")
def export_ranking_json(from_ts: Optional[str] = None, to_ts: Optional[str] = None):
    # 기존 ranking() 로직 그대로 재사용
    data = ranking(from_ts=from_ts, to_ts=to_ts)
    # ranking()이 에러 dict 반환할 수도 있으니 그대로 리턴
    return {"ok": True, "data": data} if isinstance(data, list) else data


@app.get("/api/export/ranking.csv")
def export_ranking_csv(from_ts: Optional[str] = None, to_ts: Optional[str] = None):
    data = ranking(from_ts=from_ts, to_ts=to_ts)
    if not isinstance(data, list):
        return data  # 에러 dict 그대로

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["rank", "track_id", "title", "artist", "votes"])

    for i, t in enumerate(data, start=1):
        writer.writerow([i, t.get("id"), t.get("title"), t.get("artist"), t.get("votes")])

    csv_text = output.getvalue()
    filename = f"runrank_ranking_{datetime.now(KST).strftime('%Y%m%d_%H%M')}.csv"
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



@app.get("/api/hot_ranking")
def hot_ranking(tau_hours: float = 24.0):
    if tau_hours <= 0:
        return {"ok": False, "error": "tau_hours must be > 0"}

    now_epoch = now_utc_epoch()
    tau = tau_hours * 3600.0

    conn = db()
    cur = conn.cursor()
    if is_postgres():
        cur.execute("SELECT track_id, ts_epoch FROM submissions")
    else:
        cur.execute("SELECT track_id, ts_epoch FROM submissions")
    rows = cur.fetchall()
    conn.close()

    scores: Dict[int, float] = {t["id"]: 0.0 for t in TRACKS}
    for r in rows:
        tid = r["track_id"]
        age = max(0, now_epoch - int(r["ts_epoch"]))
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
# Runs API (photo upload)
# ----------------------

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8MB
MAX_SIDE = 1600

def _save_upload_stream_to_temp(upload: UploadFile, tmp_path: Path) -> int:
    total = 0
    with open(tmp_path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise ValueError("too_large")
            f.write(chunk)
    return total

def _process_image_to_jpeg(src_path: Path, dst_path: Path) -> None:
    if Image is None:
        shutil.copyfile(src_path, dst_path)
        return

    img = Image.open(src_path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    m = max(w, h)
    if m > MAX_SIDE:
        scale = MAX_SIDE / float(m)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size)

    img.save(dst_path, format="JPEG", quality=85, optimize=True, progressive=True)

@app.post("/api/runs")
async def create_run(
    user: str = Form(...),
    distance_km: float = Form(...),
    duration_sec: int = Form(...),
    date_label: str = Form(...),
    track_id: Optional[int] = Form(None),
    track_ids: Optional[str] = Form(None),
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

    parsed_track_ids: List[int] = []
    if track_ids and track_ids.strip():
        s = track_ids.strip()
        try:
            if s.startswith("["):
                arr = json.loads(s)
                if isinstance(arr, list):
                    parsed_track_ids = [int(x) for x in arr]
            else:
                parsed_track_ids = [int(x) for x in s.split(",") if str(x).strip()]
        except Exception:
            parsed_track_ids = []

    if track_id is not None:
        parsed_track_ids = [int(track_id)] + parsed_track_ids
    parsed_track_ids = [tid for tid in dict.fromkeys(parsed_track_ids) if tid in TRACK_IDS]

    ts_epoch = now_utc_epoch()
    day_kst = kst_day_str_from_epoch(ts_epoch)

    photo_url = None
    if photo is not None:
        # 최종 저장은 jpg로 고정 (리사이즈/압축 때문)
        fname = f"run_{ts_epoch}_{uuid.uuid4().hex[:8]}.jpg"
        out_path = UPLOAD_DIR / fname

        tmp_path = UPLOAD_DIR / f"._tmp_{ts_epoch}_{uuid.uuid4().hex[:8]}"
        try:
            # 1) 스트리밍 저장 (메모리 폭발 방지)
            _save_upload_stream_to_temp(photo, tmp_path)

            # 2) EXIF/리사이즈/재압축 후 저장
            _process_image_to_jpeg(tmp_path, out_path)

            photo_url = f"/static/uploads/{fname}"
        except ValueError as e:
            if str(e) == "too_large":
                return {"ok": False, "error": "사진 파일이 너무 큽니다. (최대 8MB) 줄여서 올려줘."}
            raise
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    conn = db()
    cur = conn.cursor()

    if is_postgres():
        cur.execute(
            f"""
            INSERT INTO runs(user_name, ts_epoch, day_kst, date_label, distance_km, duration_sec, track_id, photo_url)
            VALUES({ph()}, {ph()}, {ph()}, {ph()}, {ph()}, {ph()}, {ph()}, {ph()})
            RETURNING id
            """,
            (user, ts_epoch, day_kst, date_label, float(distance_km), int(duration_sec), track_id, photo_url),
        )
        run_id = cur.fetchone()["id"]
    else:
        cur.execute("""
            INSERT INTO runs(user, ts_epoch, day_kst, date_label, distance_km, duration_sec, track_id, photo_url)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """, (user, ts_epoch, day_kst, date_label, float(distance_km), int(duration_sec), track_id, photo_url))
        conn.commit()
        run_id = cur.lastrowid

    # run_tracks
    for tid in parsed_track_ids:
        if is_postgres():
            cur.execute(
                f"INSERT INTO run_tracks(run_id, track_id) VALUES({ph()}, {ph()}) ON CONFLICT DO NOTHING",
                (run_id, tid),
            )
        else:
            cur.execute("INSERT OR IGNORE INTO run_tracks(run_id, track_id) VALUES(?, ?)", (run_id, tid))

    conn.commit()
    conn.close()

    pace = calc_pace_str(float(distance_km), int(duration_sec))
    vote_result = try_auto_vote_tracks(user, parsed_track_ids, run_id=int(run_id))

    return {
        "ok": True,
        "id": int(run_id),
        "day_kst": day_kst,
        "pace": pace,
        "photo_url": photo_url,
        "tracks": parsed_track_ids,
        "vote": vote_result,
    }


@app.get("/api/runs")
def list_runs(user: Optional[str] = None, limit: int = 30):
    n = max(1, min(int(limit), 100))
    conn = db()
    cur = conn.cursor()

    if user and user.strip():
        if is_postgres():
            cur.execute(
                f"SELECT * FROM runs WHERE user_name={ph()} ORDER BY ts_epoch DESC LIMIT {ph()}",
                (user.strip(), n),
            )
        else:
            cur.execute("SELECT * FROM runs WHERE user=? ORDER BY ts_epoch DESC LIMIT ?", (user.strip(), n))
    else:
        if is_postgres():
            cur.execute(f"SELECT * FROM runs ORDER BY ts_epoch DESC LIMIT {ph()}", (n,))
        else:
            cur.execute("SELECT * FROM runs ORDER BY ts_epoch DESC LIMIT ?", (n,))
    rows = cur.fetchall()

    run_ids = [r["id"] for r in rows]
    tracks_by_run: Dict[int, List[int]] = {int(rid): [] for rid in run_ids}

    if run_ids:
        in_clause = make_in_clause(len(run_ids))
        cur.execute(f"SELECT run_id, track_id FROM run_tracks WHERE run_id IN {in_clause} ORDER BY run_id", run_ids)
        for rr in cur.fetchall():
            tracks_by_run.setdefault(int(rr["run_id"]), []).append(int(rr["track_id"]))

    conn.close()

    out = []
    for r in rows:
        tid = r["track_id"]
        t = TRACK_BY_ID.get(int(tid)) if tid is not None else None
        rid = int(r["id"])
        run_track_ids = tracks_by_run.get(rid, [])
        if (not run_track_ids) and (tid is not None):
            run_track_ids = [int(tid)]

        tracks_info = []
        for x in run_track_ids:
            tt = TRACK_BY_ID.get(int(x))
            if tt:
                tracks_info.append({
                    "id": tt["id"],
                    "title": tt["title"],
                    "artist": tt["artist"],
                    "cover_url": (tt.get("cover_url") or get_cover_from_cache(tt["id"]))
                })

        out.append({
            "id": rid,
            "user": (r["user_name"] if is_postgres() else r["user"]),
            "ts_epoch": int(r["ts_epoch"]),
            "day_kst": r["day_kst"],
            "date_label": r["date_label"],
            "distance_km": float(r["distance_km"]),
            "duration_sec": int(r["duration_sec"]),
            "pace": calc_pace_str(float(r["distance_km"]), int(r["duration_sec"])),
            "track": (
                {"id": t["id"], "title": t["title"], "artist": t["artist"],
                 "cover_url": (t.get("cover_url") or get_cover_from_cache(t["id"]))}
                if t else None
            ),
            "tracks": tracks_info,
            "photo_url": r["photo_url"]
        })
    return out


def _get_run_with_tracks(run_id: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch a run row + its tracks (first is most recent selected)."""
    conn = db()
    cur = conn.cursor()

    if is_postgres():
        cur.execute(f"SELECT * FROM runs WHERE id={ph()} LIMIT 1", (int(run_id),))
    else:
        cur.execute("SELECT * FROM runs WHERE id=? LIMIT 1", (int(run_id),))
    r = cur.fetchone()
    if not r:
        conn.close()
        return None, []

    # tracks for this run
    cur.execute(
        f"SELECT track_id FROM run_tracks WHERE run_id={ph()} ORDER BY track_id",
        (int(run_id),),
    )
    tids = [int(x[0] if isinstance(x, tuple) else x["track_id"]) for x in cur.fetchall()]
    conn.close()

    run = {
        "id": int(r["id"]),
        "user": (r["user_name"] if is_postgres() else r["user"]),
        "date_label": r["date_label"],
        "distance_km": float(r["distance_km"]),
        "duration_sec": int(r["duration_sec"]),
        "photo_url": r["photo_url"],
    }

    tracks: List[Dict[str, Any]] = []
    for tid in tids:
        t = TRACK_BY_ID.get(int(tid))
        if t:
            tracks.append({
                "id": t["id"],
                "title": t.get("title"),
                "artist": t.get("artist"),
                "cover_url": (t.get("cover_url") or get_cover_from_cache(t["id"])),
            })

    return run, tracks


@app.get("/api/runs/{run_id}/share.png", include_in_schema=False)
def run_share_image(run_id: int, aspect: str = "story"):
    """Generate a shareable image with the run photo + stats (NRC-ish)."""
    run, tracks = _get_run_with_tracks(int(run_id))
    if not run:
        return Response(status_code=404)
    try:
        png = _render_run_share_image(run=run, tracks=tracks, aspect=aspect)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
