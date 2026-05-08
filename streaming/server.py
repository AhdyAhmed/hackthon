import ssl
import cv2
import base64
import numpy as np
import os
import json
import smtplib
import psycopg2
import psycopg2.extras
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from scipy.spatial.distance import cosine
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

# ── Flask & SocketIO ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'face-verify-secret'
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ── Folders ───────────────────────────────────────────────────────────────────
PERSONS_DIR = "persons"
os.makedirs(PERSONS_DIR, exist_ok=True)

# ── DB config ─────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "face_verify"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# ── Gmail config ──────────────────────────────────────────────────────────────
GMAIL_USER     = os.getenv("GMAIL_USER", "")
GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

# ── Face encoder (MobileNetV2 feature extractor) ──────────────────────────────
print("[INFO] Loading MobileNetV2 feature extractor …")
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)
base_model.trainable = False
print("[INFO] Model ready.")

# ── Haar cascade ──────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

SIMILARITY_THRESHOLD = 0.35   # cosine distance; lower = more similar


# ─────────────────────────────────────────────────────────────────────────────
#  Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id           SERIAL PRIMARY KEY,
            first_name   VARCHAR(100) NOT NULL,
            last_name    VARCHAR(100) NOT NULL,
            national_id  VARCHAR(50)  UNIQUE NOT NULL,
            email        VARCHAR(150),
            phone        VARCHAR(30),
            photo_path   TEXT,
            embedding    TEXT,
            created_at   TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verification_logs (
            id            SERIAL PRIMARY KEY,
            person_id     INTEGER REFERENCES persons(id) ON DELETE SET NULL,
            national_id   VARCHAR(50),
            verified      BOOLEAN NOT NULL,
            distance      FLOAT,
            latitude      DOUBLE PRECISION,
            longitude     DOUBLE PRECISION,
            location_name TEXT,
            ip_address    VARCHAR(50),
            verified_at   TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database initialised.")


# ─────────────────────────────────────────────────────────────────────────────
#  Face helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_frame(data_url: str) -> np.ndarray | None:
    """base64 data-URL → OpenCV BGR frame."""
    try:
        _, b64 = data_url.split(',', 1)
        img_bytes = base64.b64decode(b64)
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] decode_frame: {e}")
        return None


def detect_faces_raw(frame: np.ndarray):
    """
    Return list of (x, y, w, h) for all detected faces (original frame coords).
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return []
    return [tuple(int(v) for v in f) for f in faces]


def crop_face(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop a single face from frame with 20 % padding."""
    pad = int(min(w, h) * 0.20)
    x1  = max(0, x - pad);            y1 = max(0, y - pad)
    x2  = min(frame.shape[1], x + w + pad)
    y2  = min(frame.shape[0], y + h + pad)
    return frame[y1:y2, x1:x2]


def get_embedding(face_crop: np.ndarray) -> np.ndarray:
    """MobileNetV2 embedding — L2-normalised float32 vector."""
    img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, 0)
    emb = base_model.predict(img, verbose=0)[0]
    return emb / (np.linalg.norm(emb) + 1e-9)


def load_all_embeddings():
    """Fetch all persons' embeddings from DB."""
    conn = get_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT id, first_name, last_name, national_id, email, embedding
        FROM persons;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def match_embedding(probe_emb: np.ndarray, rows) -> tuple[dict | None, float]:
    """Return (best_person_row, best_cosine_distance) against all DB rows."""
    best_dist   = float('inf')
    best_person = None
    for row in rows:
        stored_emb = np.array(json.loads(row['embedding']), dtype=np.float32)
        dist = cosine(probe_emb, stored_emb)
        if dist < best_dist:
            best_dist   = dist
            best_person = row
    return best_person, best_dist


# ─────────────────────────────────────────────────────────────────────────────
#  Email helper
# ─────────────────────────────────────────────────────────────────────────────



def _send_email_task(to_email: str, first_name: str, last_name: str,
                     national_id: str, timestamp: str,
                     latitude: float | None, longitude: float | None,
                     location_name: str | None,
                     photo_b64: str | None = None):          # ← NEW param
    """Internal function run in a background thread."""
    if not GMAIL_USER or not GMAIL_PASSWORD:
        print("[WARN] Gmail credentials missing — email skipped.")
        return

    # ── Location block ────────────────────────────────────────────────────────
    if latitude is not None and longitude is not None:
        maps_url   = f"https://www.google.com/maps?q={latitude},{longitude}"
        coords_str = f"{latitude:.6f}, {longitude:.6f}"
        place_str  = location_name or coords_str
        location_html = f"""
        <div class="info-row">
          <span class="info-label">📍 Location</span>
          <span class="info-value">
            {place_str}<br>
            <small style="color:#888;font-weight:400">{coords_str}</small><br>
            <a href="{maps_url}" style="color:#0066ff;font-size:12px;text-decoration:none"
               target="_blank">View on Google Maps →</a>
          </span>
        </div>"""
    else:
        location_html = """
        <div class="info-row">
          <span class="info-label">📍 Location</span>
          <span class="info-value" style="color:#aaa">Not available</span>
        </div>"""

    # ── Photo block (inline CID attachment) ──────────────────────────────────
    if photo_b64:
        photo_html = """
        <div style="text-align:center; margin: 24px 0 8px;">
          <p style="color:#555; font-size:13px; margin:0 0 10px; font-weight:600;
                    text-transform:uppercase; letter-spacing:.6px;">
            Captured Photo
          </p>
          <img src="cid:verification_photo"
               alt="Verification Photo"
               style="max-width:260px; border-radius:10px;
                      border:3px solid #00d4aa;
                      box-shadow:0 4px 16px rgba(0,0,0,.15);" />
        </div>"""
    else:
        photo_html = ""

    subject = "📸 Your Child Has Been Verified – School Entry Notification"

    html_body = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body       {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9;
                  margin:0; padding:0; }}
    .container {{ max-width:540px; margin:40px auto; background:#fff;
                  border-radius:14px; overflow:hidden;
                  box-shadow:0 4px 24px rgba(0,0,0,.10); }}
    .header    {{ background: linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
                  padding:36px 32px; text-align:center; }}
    .header h1 {{ color:#fff; margin:0 0 4px; font-size:20px; font-weight:700;
                  letter-spacing:.5px; }}
    .header p  {{ color:#a0b4cc; font-size:13px; margin:0; }}
    .badge     {{ display:inline-block; background:#00d4aa; color:#fff;
                  border-radius:50px; padding:6px 22px; font-size:12px;
                  font-weight:700; margin-top:14px; letter-spacing:1px;
                  text-transform:uppercase; }}
    .body      {{ padding:32px; }}
    .body p    {{ color:#555; line-height:1.75; font-size:15px; margin:0 0 16px; }}
    .info-box  {{ background:#f8faff; border:1px solid #e2e8f0; border-radius:10px;
                  padding:20px 24px; margin:20px 0; }}
    .info-row  {{ display:flex; justify-content:space-between; align-items:flex-start;
                  padding:9px 0; border-bottom:1px solid #edf2f7; font-size:14px; }}
    .info-row:last-child {{ border-bottom:none; }}
    .info-label {{ color:#888; font-weight:500; min-width:110px; padding-right:10px; }}
    .info-value {{ color:#1a1a2e; font-weight:600; text-align:right; line-height:1.5; }}
    .warning   {{ background:#fff8e1; border-left:4px solid #f59e0b;
                  padding:14px 18px; border-radius:0 8px 8px 0;
                  font-size:13px; color:#92400e; margin-top:20px; }}
    .footer    {{ background:#f8faff; padding:20px 32px; text-align:center;
                  font-size:12px; color:#aaa; border-top:1px solid #edf2f7; }}
  </style>
</head>
<body>
  <div class="container">

    <div class="header">
      <h1>🏫 School Entry Notification</h1>
      <p>Face Recognition Attendance System</p>
      <span class="badge">✅ Child Verified</span>
    </div>

    <div class="body">
      <p>Dear Parent / Guardian,</p>
      <p>
        We are pleased to inform you that your child,
        <strong>{first_name} {last_name}</strong>, has been
        <strong>successfully identified and verified</strong> at the school premises
        via our facial recognition system.
      </p>

      {photo_html}

      <div class="info-box">
        <div class="info-row">
          <span class="info-label">👤 Child Name</span>
          <span class="info-value">{first_name} {last_name}</span>
        </div>
        <div class="info-row">
          <span class="info-label">🪪 National ID</span>
          <span class="info-value">{national_id}</span>
        </div>
        <div class="info-row">
          <span class="info-label">🕐 Time</span>
          <span class="info-value">{timestamp}</span>
        </div>
        {location_html}
      </div>

      <p style="font-size:14px; color:#666;">
        Please keep this record for your reference. If this verification was unexpected
        or you believe there has been an error, contact the school administration immediately.
      </p>

      <div class="warning">
        ⚠️ If your child did not arrive at school today or you did not expect this
        notification, please contact us right away.
      </div>
    </div>

    <div class="footer">
      This is an automated notification from the School Face Verify System. Do not reply.
    </div>

  </div>
</body>
</html>
"""

    # ── Build MIME message ────────────────────────────────────────────────────
    msg = MIMEMultipart("related")          # "related" allows inline CID images
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = to_email

    # Attach the HTML part inside a "alternative" sub-container
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    # Attach the photo as an inline CID image
    if photo_b64:
        try:
            img_data = base64.b64decode(photo_b64)
            from email.mime.image import MIMEImage
            img_part = MIMEImage(img_data, _subtype="jpeg")
            img_part.add_header("Content-ID", "<verification_photo>")
            img_part.add_header("Content-Disposition", "inline",
                                filename="verification.jpg")
            msg.attach(img_part)
        except Exception as img_err:
            print(f"[WARN] Could not attach photo to email: {img_err}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())
        print(f"[INFO] Notification email sent → {to_email}")
    except Exception as e:
        print(f"[ERROR] Email send failed → {to_email}: {e}")


def send_verification_email(to_email: str, first_name: str, last_name: str,
                             national_id: str, latitude: float | None = None,
                             longitude: float | None = None,
                             location_name: str | None = None,
                             photo_b64: str | None = None):   # ← NEW param
    """Fire-and-forget: send email in a background thread."""
    if not to_email:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    threading.Thread(
        target=_send_email_task,
        args=(to_email, first_name, last_name, national_id, ts,
              latitude, longitude, location_name, photo_b64),  # ← pass it
        daemon=True
    ).start()


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/register', methods=['POST'])
def register_person():
    """Register a new person: save photo + embedding + info to DB."""
    try:
        data        = request.get_json()
        first_name  = data['first_name'].strip()
        last_name   = data['last_name'].strip()
        national_id = data['national_id'].strip()
        email       = data.get('email', '').strip()
        phone       = data.get('phone', '').strip()
        frame_data  = data['frame']

        frame = decode_frame(frame_data)
        if frame is None:
            return jsonify(success=False, message="Invalid image data."), 400

        faces = detect_faces_raw(frame)
        if not faces:
            return jsonify(success=False,
                           message="No face detected. Please try again."), 400

        # Use largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_crop  = crop_face(frame, x, y, w, h)
        embedding  = get_embedding(face_crop)

        folder_name = f"{national_id}_{first_name}_{last_name}".replace(' ', '_')
        person_dir  = os.path.join(PERSONS_DIR, folder_name)
        os.makedirs(person_dir, exist_ok=True)
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = os.path.join(person_dir, f"photo_{ts}.jpg")
        cv2.imwrite(photo_path, frame)

        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO persons
                (first_name, last_name, national_id, email, phone, photo_path, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (national_id) DO UPDATE
              SET first_name  = EXCLUDED.first_name,
                  last_name   = EXCLUDED.last_name,
                  email       = EXCLUDED.email,
                  phone       = EXCLUDED.phone,
                  photo_path  = EXCLUDED.photo_path,
                  embedding   = EXCLUDED.embedding
            RETURNING id;
        """, (first_name, last_name, national_id, email, phone,
              photo_path, json.dumps(embedding.tolist())))
        person_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return jsonify(
            success=True,
            message=f"Registered successfully! ID: {person_id}",
            person_id=person_id,
            # Return the box so the frontend can draw it on the snapshot
            face_box={"x": x, "y": y, "w": w, "h": h},
            label=f"{first_name} {last_name}"
        )

    except psycopg2.errors.UniqueViolation:
        return jsonify(success=False,
                       message="National ID already registered."), 409
    except Exception as e:
        print(f"[ERROR] /api/register: {e}")
        return jsonify(success=False, message=str(e)), 500


@app.route('/api/verify', methods=['POST'])
def verify_person():
    """
    Verify a face against all registered embeddings.
    Accepts optional latitude/longitude/location_name from the client.
    Logs every attempt to verification_logs.
    Sends a parent/guardian notification email with the captured photo.
    Returns face_boxes list with label + verified flag for canvas drawing.
    """
    try:
        data          = request.get_json()
        frame_data    = data['frame']
        latitude      = data.get('latitude')       # float or None
        longitude     = data.get('longitude')      # float or None
        location_name = data.get('location_name')  # reverse-geocoded string or None
        ip_address    = request.remote_addr

        frame = decode_frame(frame_data)
        if frame is None:
            return jsonify(success=False, verified=False,
                           message="Invalid image data."), 400

        faces = detect_faces_raw(frame)
        if not faces:
            return jsonify(
                success=False, verified=False,
                message="No face detected. Please look at the camera.",
                face_boxes=[]
            ), 400

        rows = load_all_embeddings()

        results              = []
        any_verified         = False
        best_verified_person = None

        for (x, y, w, h) in faces:
            face_crop = crop_face(frame, x, y, w, h)
            probe_emb = get_embedding(face_crop)

            if rows:
                best_person, best_dist = match_embedding(probe_emb, rows)
                verified = bool(best_dist <= SIMILARITY_THRESHOLD)
            else:
                best_person, best_dist, verified = None, 1.0, False

            if verified and best_person:
                label = f"{best_person['first_name']} {best_person['last_name']}"
                any_verified = True
                if best_verified_person is None:
                    best_verified_person = (best_person, best_dist)
            else:
                label    = "Unknown"
                verified = False

            results.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label":    label,
                "verified": bool(verified),
                "distance": round(float(best_dist), 4) if rows else None,
            })

        # ── Log every verification attempt to DB ──────────────────────────────
        try:
            conn_log = get_db()
            cur_log  = conn_log.cursor()
            if best_verified_person:
                p_log, d_log = best_verified_person
                cur_log.execute("""
                    INSERT INTO verification_logs
                        (person_id, national_id, verified, distance,
                         latitude, longitude, location_name, ip_address)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """, (p_log['id'], p_log['national_id'], True,
                      round(float(d_log), 4),
                      latitude, longitude, location_name, ip_address))
            else:
                cur_log.execute("""
                    INSERT INTO verification_logs
                        (person_id, national_id, verified, distance,
                         latitude, longitude, location_name, ip_address)
                    VALUES (NULL, NULL, FALSE, NULL, %s, %s, %s, %s);
                """, (latitude, longitude, location_name, ip_address))
            conn_log.commit()
            cur_log.close()
            conn_log.close()
        except Exception as log_err:
            print(f"[WARN] Failed to write verification log: {log_err}")

        # ── Encode the verified frame as JPEG base64 for the email ────────────
        verified_photo_b64 = None
        try:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            verified_photo_b64 = base64.b64encode(buf).decode('utf-8')
        except Exception as enc_err:
            print(f"[WARN] Could not encode frame for email: {enc_err}")

        # ── Email + response ──────────────────────────────────────────────────
        if best_verified_person:
            p, dist = best_verified_person
            send_verification_email(
                to_email      = p['email'],
                first_name    = p['first_name'],
                last_name     = p['last_name'],
                national_id   = p['national_id'],
                latitude      = latitude,
                longitude     = longitude,
                location_name = location_name,
                photo_b64     = verified_photo_b64,
            )
            return jsonify(
                success=True,
                verified=True,
                message="Identity Verified ✓",
                email_sent=bool(p['email']),
                person={
                    "id":          p['id'],
                    "first_name":  p['first_name'],
                    "last_name":   p['last_name'],
                    "national_id": p['national_id'],
                    "email":       p['email'],
                },
                distance=round(float(dist), 4),
                face_boxes=results
            )
        else:
            return jsonify(
                success=True,
                verified=False,
                message="Failed to Verify",
                face_boxes=results
            )

    except Exception as e:
        print(f"[ERROR] /api/verify: {e}")
        return jsonify(success=False, verified=False, message=str(e)), 500


# ─────────────────────────────────────────────────────────────────────────────
#  SocketIO – live face-detection + real-time bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

# Cache DB rows every 5 s so the socket handler isn't hitting DB every frame
_db_cache       = {"rows": [], "ts": 0}
_DB_CACHE_TTL   = 5   # seconds

def get_cached_rows():
    import time
    now = time.time()
    if now - _db_cache["ts"] > _DB_CACHE_TTL:
        try:
            _db_cache["rows"] = load_all_embeddings()
            _db_cache["ts"]   = now
        except Exception as e:
            print(f"[WARN] DB cache refresh failed: {e}")
    return _db_cache["rows"]


@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Detect faces in the live frame, optionally match them, and emit:
      face_status { detected, boxes: [{x,y,w,h,label,verified}] }
    """
    try:
        frame = decode_frame(data)
        if frame is None:
            return

        faces = detect_faces_raw(frame)
        if not faces:
            emit('face_status', {'detected': False, 'boxes': []})
            return

        rows   = get_cached_rows()
        boxes  = []

        for (x, y, w, h) in faces:
            if rows:
                face_crop = crop_face(frame, x, y, w, h)
                probe_emb = get_embedding(face_crop)
                best_person, best_dist = match_embedding(probe_emb, rows)
                verified = bool(best_dist <= SIMILARITY_THRESHOLD)
                label    = (f"{best_person['first_name']} {best_person['last_name']}"
                            if verified and best_person else "Unknown")
            else:
                label    = "Unknown"
                verified = False

            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                          "label": label, "verified": verified})

        emit('face_status', {'detected': True, 'boxes': boxes})

    except Exception as e:
        print(f"[ERROR] handle_video_frame: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()

    cert_file = 'certificates/certificate.crt'
    key_file  = 'certificates/private.key'

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    socketio.run(app, host='0.0.0.0', port=5000, ssl_context=ssl_context)