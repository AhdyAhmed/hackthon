document.addEventListener('DOMContentLoaded', () => {
    var host    = window.location.host;
    var socket  = io(`wss://${host}`);
    var video   = document.getElementById('video');
    var overlay = document.getElementById('overlay');
    var ctx     = overlay.getContext('2d');

    // ── UI elements ──────────────────────────────────────────────────────────
    var faceIndicator = document.getElementById('face-indicator');
    var captureBtn    = document.getElementById('capture-btn');
    var registerForm  = document.getElementById('register-form');
    var verifyBtn     = document.getElementById('verify-btn');
    var resultBox     = document.getElementById('result-box');
    var modeRegister  = document.getElementById('mode-register');
    var modeVerify    = document.getElementById('mode-verify');

    // Off-screen canvas for frame capture only (not shown)
    var captureCanvas = document.createElement('canvas');
    var captureCtx    = captureCanvas.getContext('2d');

    var faceDetected = false;
    var latestBoxes  = [];   // boxes from socket (live)
    var frozenBoxes  = null; // boxes frozen after verify/register click
    var isFrozen     = false;

    // ── Helpers ──────────────────────────────────────────────────────────────

    function syncSize() {
        overlay.width  = overlay.offsetWidth  || 640;
        overlay.height = overlay.offsetHeight || 480;
        captureCanvas.width  = video.videoWidth  || overlay.width;
        captureCanvas.height = video.videoHeight || overlay.height;
    }

    // Draw one frame of the video + bounding boxes onto the overlay canvas.
    // The video is mirrored (scaleX -1) to feel like a mirror to the user.
    function renderFrame() {
        if (!video.srcObject) { requestAnimationFrame(renderFrame); return; }

        syncSize();
        var W = overlay.width;
        var H = overlay.height;

        // Mirror transform
        ctx.save();
        ctx.translate(W, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, W, H);
        ctx.restore();

        // Draw bounding boxes
        var boxes = isFrozen ? frozenBoxes : latestBoxes;
        if (boxes && boxes.length) {
            var scaleX = W  / (captureCanvas.width  || W);
            var scaleY = H  / (captureCanvas.height || H);

            boxes.forEach(function (b) {
                var verified = b.verified;
                var color    = verified ? '#07a66b' : '#d95525';
                var label    = b.label;

                // Flip x for mirror display
                var drawX = W - (b.x + b.w) * scaleX;
                var drawY = b.y * scaleY;
                var drawW = b.w * scaleX;
                var drawH = b.h * scaleY;

                // Box
                ctx.strokeStyle = color;
                ctx.lineWidth   = 2.5;
                ctx.shadowColor = color;
                ctx.shadowBlur  = 10;
                ctx.strokeRect(drawX, drawY, drawW, drawH);
                ctx.shadowBlur  = 0;

                // Corner accents (small L-shapes at each corner)
                var cs = 14;
                ctx.lineWidth = 3.5;
                [
                    [drawX,        drawY,        cs,  0,  cs,  0,  0,  cs],
                    [drawX+drawW,  drawY,       -cs,  0, -cs,  0,  0,  cs],
                    [drawX,        drawY+drawH,  cs,  0,  cs,  0,  0, -cs],
                    [drawX+drawW,  drawY+drawH, -cs,  0, -cs,  0,  0, -cs],
                ].forEach(function (c) {
                    ctx.beginPath();
                    ctx.moveTo(c[0]+c[2], c[1]+c[3]);
                    ctx.lineTo(c[0],      c[1]);
                    ctx.lineTo(c[0]+c[6], c[1]+c[7]);
                    ctx.strokeStyle = '#f29422';
                    ctx.stroke();
                });

                // Label background pill
                ctx.font = 'bold 13px "Nunito", sans-serif';
                var tw   = ctx.measureText(label).width;
                var lx   = drawX;
                var ly   = drawY > 26 ? drawY - 30 : drawY + drawH + 5;
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.roundRect(lx - 2, ly, tw + 14, 22, 4);
                ctx.fill();

                // Label text
                ctx.fillStyle = '#ffffff';
                ctx.fillText(label, lx + 5, ly + 15);
            });
        }

        requestAnimationFrame(renderFrame);
    }

    // ── Camera ───────────────────────────────────────────────────────────────
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
            video.addEventListener('loadedmetadata', function () {
                captureCanvas.width  = video.videoWidth  || 640;
                captureCanvas.height = video.videoHeight || 480;
                syncSize();
                requestAnimationFrame(renderFrame);
            });

            // Send frames every 500 ms for live detection
            setInterval(function () {
                if (video.readyState !== 4) return;
                captureCtx.drawImage(video, 0, 0,
                    captureCanvas.width, captureCanvas.height);
                socket.emit('video_frame',
                    captureCanvas.toDataURL('image/jpeg', 0.6));
            }, 500);
        })
        .catch(function (err) {
            console.error('Camera error:', err);
            showResult(false, 'Camera access denied: ' + err.message);
        });

    window.addEventListener('resize', syncSize);

    // ── Socket events ────────────────────────────────────────────────────────
    socket.on('connect',    function () { console.log('WebSocket connected'); });
    socket.on('disconnect', function () { console.log('WebSocket disconnected'); });

    socket.on('face_status', function (data) {
        faceDetected = data.detected;
        if (!isFrozen) {
            latestBoxes = data.boxes || [];
        }
        if (faceIndicator) {
            if (faceDetected) {
                faceIndicator.innerHTML = `
                    <svg viewBox="0 0 24 24" style="width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round">
                        <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.58-7 8-7s8 3 8 7"/>
                    </svg>
                    Face Detected ✔`;
                faceIndicator.className = 'face-badge ok';
            } else {
                faceIndicator.innerHTML = `
                    <svg viewBox="0 0 24 24" style="width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round">
                        <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.58-7 8-7s8 3 8 7"/>
                    </svg>
                    No Face Detected`;
                faceIndicator.className = 'face-badge fail';
            }
        }
    });

    // ── Mode toggle ──────────────────────────────────────────────────────────
    function setMode(mode) {
        var isRegister = (mode === 'register');
        registerForm.style.display = isRegister ? 'block' : 'none';
        document.getElementById('verify-section').style.display = isRegister ? 'none' : 'block';
        resultBox.innerHTML     = '';
        resultBox.style.display = 'none';
        isFrozen    = false;
        frozenBoxes = null;
        modeRegister.classList.toggle('active', isRegister);
        modeVerify.classList.toggle('active', !isRegister);
    }

    modeRegister.addEventListener('click', function () { setMode('register'); });
    modeVerify.addEventListener('click',   function () { setMode('verify');   });

    // ── Capture current frame (un-mirrored) as base64 ────────────────────────
    function captureFrame() {
        captureCtx.drawImage(video, 0, 0,
            captureCanvas.width, captureCanvas.height);
        return captureCanvas.toDataURL('image/jpeg', 0.9);
    }

    // ── Register ──────────────────────────────────────────────────────────────
    captureBtn.addEventListener('click', async function () {
        var firstName  = document.getElementById('first_name').value.trim();
        var lastName   = document.getElementById('last_name').value.trim();
        var nationalId = document.getElementById('national_id').value.trim();
        var email      = document.getElementById('email').value.trim();
        var phone      = document.getElementById('phone').value.trim();

        if (!firstName || !lastName || !nationalId) {
            showResult(false, "Please fill in the child's First Name, Last Name, and National ID.");
            return;
        }

        captureBtn.disabled    = true;
        captureBtn.textContent = 'Processing…';

        var frame = captureFrame();
        try {
            var res  = await fetch('/api/register', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({
                    first_name: firstName, last_name: lastName,
                    national_id: nationalId, email, phone, frame
                })
            });
            var json = await res.json();
            showResult(json.success, json.message);

            // Show the returned bounding box briefly (3 s)
            if (json.success && json.face_box) {
                var b = json.face_box;
                frozenBoxes = [{
                    x: b.x, y: b.y, w: b.w, h: b.h,
                    label: json.label || (firstName + ' ' + lastName),
                    verified: true
                }];
                isFrozen = true;
                setTimeout(function () { isFrozen = false; frozenBoxes = null; }, 3000);
            }
        } catch (e) {
            showResult(false, 'Network error: ' + e.message);
        } finally {
            captureBtn.disabled    = false;
            captureBtn.innerHTML   = `
                <svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:none;stroke:#fff;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round">
                    <rect x="2" y="7" width="20" height="15" rx="2"/>
                    <path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/>
                    <circle cx="12" cy="14" r="3"/>
                </svg>
                Capture &amp; Register`;
        }
    });

    // ── Geolocation helper ────────────────────────────────────────────────────
    function getLocation() {
        return new Promise(function (resolve) {
            if (!navigator.geolocation) {
                resolve({ latitude: null, longitude: null, location_name: null });
                return;
            }
            navigator.geolocation.getCurrentPosition(
                function (pos) {
                    resolve({
                        latitude:  pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        location_name: null
                    });
                },
                function () {
                    resolve({ latitude: null, longitude: null, location_name: null });
                },
                { timeout: 5000, maximumAge: 60000 }
            );
        });
    }

    async function reverseGeocode(lat, lon) {
        try {
            var res  = await fetch(
                `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`,
                { headers: { 'Accept-Language': 'en' } }
            );
            var data = await res.json();
            var a    = data.address || {};
            var parts = [
                a.city || a.town || a.village || a.county || '',
                a.state || '',
                a.country || ''
            ].filter(Boolean);
            return parts.length ? parts.join(', ') : data.display_name || null;
        } catch (_) {
            return null;
        }
    }

    // ── Verify ────────────────────────────────────────────────────────────────
    verifyBtn.addEventListener('click', async function () {
        verifyBtn.disabled    = true;
        verifyBtn.textContent = 'Getting location…';

        // 1. Grab GPS (non-blocking — times out after 5 s)
        var loc = await getLocation();
        if (loc.latitude !== null) {
            loc.location_name = await reverseGeocode(loc.latitude, loc.longitude);
        }

        verifyBtn.textContent = 'Verifying…';
        var frame = captureFrame();
        try {
            var res  = await fetch('/api/verify', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({
                    frame,
                    latitude:      loc.latitude,
                    longitude:     loc.longitude,
                    location_name: loc.location_name
                })
            });
            var json = await res.json();

            // Freeze bounding boxes from verify response (4 s)
            if (json.face_boxes && json.face_boxes.length) {
                frozenBoxes = json.face_boxes;
                isFrozen    = true;
                setTimeout(function () { isFrozen = false; frozenBoxes = null; }, 4000);
            }

            if (json.verified) {
                var p       = json.person;
                var locLine = loc.location_name
                    ? '<br><span class="person-info">📍 ' + loc.location_name + '</span>'
                    : '';
                showResult(true,
                    json.message + '<br>' +
                    '<span class="person-info">' +
                        p.first_name + ' ' + p.last_name +
                        ' &nbsp;|&nbsp; ID: ' + p.national_id +
                    '</span>' + locLine
                );
            } else {
                showResult(false, json.message || 'Failed to Verify');
            }
        } catch (e) {
            showResult(false, 'Network error: ' + e.message);
        } finally {
            verifyBtn.disabled    = false;
            verifyBtn.innerHTML   = `
                <svg viewBox="0 0 24 24" style="width:19px;height:19px;fill:none;stroke:#fff;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                Verify Identity`;
        }
    });

    // ── Result helper ─────────────────────────────────────────────────────────
    function showResult(ok, msg) {
        resultBox.innerHTML     = msg;
        resultBox.className     = 'result-box ' + (ok ? 'result-ok' : 'result-fail');
        resultBox.style.display = 'block';
    }

    // default mode
    setMode('verify');
});