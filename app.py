"""CYM: localhost-only background movie colorization web app."""
import json
import math
import os
import shutil
import subprocess
import threading
import time
import logging
import uuid
from pathlib import Path

from enhance import enhance_video
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
ALLOWED = {".mp4", ".mkv", ".mov", ".avi", ".mpg", ".mpeg"}
CHUNK_SECONDS = 60
LOCK = threading.Lock()
app = FastAPI(title="CYM — Color Your Movie")
worker = None
PROCESS_LOCK = threading.Lock()
LOG = logging.getLogger("cym")


def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-2500:])
    return p.stdout


def probe(path):
    info = json.loads(command(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                               "-of", "json", str(path)]))
    video = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), None)
    if not video:
        raise ValueError("No video stream found")
    duration = float(info["format"].get("duration", video.get("duration", 0)))
    if duration <= 0:
        raise ValueError("Unable to determine video duration")
    return duration, any(s["codec_type"] == "audio" for s in info["streams"])


def state_path(job):
    return DATA / job / "state.json"


def read_state(job):
    path = state_path(job)
    if not path.is_file():
        raise HTTPException(404, "Job not found")
    return json.loads(path.read_text())


def write_state(job, **updates):
    with LOCK:
        path = state_path(job)
        state = json.loads(path.read_text())
        state.update(updates)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(path)


class Colorizer:
    def __init__(self):
        proto = MODELS / "colorization_deploy_v2.prototxt"
        weights = MODELS / "colorization_release_v2.caffemodel"
        pts_file = MODELS / "pts_in_hull.npy"
        if not all(p.is_file() for p in (proto, weights, pts_file)):
            raise RuntimeError("Colorization model missing. Run bash download_models.sh first.")
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        pts = np.load(str(pts_file)).transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(self.net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
        self.net.getLayer(self.net.getLayerId("conv8_313_rh")).blobs = [
            np.full((1, 313), 2.606, dtype=np.float32)
        ]

    def frame(self, bgr):
        # OpenCV's Zhang colorization: grayscale L channel in Lab space.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_orig = lab[:, :, 0]
        l_small = cv2.resize(l_orig, (224, 224))
        l_small -= 50
        self.net.setInput(cv2.dnn.blobFromImage(l_small))
        ab = self.net.forward()[0].transpose((1, 2, 0))
        ab = cv2.resize(ab, (bgr.shape[1], bgr.shape[0]))
        color_lab = np.concatenate((l_orig[:, :, None], ab), axis=2).astype(np.float32)
        color_rgb = np.clip(cv2.cvtColor(color_lab, cv2.COLOR_LAB2RGB), 0, 1)
        return cv2.cvtColor((color_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def colorize_video(src, dest, model):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("Could not read input video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video frame rate")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 1 or height < 1:
        cap.release()
        raise RuntimeError("Invalid video dimensions")
    # OpenCV's mp4v is a temporary intermediate, final output is H.264.
    out = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Could not open video writer")
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(model.frame(frame))
            count += 1
    finally:
        cap.release()
        out.release()
    if count == 0:
        raise RuntimeError("No frames decoded from input video")


def process(job):
    folder = DATA / job
    try:
        st = read_state(job)
        src = folder / st["input"]
        duration, has_audio = probe(src)
        parts = math.ceil(duration / CHUNK_SECONDS)
        write_state(job, status="running", total_parts=parts, error=None)
        mode = st.get('mode', 'color')
        model = Colorizer() if mode != 'enhance' else None
        for i in range(parts):
            result = folder / f"part_{i:05d}.mp4"
            if result.is_file() and result.stat().st_size > 0:
                write_state(job, completed_parts=i + 1, progress=round(90 * (i + 1) / parts))
                continue
            raw = folder / f"raw_{i:05d}.mp4"
            colored = folder / f"colored_{i:05d}.mp4"
            temp_result = folder / f"part_{i:05d}.tmp.mp4"
            start = i * CHUNK_SECONDS
            length = min(CHUNK_SECONDS, duration - start)
            try:
                command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                         "-ss", str(start), "-i", str(src), "-t", str(length),
                         "-map", "0:v:0", "-an", "-c:v", "mpeg4", "-q:v", "3", str(raw)])
                if mode == 'enhance':
                    enhance_video(raw, colored, st.get('height', 1080), st.get('strength', 'balanced'))
                elif mode == 'both':
                    first = folder / f"first_{i:05d}.mp4"
                    try:
                        colorize_video(raw, first, model)
                        enhance_video(first, colored, st.get('height', 1080), st.get('strength', 'balanced'))
                    finally:
                        first.unlink(missing_ok=True)
                else:
                    colorize_video(raw, colored, model)
                args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(colored)]
                if has_audio:
                    args += ["-ss", str(start), "-t", str(length), "-i", str(src),
                             "-map", "0:v:0", "-map", "1:a:0",
                             "-af", "highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
                             "-c:a", "aac", "-b:a", "192k"]
                else:
                    args += ["-an"]
                args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                         "-shortest", str(temp_result)]
                command(args)
                temp_result.replace(result)
            finally:
                for tmp in (raw, colored, temp_result):
                    tmp.unlink(missing_ok=True)
            write_state(job, completed_parts=i + 1, progress=round(90 * (i + 1) / parts))
        manifest = folder / "concat.txt"
        manifest.write_text("".join(f"file 'part_{i:05d}.mp4'\n" for i in range(parts)))
        final_tmp = folder / "output.tmp.mp4"
        command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "concat", "-safe", "0", "-i", str(manifest),
                 "-c", "copy", "-movflags", "+faststart", str(final_tmp)])
        final_tmp.replace(folder / "output.mp4")
        write_state(job, status="complete", progress=100)
    except Exception as exc:
        LOG.exception("CYM processing failed for %s", job)
        write_state(job, status="failed", error=str(exc)[:2000])
    finally:
        PROCESS_LOCK.release()


def start_job(job):
    global worker
    if not PROCESS_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another movie is already processing")
    try:
        worker = threading.Thread(target=process, args=(job,), daemon=True)
        worker.start()
    except Exception:
        PROCESS_LOCK.release()
        raise


@app.on_event("startup")
def recover():
    # Jobs persist across restarts. Resume the oldest interrupted job.
    for p in sorted(DATA.glob("*/state.json")):
        try:
            state = json.loads(p.read_text())
            if state["status"] == "running":
                write_state(p.parent.name, status="interrupted")
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/api/jobs")
async def upload(file: UploadFile = File(...), mode: str = 'color', height: int = 1080, strength: str = 'balanced'):
    if mode not in ('color', 'enhance', 'both') or height not in (0, 720, 1080, 2160) or strength not in ('gentle', 'balanced', 'strong'):
        raise HTTPException(400, 'Invalid processing options')
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Supported formats: MP4, MKV, MOV, AVI, MPG, MPEG")
    job = uuid.uuid4().hex
    folder = DATA / job
    folder.mkdir()
    target = folder / ("input" + suffix)
    try:
        with target.open("wb") as out:
            while chunk := await file.read(4 * 1024 * 1024):
                out.write(chunk)
        probe(target)
        (folder / "state.json").write_text(json.dumps({
            "id": job, "input": target.name, "status": "queued", "progress": 0,
            "completed_parts": 0, "total_parts": 0, "error": None,
            "mode": mode, "height": height, "strength": strength
        }))
        try:
            start_job(job)
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            # Preserve the upload and allow processing when the active job ends.
        return {"id": job}
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        await file.close()


@app.get("/api/jobs/{job}")
def status(job: str):
    if not job.isalnum() or len(job) != 32:
        raise HTTPException(404)
    return read_state(job)


@app.post("/api/jobs/{job}/resume")
def resume(job: str):
    if not job.isalnum() or len(job) != 32:
        raise HTTPException(404)
    state = read_state(job)
    if state["status"] == "complete":
        return state
    start_job(job)
    return read_state(job)


@app.get("/api/jobs/{job}/download")
def download(job: str):
    if not job.isalnum() or len(job) != 32:
        raise HTTPException(404)
    state = read_state(job)
    path = DATA / job / "output.mp4"
    if state["status"] != "complete" or not path.is_file():
        raise HTTPException(409, "Movie is not ready")
    return FileResponse(path, media_type="video/mp4", filename="CYM_" + state.get("mode", "color") + "_movie.mp4")


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CYM — Color Your Movie</title>
<style>
body{font:16px -apple-system,BlinkMacSystemFont,sans-serif;background:#f6f8fb;color:#17233c;margin:0}
main{max-width:660px;margin:7vh auto;padding:32px;background:white;border-radius:20px;box-shadow:0 12px 45px #182b4a18}
h1{margin:0 0 8px;font-size:30px}p{color:#526078;line-height:1.6}
input{display:block;margin:25px 0;max-width:100%}button,a.action{background:#2357db;color:white;border:0;padding:13px 19px;border-radius:10px;font-size:16px;cursor:pointer;text-decoration:none;display:inline-block}
button:disabled{opacity:.5}progress{width:100%;height:20px;margin:18px 0}
#panel{display:none;border-top:1px solid #e1e5ee;margin-top:25px;padding-top:20px}
small{color:#637087;display:block;margin-top:15px;word-break:break-word}
</style></head><body><main>
<h1>🎞️ CYM — Color Your Movie</h1>
<p>Turn black-and-white footage into color and clean up its original soundtrack. All processing stays on your Mac.</p>
<form id="form">
<label>Mode <select id="mode"><option value="color">COLOR</option><option value="enhance">ENHANCE</option><option value="both">COLOR + ENHANCE</option></select></label>
<label>Resolution <select id="height"><option value="0">Original</option><option value="720">720p</option><option value="1080" selected>1080p</option><option value="2160">4K</option></select></label>
<label>Strength <select id="strength"><option value="gentle">Gentle</option><option value="balanced" selected>Balanced</option><option value="strong">Strong</option></select></label>
<input type="file" id="movie" accept=".mp4,.mkv,.mov,.avi,.mpg,.mpeg" required>
<button id="go">Transform movie</button></form>
<div id="panel"><h2 id="status">Preparing…</h2><progress id="progress" value="0" max="100"></progress>
<p id="detail"></p><button id="resume" style="display:none">Resume processing</button>
<a class="action" id="download" style="display:none">Download finished movie</a>
<small id="jobid"></small></div>
<small>Keep the Mac plugged in and awake. You may close this browser tab while processing.</small>
</main><script>
const $=id=>document.getElementById(id);let active=null;
async function check(){
 if(!active)return;
 try{
  const r=await fetch('/api/jobs/'+active);if(!r.ok)throw Error('Status unavailable');
  const s=await r.json();$('panel').style.display='block';$('status').textContent=s.status.toUpperCase();
  $('progress').value=s.progress||0;
  $('detail').textContent=s.error||('Completed segments: '+s.completed_parts+' / '+s.total_parts+' ('+s.progress+'%)');
  $('jobid').textContent='Job ID: '+active;
  $('resume').style.display=['failed','interrupted','queued'].includes(s.status)?'inline-block':'none';
  $('download').style.display=s.status==='complete'?'inline-block':'none';
  $('download').href='/api/jobs/'+active+'/download';
  if(s.status==='complete')$('go').disabled=false;
 }catch(e){$('detail').textContent=e.message}
}
$('form').onsubmit=async e=>{
 e.preventDefault();$('go').disabled=true;$('panel').style.display='block';$('status').textContent='Uploading…';
 const fd=new FormData();fd.append('file',$('movie').files[0]);fd.append('mode',$('mode').value);fd.append('height',$('height').value);fd.append('strength',$('strength').value);
 try{const r=await fetch('/api/jobs',{method:'POST',body:fd});const j=await r.json();if(!r.ok)throw Error(j.detail||'Upload failed');
 active=j.id;localStorage.setItem('cym_job',active);await check()
 }catch(err){$('status').textContent='ERROR';$('detail').textContent=err.message;$('go').disabled=false}
};
$('resume').onclick=async()=>{const r=await fetch('/api/jobs/'+active+'/resume',{method:'POST'});if(!r.ok){const j=await r.json();alert(j.detail||'Could not resume')}await check()};
active=localStorage.getItem('cym_job');if(active)check();setInterval(check,3000);
</script></body></html>"""
