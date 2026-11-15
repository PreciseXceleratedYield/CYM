"""Offline FFmpeg-based video restoration for CYM.

This is conventional image processing, not neural super-resolution.
"""
import subprocess

def enhance_video(source, destination, height=1080, strength="balanced"):
    if height not in (0, 720, 1080, 2160):
        raise ValueError("Invalid resolution")
    settings = {
        "gentle": ("1.5:1.5:3:3", "3:3:0.35:3:3:0"),
        "balanced": ("3:3:6:6", "5:5:0.55:3:3:0"),
        "strong": ("5:5:8:8", "5:5:0.8:3:3:0"),
    }
    if strength not in settings:
        raise ValueError("Invalid enhancement strength")
    denoise, sharpen = settings[strength]
    filters = [f"hqdn3d={denoise}"]
    if height:
        # Scale up only, retain original aspect ratio and even dimensions.
        filters.append(
            f"scale=w='trunc(iw*max(1\\,{height}/ih)/2)*2':"
            f"h='trunc(ih*max(1\\,{height}/ih)/2)*2':flags=lanczos"
        )
    filters.append(f"unsharp={sharpen}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(source), "-map", "0:v:0", "-an",
           "-vf", ",".join(filters), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           str(destination)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2500:])
