# CYM — Color Your Movie

Local black-and-white movie colorization and audio cleanup for macOS. No cloud AI account, API key, or recurring AI fee required.

## Requirements
- Apple Silicon Mac (tested installation target: M3 / 16 GB; actual end-to-end processing **not yet tested**)
- macOS, Homebrew, Python 3.12, FFmpeg
- Free disk space: preferably at least 3× input movie size
- The original footage must be black-and-white and you must have the rights to process/distribute it.

## Setup

```bash
git clone https://github.com/PreciseXceleratedYield/CYM.git
cd CYM
bash install.sh
bash download_models.sh
bash start.sh
```

Open **http://127.0.0.1:8000** in Safari. Upload a short black-and-white test clip first. Processing happens locally in the background; keep the Mac awake, plugged in, and the server running. Closing Safari is fine, but closing the laptop lid normally sleeps the machine.

## Local enhancement modes

CYM now offers three modes in the web UI: **COLOR**, **ENHANCE**, and **COLOR + ENHANCE**.
ENHANCE preserves existing colors and applies offline FFmpeg denoising, Lanczos upscaling
(up to 720p/1080p/4K), and configurable sharpening. The default is balanced, up to 1080p.
Upscaling never intentionally reduces source resolution. Original frame rate is retained.

**Important:** This first enhancement implementation uses traditional FFmpeg filters,
**not neural AI upscaling**. It can improve perceived quality but cannot reconstruct
missing detail or perform AI face restoration, scratch inpainting, or optical-flow
frame interpolation. Those require separately integrated and tested local models.

## Features
- Upload MP4/MKV/MOV/AVI
- Automatic shot-by-shot colorization using the Zhang/OpenCV Caffe model
- Audio cleaning using FFmpeg's `afftdn` filter and `loudnorm` (DSP, **not AI voice separation**)
- Progress and status; download final MP4
- One processing job at a time to protect memory on 16 GB machines
- Resumable processing at completed video segment boundaries after app restart
- Localhost-only web server; files remain on the Mac
- Jobs and inputs persist in `data/`; delete files there when no longer needed

## Important limitations
- This is a functional initial implementation, **not yet benchmarked or verified end-to-end on your Mac**.
- Automated colorization can flicker, miscolor clothing/skin, or change colors across shots. The model predicts plausible colors, not historical accuracy.
- Audio cleanup improves some noise and loudness, but cannot reliably isolate speech from music or reconstruct damaged dialogue.
- Colorization is performed at 256×256 inference resolution and resized back to source dimensions; it does not add genuine 4K detail.
- Video is re-encoded to H.264 and audio to AAC. Subtitle tracks and additional audio tracks are not retained in v1.
- A job can take days or weeks on a fanless MacBook Air. Keep enough free disk space.
- The model weights are downloaded from a third party; verify their **commercial-use rights** independently before monetizing restored films. The repository does not redistribute weights.
- Check film, music, script, performance, master and YouTube distribution rights before publishing.

## Processing details
The server splits the input into 60-second segments with FFmpeg, colorizes frames locally using OpenCV DNN, and muxes enhanced audio. Completed segments are saved to disk and skipped on retry. The final movie is assembled from processed segments. Progress is approximate.

## License
Source code in this repository: MIT (see LICENSE). External model and media licenses are separate.
