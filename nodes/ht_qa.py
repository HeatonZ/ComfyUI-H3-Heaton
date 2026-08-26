"""Output QA: fail loudly on missing ffprobe or malformed media."""
import json, os, subprocess, tempfile
from pathlib import Path

import torch
from ..core.latent import probe


def _run(command, label):
    try:
        return subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("HT QA: %s was not found; install FFmpeg and put it on PATH" % label) from exc


def _ffprobe(path):
    p = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], "ffprobe")
    if p.returncode:
        raise RuntimeError("HT QA: ffprobe failed for %s: %s" % (path, p.stderr.strip()))
    return json.loads(p.stdout)


def _silences(path):
    p = _run(["ffmpeg", "-hide_banner", "-i", path, "-af", "silencedetect=noise=-35dB:d=1.5", "-f", "null", "-"], "ffmpeg")
    if p.returncode:
        raise RuntimeError("HT QA: ffmpeg silencedetect failed for %s: %s" % (path, p.stderr.strip()))
    starts, result = [], []
    for line in p.stderr.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.rsplit("silence_start:", 1)[1]))
        elif "silence_end:" in line:
            value = line.rsplit("silence_end:", 1)[1].split("|", 1)[0].strip()
            end = float(value)
            result.append((starts.pop(0) if starts else 0.0, end))
    return result


def _motion(path):
    with tempfile.TemporaryDirectory(prefix="ht-qa-") as directory:
        output = Path(directory) / "frames.raw"
        p = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                  "-vf", "fps=1,scale=160:90,format=gray", "-f", "rawvideo", str(output)], "ffmpeg")
        if p.returncode:
            raise RuntimeError("HT QA: ffmpeg frame extraction failed for %s: %s" % (path, p.stderr.strip()))
        raw = output.read_bytes()
        frame_size = 160 * 90
        values = []
        previous = None
        for offset in range(0, len(raw) - frame_size + 1, frame_size):
            current = torch.frombuffer(raw[offset:offset + frame_size], dtype=torch.uint8).float()
            if previous is not None:
                values.append(float((current - previous).abs().mean()))
            previous = current
        return values

class HTQAVideoCheck:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"stagnant_mae":("FLOAT",{"default":2.,"min":0}),"jump_mae":("FLOAT",{"default":14.,"min":0})}}
 RETURN_TYPES=("STRING","BOOLEAN"); RETURN_NAMES=("report","passed"); FUNCTION="check"; CATEGORY="H3-Heaton/qa"
 def check(self,video_path,stagnant_mae=2.,jump_mae=14.):
  if not video_path or not os.path.isfile(video_path): raise FileNotFoundError("HT QA: video path does not exist: %s"%video_path)
  data=_ffprobe(video_path); streams=data.get("streams",[]); audio=[s for s in streams if s.get("codec_type")=="audio"]; video=[s for s in streams if s.get("codec_type")=="video"]
  if not video: raise ValueError("HT QA: ffprobe found no video stream")
  duration=float(data.get("format",{}).get("duration",0)); silences=_silences(video_path); motion=_motion(video_path)
  stagnant=[(i + 1, value) for i, value in enumerate(motion) if value < stagnant_mae]
  jumps=[(i + 1, value) for i, value in enumerate(motion) if value > jump_mae]
  stats="min=%.4f max=%.4f mean=%.4f" % ((min(motion), max(motion), sum(motion)/len(motion)) if motion else (0., 0., 0.))
  passed=bool(audio and duration>0 and not stagnant and not jumps)
  report="video=%s duration=%.3fs audio=%s audio_streams=%d silences=%s motion_mae(%s) values=%s"%(video_path,duration,bool(audio),len(audio),silences,stats,motion)
  if not audio: report+=" | FAIL: no audio stream"
  if stagnant: report+=" | FAIL: stagnant frames=%s"%stagnant
  if jumps: report+=" | FAIL: jump frames=%s"%jumps
  return (report,passed)

class HTQALatentProbe:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"latent":("LATENT",)}}
 RETURN_TYPES=("STRING",); FUNCTION="check"; CATEGORY="H3-Heaton/qa"
 def check(self,latent): return (probe(latent["samples"]),)
NODE_CLASS_MAPPINGS={"HT_QA_VideoCheck":HTQAVideoCheck,"HT_QA_LatentProbe":HTQALatentProbe}
NODE_DISPLAY_NAME_MAPPINGS={"HT_QA_VideoCheck":"HT · QA Video Check","HT_QA_LatentProbe":"HT · QA Latent Probe"}
