# MOIRAI Local VOD AI Analyzer (prototype)
# Usage: python vod_ai.py "VIDEO.mp4"
# Optional YouTube input requires yt-dlp installed: python vod_ai.py "https://youtu.be/..."
#
# This first prototype uses audio/transcription to propose event candidates.
# It does NOT pretend audio alone can reliably identify visual saves/blocks/shots.

import sys, os, re, csv, json, subprocess, tempfile, hashlib
from pathlib import Path

def get_video(src, work):
    if src.startswith(("http://","https://")):
        cache=Path.home()/".moirai-vod-cache"
        cache.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
        cached=cache/f"{key}.mp4"
        if cached.exists() and cached.stat().st_size>1024*1024:
            print(f"Using cached VOD: {cached}")
            return str(cached)
        out=str(cache/f"{key}.%(ext)s")
        print(f"Downloading VOD once; future runs will reuse: {cached}")
        subprocess.run(["yt-dlp","-f","bv*+ba/b","--merge-output-format","mp4","-o",out,src],check=True)
        if cached.exists(): return str(cached)
        files=list(cache.glob(f"{key}.*"))
        if not files: raise RuntimeError("yt-dlp produced no video")
        return str(files[0])
    return src

def extract_audio(video, work):
    wav=str(Path(work)/"audio.wav")
    subprocess.run(["ffmpeg","-y","-i",video,"-vn","-ac","1","-ar","16000",wav],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return wav

def transcribe(wav):
    try:
        from faster_whisper import WhisperModel
        import numpy as np
    except ImportError:
        raise SystemExit("Install first: pip install faster-whisper numpy")
    # Decode WAV with Python instead of PyAV. This avoids PyAV API-version
    # incompatibilities (e.g. metadata_errors) on Windows.
    import wave
    with wave.open(wav,"rb") as wf:
        channels=wf.getnchannels()
        width=wf.getsampwidth()
        rate=wf.getframerate()
        raw=wf.readframes(wf.getnframes())
    if width != 2:
        raise RuntimeError(f"Expected 16-bit WAV, got {width*8}-bit")
    audio=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    if channels>1:
        audio=audio.reshape(-1,channels).mean(axis=1)
    if rate != 16000:
        raise RuntimeError(f"Expected 16000 Hz WAV, got {rate}")
    model=WhisperModel("small",device="cpu",compute_type="int8")
    print("Whisper transcription started...")
    segs,info=model.transcribe(audio,vad_filter=True)
    return [{"start":x.start,"end":x.end,"text":x.text.strip()} for x in segs]

KEYWORDS={
 "GOAL":["goal","scores","scored","tor","treffer"],
 "SAVE":["save","saved","parade","gehalten","hält"],
 "ASSIST":["assist","vorlage"],
 "BLOCK":["block","blocked","geblockt"],
 "SHOT":["shot","shoot","schuss","schießt"],
 "CONCEDED":["conceded","gegentreffer"]
}
def candidates(segs):
    out=[]
    for s in segs:
        t=s["text"].lower()
        for typ,words in KEYWORDS.items():
            if any(re.search(r"(?<!\w)"+re.escape(w)+r"(?!\w)",t) for w in words):
                out.append({"timestamp":round(s["start"],1),"type":typ,"transcript":s["text"],"confidence":"candidate"})
    return out

def stamp(sec):
    sec=int(sec); return f"{sec//60:02d}:{sec%60:02d}"

def main():
    if len(sys.argv)<2: raise SystemExit('Usage: python vod_ai.py "VIDEO.mp4 or YouTube URL"')
    with tempfile.TemporaryDirectory() as work:
        video=get_video(sys.argv[1],work)
        wav=extract_audio(video,work)
        segs=transcribe(wav)
        events=candidates(segs)
        Path("moirai-transcript.json").write_text(json.dumps(segs,ensure_ascii=False,indent=2),encoding="utf-8")
        with open("moirai-ai-candidates.csv","w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f,delimiter=";")
            w.writerow(["TIMESTAMP","TYPE","CONFIDENCE","TRANSCRIPT"])
            for e in events:w.writerow([stamp(e["timestamp"]),e["type"],e["confidence"],e["transcript"]])
        print(f"Done: {len(segs)} transcript segments, {len(events)} event candidates")
        print("Created moirai-ai-candidates.csv and moirai-transcript.json")

if __name__=="__main__": main()
