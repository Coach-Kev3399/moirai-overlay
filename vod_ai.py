# MOIRAI Local VOD AI Analyzer V3 (continuous visual scan + scene segmentation)
# Usage: python vod_ai.py "VIDEO.mp4"
# Optional YouTube input requires yt-dlp installed: python vod_ai.py "https://youtu.be/..."
#
# This first prototype uses audio/transcription to propose event candidates.
# It does NOT pretend audio alone can reliably identify visual saves/blocks/shots.

import sys, os, re, csv, json, subprocess, tempfile, hashlib, math
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

def video_duration(video):
    p=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",video],
                     capture_output=True,text=True,check=True)
    return float(p.stdout.strip())

def extract_frame(video, sec, out):
    subprocess.run(["ffmpeg","-y","-ss",f"{max(0,sec):.2f}","-i",video,"-frames:v","1","-q:v","2",str(out)],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)


def continuous_visual_scan(video, out_dir, interval=2.0):
    """Scan the full VOD cheaply and flag major visual transitions for dense review."""
    try:
        from PIL import Image, ImageStat, ImageChops
    except ImportError:
        raise SystemExit("V3 needs Pillow once: pip install pillow")
    out_dir=Path(out_dir); scan_dir=out_dir/"scan"; event_dir=out_dir/"transitions"
    scan_dir.mkdir(parents=True,exist_ok=True); event_dir.mkdir(parents=True,exist_ok=True)
    duration=video_duration(video)
    samples=[]; previous=None
    total=max(1,math.ceil(duration/interval))
    print(f"Continuous visual scan: ~{total} samples every {interval:g}s...")
    i=0; t=0.0
    while t<duration:
        tmp=scan_dir/f"{i:05d}.jpg"
        extract_frame(video,t,tmp)
        with Image.open(tmp) as im:
            thumb=im.convert("L").resize((160,90))
            brightness=ImageStat.Stat(thumb).mean[0]
            change=0.0 if previous is None else ImageStat.Stat(ImageChops.difference(thumb,previous)).mean[0]
            previous=thumb.copy()
        samples.append({"time":round(t,2),"timestamp":stamp(t),"brightness":round(brightness,2),"change":round(change,2)})
        tmp.unlink(missing_ok=True)
        i+=1; t+=interval
        if i%100==0: print(f"  scanned {min(t,duration):.0f}/{duration:.0f}s")
    # Dynamic threshold: scene/UI transitions should stand out from ordinary play.
    changes=sorted(x["change"] for x in samples[1:])
    p90=changes[int(.90*(len(changes)-1))] if changes else 0
    threshold=max(18.0,p90*1.35)
    raw=[x for x in samples if x["change"]>=threshold]
    # Merge nearby detections into one transition.
    transitions=[]
    for x in raw:
        if not transitions or x["time"]-transitions[-1]["time"]>6:
            transitions.append(dict(x))
        elif x["change"]>transitions[-1]["change"]:
            transitions[-1]=dict(x)
    # Dense evidence around each transition.
    evidence=[]
    for idx,x in enumerate(transitions,1):
        for delta in (-6,-4,-2,0,2,4,6):
            tt=min(max(0,x["time"]+delta),max(0,duration-.1))
            name=f"transition_{idx:03d}_{stamp(tt).replace(':','-')}_{delta:+d}s.jpg"
            extract_frame(video,tt,event_dir/name)
            evidence.append({"transition_id":idx,"time":round(tt,2),"timestamp":stamp(tt),
                             "offset":delta,"file":f"transitions/{name}","change":x["change"]})
    result={"interval_seconds":interval,"threshold":round(threshold,2),
            "samples":samples,"transitions":transitions,"evidence":evidence}
    (out_dir/"visual-scan.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    # A compact CSV makes the detected boundaries easy to inspect.
    with open(out_dir/"transitions.csv","w",newline="",encoding="utf-8-sig") as fh:
        w=csv.writer(fh,delimiter=";"); w.writerow(["TIMESTAMP","SECONDS","CHANGE"])
        for x in transitions:w.writerow([x["timestamp"],x["time"],x["change"]])
    return result

def visual_evidence(video, events, out_dir):
    """Extract context frames around audio candidates plus periodic overview frames."""
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    duration=video_duration(video)
    manifest=[]
    wanted=[]
    # Candidate windows: before / at / after. This gives visual context rather than one-frame guesses.
    for i,e in enumerate(events,1):
        for delta,label in [(-4,"before"),(-2,"pre"),(0,"event"),(2,"post"),(4,"after")]:
            t=min(max(0,float(e["timestamp"])+delta),max(0,duration-.1))
            wanted.append((t,f"candidate_{i:03d}_{e['type'].lower()}_{label}",i,e["type"],e["transcript"]))
    # Sparse overview frames help us later learn HUD/game boundaries even when audio says nothing.
    t=0.0
    while t<duration:
        wanted.append((t,"overview",None,"OVERVIEW",""))
        t+=30.0
    seen=set()
    for n,(t,label,cid,typ,transcript) in enumerate(wanted,1):
        key=(round(t,1),label)
        if key in seen: continue
        seen.add(key)
        name=f"{n:04d}_{stamp(t).replace(':','-')}_{label}.jpg"
        path=out_dir/name
        extract_frame(video,t,path)
        manifest.append({"time":round(t,2),"timestamp":stamp(t),"file":name,
                         "candidate_id":cid,"candidate_type":typ,"transcript":transcript})
    (out_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest

def stamp(sec):
    sec=int(sec); return f"{sec//60:02d}:{sec%60:02d}"

def main():
    if len(sys.argv)<2: raise SystemExit('Usage: python vod_ai.py "VIDEO.mp4 or YouTube URL"')
    with tempfile.TemporaryDirectory() as work:
        video=get_video(sys.argv[1],work)
        wav=extract_audio(video,work)
        segs=transcribe(wav)
        events=candidates(segs)
        scan=continuous_visual_scan(video,"moirai-v3-visual-scan",2.0)
        print(f"Detected {len(scan['transitions'])} major visual transitions.")
        print(f"Extracting visual evidence around {len(events)} audio candidates...")
        frames=visual_evidence(video,events,"moirai-visual-evidence")
        Path("moirai-transcript.json").write_text(json.dumps(segs,ensure_ascii=False,indent=2),encoding="utf-8")
        with open("moirai-ai-candidates.csv","w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f,delimiter=";")
            w.writerow(["TIMESTAMP","TYPE","CONFIDENCE","TRANSCRIPT"])
            for e in events:w.writerow([stamp(e["timestamp"]),e["type"],e["confidence"],e["transcript"]])
        print(f"Done: {len(segs)} transcript segments, {len(events)} audio candidates, {len(scan['transitions'])} visual transitions, {len(frames)} audio evidence frames")
        print("Created moirai-ai-candidates.csv, moirai-transcript.json, and moirai-visual-evidence/")
        print("V3 created moirai-v3-visual-scan/ with transitions.csv, visual-scan.json and dense transition frames.")
        print("Next: zip moirai-v3-visual-scan and upload it for calibration.")

if __name__=="__main__": main()
