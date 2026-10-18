#!/usr/bin/env python3
"""Title in, finished vertical video out (Burmese story, AI images, AI voice, subtitles).

Steps: story (Claude) -> scenes -> images (Pollinations) -> voice (edge-tts / gTTS /
optional ElevenLabs clone) -> subtitles -> video (ffmpeg).

Usage:
  python make_video.py "ခေါင်းစဉ်"
  python make_video.py "ခေါင်းစဉ်" --genre "သရဲ" --scenes 7 --cast မြမြ,အဘွား --voice female
  python make_video.py "ခေါင်းစဉ်" --story story.json     (skip Claude, use your own story file)

Needs: pkg install ffmpeg python ; pip install requests edge-tts gtts
Settings go in config.json next to this script (see config.example.json).
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "config.json"
CAST_FILE = BASE / "characters.json"
VOICES = {"female": "my-MM-NilarNeural", "male": "my-MM-ThihaNeural"}

DEFAULT_CAST = [
    {"id": "mya", "name": "မြမြ", "gender": "female",
     "look": "young Burmese woman, long black hair, wearing a green longyi and a white blouse"},
    {"id": "koko", "name": "ကိုကို", "gender": "male",
     "look": "young Burmese man, short black hair, wearing a white shirt and a dark longyi"},
    {"id": "granny", "name": "အဘွား", "gender": "female",
     "look": "elderly Burmese grandmother, grey hair in a bun, kind wrinkled face, faded maroon longyi"},
    {"id": "elder", "name": "ဦးကြီး", "gender": "male",
     "look": "elderly Burmese man, white hair, thin face, traditional taikpon jacket and gaung baung headcloth"},
    {"id": "child", "name": "ကလေး", "gender": "female",
     "look": "small Burmese child, round face, bright eyes, simple yellow dress"},
]

SYSTEM = """You write short, gripping vertical-video stories (about 45-60 seconds when read aloud) in Burmese (Myanmar Unicode, never Zawgyi).
Return ONLY one JSON object, no markdown, in exactly this shape:
{"title":"Burmese title","hashtags":["#tag"],"style":"one English line describing the shared visual style",
"characters":[{"id":"short_ascii_id","name":"Burmese name","gender":"male|female","look":"English visual description of face, hair, clothes"}],
"scenes":[{"speaker":"narrator or a character id","narration":"Burmese text read aloud in this scene","characters":["ids visible in the scene"],"image_prompt":"English description of what the image shows"}]}
Rules: the first scene is a strong hook; each narration is 1-2 short sentences (max about 90 characters); the story is coherent with a clear ending; no gore, no sexual content, no real people; image_prompt describes the scene, contains no text in the image, and copies each visible character's look verbatim."""


def setting(name, default=None, required=False):
    if os.environ.get(name):
        return os.environ[name]
    if CONFIG.exists():
        value = json.loads(CONFIG.read_text(encoding="utf-8")).get(name)
        if value not in (None, ""):
            return value
    if required:
        sys.exit(f"Missing setting {name}: add it to config.json or set it as an env var")
    return default


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"Command failed: {' '.join(map(str, cmd[:3]))} ...\n{r.stderr[-800:]}")
    return r.stdout


# ---------- cast and story ----------
def load_cast():
    if not CAST_FILE.exists():
        CAST_FILE.write_text(json.dumps(DEFAULT_CAST, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(CAST_FILE.read_text(encoding="utf-8"))


def pick_cast(names):
    library, chosen = load_cast(), []
    for n in names:
        n = n.strip()
        found = next((c for c in library if n in (c["id"], c["name"])), None)
        if not found:
            sys.exit(f"Character '{n}' not found. Available: " + ", ".join(c["name"] for c in library))
        chosen.append(found)
    return chosen


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON found in reply")
    return json.loads(text[start:end + 1])


def check_story(s):
    if not isinstance(s.get("scenes"), list) or not s["scenes"]:
        raise ValueError("story has no scenes")
    s.setdefault("title", "")
    s.setdefault("hashtags", [])
    s.setdefault("style", "cinematic, dramatic lighting, highly detailed")
    s.setdefault("characters", [])
    for sc in s["scenes"]:
        if not sc.get("narration") or not sc.get("image_prompt"):
            raise ValueError("a scene is missing narration or image_prompt")
        sc.setdefault("speaker", "narrator")
        sc.setdefault("characters", [])
    return s


def make_story(title, genre, n_scenes, cast):
    key = setting("ANTHROPIC_API_KEY", required=True)
    if cast:
        who = "Use ONLY these characters (keep ids, names, genders; copy 'look' verbatim): " + json.dumps(cast, ensure_ascii=False)
    else:
        who = "Invent 1-3 characters (id, Burmese name, gender, English look)."
    user = f"Title: {title}\nGenre/tone: {genre}\nNumber of scenes: {n_scenes}\n{who}"
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": setting("story_model", "claude-haiku-5-5"), "max_tokens": 4000,
              "system": SYSTEM, "messages": [{"role": "user", "content": user}]},
        timeout=180,
    )
    if r.status_code != 200:
        sys.exit(f"Claude API error {r.status_code}: {r.text[:300]}")
    return check_story(parse_json(r.json()["content"][0]["text"]))


# ---------- images ----------
def make_placeholder(out, w, h):
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x1b1f3b:s={w}x{h}", "-frames:v", "1", str(out)])


def gen_image(prompt, seed, out, w, h):
    url = (f"https://image.pollinations.ai/prompt/{quote(prompt)}"
           f"?width={w}&height={h}&seed={seed}&nologo=true&model=flux")
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=240)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") and len(r.content) > 5000:
                out.write_bytes(r.content)
                return True
            print(f"   image try {attempt + 1} failed ({r.status_code})")
        except requests.RequestException as e:
            print(f"   image try {attempt + 1} error: {e}")
        time.sleep(20)
    return False


def full_prompt(story, sc, looks):
    who = "; ".join(looks[c] for c in sc["characters"] if c in looks)
    parts = [sc["image_prompt"].rstrip(".")]
    if who:
        parts.append("Characters: " + who)
    parts.append("Style: " + story["style"])
    parts.append("vertical 9:16 composition, no text, no watermark")
    return ". ".join(parts)[:900]


# ---------- voice ----------
def speak_eleven(text, out):
    key, vid = setting("ELEVENLABS_API_KEY"), setting("clone_voice_id")
    if not key or not vid:
        return False
    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json={"text": text, "model_id": setting("clone_model_id", "eleven_v3")},
            timeout=180,
        )
        if r.status_code == 200 and len(r.content) > 1000:
            out.write_bytes(r.content)
            return True
        print("   ElevenLabs error:", r.status_code, r.text[:200])
    except requests.RequestException as e:
        print("   ElevenLabs error:", e)
    return False


def speak(text, voice, out, clone=False):
    if clone and speak_eleven(text, out):
        return
    try:
        import edge_tts

        async def go():
            await edge_tts.Communicate(text, voice, rate=setting("voice_rate", "-5%")).save(str(out))

        asyncio.run(go())
        if out.exists() and out.stat().st_size > 1000:
            return
    except ImportError:
        pass
    except Exception as e:
        print("   edge-tts failed:", e)
    try:
        from gtts import gTTS
        gTTS(text=text, lang="my").save(str(out))
    except ImportError:
        sys.exit("No voice engine installed. Run: pip install edge-tts   (or: pip install gtts)")


def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


# ---------- subtitles ----------
def cut_ok(t, i):
    c = t[i]
    return 0x1000 <= ord(c) <= 0x1021 and t[i - 1] != "\u1039" and (i + 1 >= len(t) or t[i + 1] not in "\u103a\u1039")


def split_long(p, maxlen):
    out = []
    while len(p) > maxlen:
        cut = p.rfind(" ", 8, maxlen + 1)
        if cut == -1:
            cut = next((j for j in range(maxlen, 7, -1) if cut_ok(p, j)), maxlen)
        out.append(p[:cut].strip())
        p = p[cut:].strip()
    if p:
        out.append(p)
    return out


def chunks(text, maxlen=24):
    sentences = [s.strip() for s in re.split(r"(?<=။)", text.replace("\n", " ")) if s.strip()]
    return [c for s in sentences for c in split_long(s, maxlen)]


def ass_time(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def make_ass(text, dur, path, w, h, font):
    parts = chunks(text)
    total = sum(len(c) for c in parts) or 1
    t, lines = 0.0, []
    for c in parts:
        d = dur * len(c) / total
        lines.append(f"Dialogue: 0,{ass_time(t)},{ass_time(t + d)},Main,,0,0,0,,{c}")
        t += d
    head = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Main,{font},{int(h * 0.038)},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,40,40,{int(h * 0.12)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


def find_font():
    f = setting("font_file")
    if f and Path(f).exists():
        return Path(f)
    for p in ["/system/fonts/NotoSansMyanmar-Regular.ttf", "/system/fonts/NotoSansMyanmarUI-Regular.ttf",
              "/system/fonts/Padauk-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf"]:
        if Path(p).exists():
            return Path(p)
    for p in sorted(Path("/usr/share/fonts").glob("**/NotoSansMyanmar*-Regular.ttf")):
        return p
    d = Path("/system/fonts")
    if d.exists():
        for p in sorted(d.iterdir()):
            if any(k in p.name.lower() for k in ("myanmar", "padauk", "burm")):
                return p
    sys.exit("Myanmar font not found. Put NotoSansMyanmar-Regular.ttf next to this script and set "
             '"font_file" in config.json')


def font_family(path):
    try:
        out = subprocess.run(["fc-scan", "--format", "%{family}\n", str(path)], capture_output=True, text=True).stdout
        name = out.strip().split("\n")[0].split(",")[0]
        if name:
            return name
    except Exception:
        pass
    return setting("font_name", "Noto Sans Myanmar")


# ---------- video ----------
def build_scene(i, workdir, dur, w, h, fontsdir):
    fps, frames = 30, int(dur * 30) + 1
    if i % 2 == 0:
        z = "min(zoom+0.0007,1.18)"
    else:
        z = "if(eq(on,1),1.18,max(zoom-0.0007,1.0))"
    vf = (f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
          f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps},"
          f"subtitles=sub_{i:02d}.ass:fontsdir='{fontsdir}'")
    run(["ffmpeg", "-y", "-i", f"img_{i:02d}.jpg", "-i", f"voice_{i:02d}.mp3", "-vf", vf, "-af", "apad",
         "-t", f"{dur:.2f}", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
         f"scene_{i:02d}.mp4"], cwd=workdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("title", nargs="?")
    ap.add_argument("--genre", default="အံ့ဩစရာ twist ending ပါတဲ့ ကြောက်စရာဇာတ်လမ်း")
    ap.add_argument("--scenes", type=int, default=7)
    ap.add_argument("--cast", help="comma separated names/ids from characters.json")
    ap.add_argument("--story", help="use this story.json instead of asking Claude")
    ap.add_argument("--voice", choices=["male", "female"], default="male", help="narrator voice")
    ap.add_argument("--out", help="output folder name (re-use to resume)")
    a = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found. Run: pkg install ffmpeg")

    title = a.title or input("ခေါင်းစဉ်: ").strip()
    if not title and not a.story:
        sys.exit("Title is empty")

    w, h = (int(x) for x in str(setting("size", "720x1280")).split("x"))
    out = BASE / "output" / (a.out or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    story_file = out / "story.json"

    if a.story:
        story = check_story(json.loads(Path(a.story).read_text(encoding="utf-8")))
    elif story_file.exists():
        story = check_story(json.loads(story_file.read_text(encoding="utf-8")))
    else:
        print("1/4 ဇာတ်လမ်းရေးနေပါတယ်...")
        cast = pick_cast(a.cast.split(",")) if a.cast else None
        story = make_story(title, a.genre, a.scenes, cast)
    story.setdefault("seed", int(time.time()) % 100000)
    story_file.write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")

    looks = {c["id"]: c.get("look", "") for c in story["characters"]}
    genders = {c["id"]: c.get("gender", "female") for c in story["characters"]}
    font = find_font()
    family, fontsdir = font_family(font), str(font.parent)
    n = len(story["scenes"])
    print(f"   {n} scenes: {story.get('title') or title}")

    print("2/4 ပုံတွေ ဖန်တီးနေပါတယ် (တစ်ပုံကို ၂၀ စက္ကန့်ခန့်)...")
    last = 0.0
    for i, sc in enumerate(story["scenes"], 1):
        img = out / f"img_{i:02d}.jpg"
        if img.exists():
            continue
        wait = float(setting("image_delay", 16)) - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        print(f"   scene {i}/{n}")
        ok = gen_image(full_prompt(story, sc, looks), story["seed"], img, w, h)
        last = time.time()
        if not ok:
            print("   ပုံမရလို့ အစားထိုးပုံ သုံးပါမယ်")
            make_placeholder(img, w, h)

    print("3/4 အသံနဲ့ subtitle ပြင်ဆင်နေပါတယ်...")
    for i, sc in enumerate(story["scenes"], 1):
        voice_file = out / f"voice_{i:02d}.mp3"
        speaker = sc["speaker"]
        narrator = speaker == "narrator" or speaker not in genders
        voice = VOICES[a.voice] if narrator else VOICES.get(genders[speaker], VOICES["female"])
        if not voice_file.exists():
            speak(sc["narration"], voice, voice_file, clone=narrator)
        dur = duration(voice_file) + 0.5
        make_ass(sc["narration"], dur, out / f"sub_{i:02d}.ass", w, h, family)
        sc["_dur"] = dur

    print("4/4 ဗီဒီယို ပေါင်းစပ်နေပါတယ်...")
    for i, sc in enumerate(story["scenes"], 1):
        build_scene(i, out, sc["_dur"], w, h, fontsdir)
    (out / "list.txt").write_text("".join(f"file 'scene_{i:02d}.mp4'\n" for i in range(1, n + 1)), encoding="utf-8")
    raw = out / "raw.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "raw.mp4"], cwd=out)

    final = out / "video.mp4"
    music = setting("music_file")
    if music and Path(music).exists():
        run(["ffmpeg", "-y", "-i", "raw.mp4", "-stream_loop", "-1", "-i", str(music), "-filter_complex",
             "[1:a]volume=0.10[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
             "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "video.mp4"], cwd=out)
        raw.unlink()
    else:
        raw.rename(final)

    tags = " ".join(story.get("hashtags", []))
    (out / "caption.txt").write_text(f"{story.get('title') or title} {tags}".strip() + "\n", encoding="utf-8")
    print(f"\nပြီးပါပြီ: {final}")
    print(f"Caption: {out / 'caption.txt'}")


if __name__ == "__main__":
    main()
