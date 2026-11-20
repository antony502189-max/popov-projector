"""Windows host: native fullscreen portrait, audio output and rotating phone QR."""
import base64
from collections import deque
import io
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from urllib.request import Request, build_opener, ProxyHandler
import wave
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def wav_bytes(pcm, sample_rate=22050):
    if sample_rate != 22050 or not pcm or len(pcm) % 2 or len(pcm) > 22050*2*120:
        raise ValueError("Invalid projector audio")
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return target.getvalue()


def pose_at(track, elapsed):
    for start, end, pose in track.get("cues", []):
        if start <= elapsed < end:
            return pose
    return "rest"


class Host:
    def __init__(self, api):
        self.api = api
        self.key = os.environ["STATION_KEY"]
        self.http = build_opener(ProxyHandler({}))
        self.inbox = queue.Queue()
        self.acks = queue.Queue()
        self.stop = threading.Event()
        self.seen = set()
        self.states = {}
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def post(self, path, data):
        request = Request(self.api.bridge_url+path, data=json.dumps(data).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer "+self.key})
        with self.http.open(request, timeout=5) as response:
            raw = response.read(12*1024*1024+1)
            if len(raw) > 12*1024*1024:
                raise ValueError("Station response too large")
            return json.loads(raw)

    def state(self, key, value):
        with self.lock:
            self.states[key] = value
        self.acks.put((key, value))

    def poll(self):
        pending = deque()
        while not self.stop.is_set():
            try:
                while not self.acks.empty():
                    pending.append(self.acks.get_nowait())
                while pending:
                    key, state = pending[0]
                    try:
                        self.post("/station/jobs/"+key, {"state": state})
                    except Exception as exc:
                        # A server-expired job must not block every future acknowledgement.
                        if getattr(exc, "code", 0) in (404, 409):
                            pending.popleft()
                            continue
                        raise
                    pending.popleft()
                data = self.post("/station/poll", {"ready": True})
                self.inbox.put(("qr", (data["url"], data["expires_at"])))
                job = data.get("job")
                if job and job["id"] not in self.seen:
                    self.seen.add(job["id"])
                    self.state(job["id"], "prepared")
                    self.inbox.put(("play", job))
            except Exception as exc:
                self.inbox.put(("error", "Нет связи с сервером: "+type(exc).__name__))
            self.stop.wait(.4)


class Projector:
    def __init__(self, host):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.host = host
        self.root = tk.Tk()
        self.root.title("Попов · Сервер и проекция")
        self.root.configure(bg="black")
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{max(480, screen_w//2)}x{max(550, screen_h-85)}+0+0")
        self.root.minsize(450, 500)
        self.root.bind("<F11>", self.fullscreen)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        toolbar = tk.Frame(self.root, bg="#172b23", padx=12, pady=8)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="ПОПОВ · ПРОЕКЦИЯ", bg="#172b23", fg="#dbe4d4", font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Button(toolbar, text="Открыть чат ↗", command=self.open_chat, bg="#e5ecdc", fg="#22422e", relief="flat", padx=12, pady=5).pack(side="right")
        tk.Button(toolbar, text="Полный экран", command=self.fullscreen, bg="#243e31", fg="#edf1e8", relief="flat", padx=10, pady=5).pack(side="right", padx=7)
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.images = {name: Image.open(ROOT/"Ресурсы"/(name+".png")).convert("RGBA")
                       for name in ("rest", "open", "soft", "round", "wide", "teeth", "hush", "reduced")}
        self.cached = {}
        self.current_size = None
        self.current_pose = None
        self.actor = self.canvas.create_image(0, 0)
        self.qr_item = self.canvas.create_image(0, 0, anchor="s")
        self.caption = self.canvas.create_text(0, 0, fill="white", font=("Segoe UI", 12), anchor="s")
        self.hint = self.canvas.create_text(0, 0, fill="#a9b6af", font=("Segoe UI", 10), anchor="nw")
        self.answer = self.canvas.create_text(0, 0, fill="#e7eee9", font=("Segoe UI", 14), anchor="center")
        self.qr_url, self.qr_until, self.qr_image = "", 0, None
        self.track, self.began, self.current = None, 0, None
        self.status = "Запускается сервер…"
        self.closing = False
        self.host.thread.start()
        self.root.after(20, self.frame)

    def fullscreen(self, event=None):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def open_chat(self):
        if self.qr_url and time.time() < self.qr_until:
            webbrowser.open(self.qr_url, new=1)
        else:
            self.status = "Ожидаем свежий QR — повторите через секунду"

    def show_qr(self, url, until):
        if url != self.qr_url:
            import qrcode
            code = qrcode.QRCode(box_size=5, border=4)
            code.add_data(url); code.make(fit=True)
            self.qr_image = code.make_image(fill_color="black", back_color="white").convert("RGB")
            self.qr_url = url
            self.qr_photo = None
        self.qr_until = until

    def play(self, job):
        key = job["id"]
        if self.current is not None or job["expiry"] <= time.time():
            self.host.state(key, "failed")
            return
        try:
            audio = wav_bytes(base64.b64decode(job["pcm"], validate=True), job.get("sample_rate"))
        except Exception:
            self.host.state(key, "failed")
            return
        self.current = key
        self.track = job["track"]
        self.began = time.perf_counter()
        self.status = "Попов говорит"
        self.canvas.itemconfigure(self.answer, text=job["text"])
        def sound():
            import winsound
            try:
                # Synchronous WinMM playback keeps the WAV bytes alive until audio ends.
                self.host.state(key, "started")
                winsound.PlaySound(audio, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
                self.host.state(key, "finished" if not self.closing else "cancelled")
            except Exception:
                self.host.state(key, "failed")
            self.host.inbox.put(("ended", key))
        threading.Thread(target=sound, daemon=True).start()

    def frame(self):
        if self.closing:
            return
        for _ in range(30):
            try:
                kind, data = self.host.inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "qr":
                self.show_qr(*data)
                if not self.current:
                    self.status = "Готов к разговору"
            elif kind == "play":
                self.play(data)
            elif kind == "ended" and data == self.current:
                self.current = self.track = None
                self.status = "Готов к разговору"
            elif kind == "error":
                self.status = data
        width, height = max(420, self.canvas.winfo_width()), max(440, self.canvas.winfo_height())
        portrait_h = int(height*.85)
        portrait_w = int(portrait_h*self.images["rest"].width/self.images["rest"].height)
        size = (portrait_w, portrait_h)
        if self.current_size != size:
            self.cached.clear()
            self.current_size = size
            self.current_pose = None
        elapsed = time.perf_counter()-self.began
        pose = pose_at(self.track, elapsed) if self.track else "rest"
        if pose not in self.images:
            pose = "soft"
        if pose not in self.cached:
            self.cached[pose] = self.ImageTk.PhotoImage(self.images[pose].resize(size, self.Image.Resampling.LANCZOS))
        self.canvas.itemconfigure(self.actor, image=self.cached[pose])
        self.canvas.coords(self.actor, width*.5, height*.45)
        qr_size = min(210, max(160, int(height*.23)))
        if self.qr_image is not None and time.time() < self.qr_until:
            if not getattr(self, "qr_photo", None) or self.qr_photo.width() != qr_size:
                self.qr_photo = self.ImageTk.PhotoImage(self.qr_image.resize((qr_size, qr_size), self.Image.Resampling.NEAREST))
            self.canvas.itemconfigure(self.qr_item, image=self.qr_photo, state="normal")
            self.canvas.coords(self.qr_item, width*.5, height-49)
            self.canvas.itemconfigure(self.caption, text="Спросить Попова · QR обновляется каждые 2 минуты")
        else:
            self.canvas.itemconfigure(self.qr_item, state="hidden")
            self.canvas.itemconfigure(self.caption, text="Ожидаем свежий QR…")
        self.canvas.coords(self.caption, width*.5, height-23)
        self.canvas.coords(self.hint, 18, 16)
        self.canvas.itemconfigure(self.hint, text=self.status+"\nF11 — полный экран · Esc — выйти из полного экрана\n"+self.host.api.public_url)
        self.canvas.coords(self.answer, width*.5, height*.60)
        self.canvas.itemconfigure(self.answer, width=int(width*.85), state="hidden" if width < 900 else "normal")
        self.root.after(33, self.frame)

    def close(self):
        self.closing = True
        import winsound
        winsound.PlaySound(None, 0)
        self.host.stop.set()
        self.host.thread.join(timeout=2)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    if sys.platform != "win32":
        raise RuntimeError("This renderer requires Windows")
    from .configure_local import configure
    from .local import LocalServer
    configure(ROOT)
    api = LocalServer(ROOT)
    try:
        Projector(Host(api)).run()
    finally:
        api.close()


if __name__ == "__main__":
    main()
