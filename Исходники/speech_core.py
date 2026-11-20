"""Rule-based Russian audio, phonemes and facial dynamics. No inference runtime."""
import argparse
import ctypes as ct
import io
import json
from pathlib import Path
import re
import sysconfig
import wave

import numpy as np


class EventID(ct.Union):
    _fields_ = [("number", ct.c_int), ("name", ct.c_void_p), ("phone", ct.c_char * 8)]


class Event(ct.Structure):
    _fields_ = [("type", ct.c_int), ("uid", ct.c_uint), ("pos", ct.c_int),
                ("length", ct.c_int), ("ms", ct.c_int), ("sample", ct.c_int),
                ("user", ct.c_void_p), ("id", EventID)]


Callback = ct.CFUNCTYPE(ct.c_int, ct.POINTER(ct.c_short), ct.c_int, ct.POINTER(Event))
GROUPS = {
    "rest": ("", "m", "b", "p"), "round": tuple("ouɔʊʉɵœøw"),
    "wide": tuple("ieɪɛɨyj"), "open": tuple("aɑæ"), "reduced": tuple("ɐʌəɘɜ"),
    "teeth": tuple("fvɸβ"), "hush": (*"szʃʂʒʐɕʑʧ", "ts","tʃ","tɕ"),
}
MOUTHS = {p: name for name, phones in GROUPS.items() for p in phones}
FPS = 50


def marked_text(text):
    if not isinstance(text, str) or len(text) > 400:
        raise ValueError("Допускается до 400 знаков.")
    runs = text.strip().split("**")
    if len(runs) % 2 == 0:
        raise ValueError("Закройте выделение второй парой **.")
    plain = "".join(runs)
    if not re.search(r"[^\W_]", plain):
        raise ValueError("Введите произносимый текст.")
    emphasized = [bool(i % 2) for i, run in enumerate(runs) for _ in run]
    return plain, emphasized


def choose_answer(question):
    """Only curated responses enter speech from visitor-question mode."""
    q = " ".join(re.findall(r"[а-яa-z0-9]+", question.lower().replace("ё","е")))
    rules = (
        (r"приемник|на столе|на стол|этот прибор|этот аппарат",
         "Посмотрите на **приёмник**. Его антенна улавливает радиоволну. В ответ срабатывает реле, замыкается электрическая цепь, и мы слышим **звонок**."),
        (r"как вас зовут|кто (вы|ты|такой попов)|как тебя зовут|представь",
         "Меня зовут Александр Степанович **Попов**. Я занимался физикой и опытами с радиоволнами."),
        (r"что такое радио|радиоволн|как работает радио|без провод",
         "Представьте передатчик и приёмник. Между ними нет провода. Сигнал переносит **радиоволна**."),
        (r"привет|здравств|добрый (день|вечер)|доброе утро",
         "Здравствуйте. Меня зовут Александр Попов. Хотите узнать, как работает мой приёмник?"),
        (r"спасибо|благодар", "Благодарю за интерес к моим опытам."),
    )
    return next((answer for pattern, answer in rules if re.search(pattern,q)),
                "Об этом у меня пока нет рассказа. Спросите, как работает **приёмник**, или что такое радиоволна.")


def dynamics(samples, rate, phones, marked):
    """Fixed-reference RMS, and separate stress pulses on the same PCM clock."""
    seconds = len(samples) / rate
    centers = np.arange(int(np.ceil(seconds * FPS)) + 1) / FPS
    accent = np.zeros(len(centers))
    gain = np.ones(len(centers))
    cues = []
    timeline = [(0.0, "", False)] + phones
    for i, (start, phone, highlight) in enumerate(timeline):
        end = min(seconds, timeline[i+1][0] if i+1 < len(timeline) else seconds)
        if end <= start: continue
        bare = re.sub("[ˈˌʲː̪͡ʰ]", "", phone)
        # Palatalization may arrive as a standalone IPA event.
        pose = cues[-1][2] if phone in ("ʲ","ː") and cues else MOUTHS.get(bare,"soft")
        if cues and cues[-1][2] == pose:
            cues[-1][1] = end
        else:
            cues.append([start,end,pose])
        if "ˈ" in phone or ("ˌ" in phone and highlight):
            center = start + .45*(end-start)
            width = .11 if highlight else .08
            beat = np.exp(-.5*((centers-center)/width)**2)
            accent = np.maximum(accent, beat * (.95 if highlight else .30))
        if highlight and bare:
            # Ramp the word gain at either end to avoid clicks.
            gain = np.maximum(gain,1+.18*np.clip(np.minimum(
                (centers-start+.035)/.035,(end+.045-centers)/.045),0,1))
    signal = samples.astype(np.float64)
    if any(marked):
        signal *= np.interp(np.arange(len(signal))/rate, centers, gain)
    signal = np.clip(signal,-32767,32767).astype("<i2")
    power = np.r_[0, np.cumsum(np.square(signal.astype(np.float64)/32768))]
    idx = (centers*rate).round().astype(int)
    left = np.clip(idx-int(rate*.014),0,len(signal))
    right = np.clip(idx+int(rate*.014),0,len(signal))
    rms = np.sqrt((power[right]-power[left])/np.maximum(1,right-left))
    db = 20*np.log10(np.maximum(rms,1e-6))
    target = np.clip((db+48)/31,0,1)
    energy = np.zeros_like(target)
    level = 0.
    for i,value in enumerate(target):
        level += (.76 if value>level else .50)*(value-level)
        energy[i] = level
    return signal, dict(cues=cues, fps=FPS, energy=energy.round(4).tolist(),
                        accent=accent.round(4).tolist(), seconds=seconds)


class SpeechCore:
    """One instance per process, called serially from the synthesis worker."""
    def __init__(self):
        data = Path(sysconfig.get_path("platlib"))/"piper"
        # This installed extension also exports the plain eSpeak C ABI. Loading it
        # directly avoids importing Piper, ONNX or any trained voice model.
        self.lib = ct.CDLL(str(data/"espeakbridge.so"))
        self.lib.espeak_Initialize.argtypes = [ct.c_int,ct.c_int,ct.c_char_p,ct.c_int]
        self.lib.espeak_Synth.argtypes = [ct.c_char_p,ct.c_size_t,ct.c_uint,ct.c_int,
                                          ct.c_uint,ct.c_uint,ct.c_void_p,ct.c_void_p]
        self.lib.espeak_SetSynthCallback.argtypes = [Callback]
        self.lib.espeak_SetSynthCallback.restype = None
        self.rate = self.lib.espeak_Initialize(2,30,str(data).encode(),3)
        if self.rate <= 0 or self.lib.espeak_SetVoiceByName(b"ru") != 0:
            raise RuntimeError("Не удалось открыть формантный русский голос eSpeak NG.")
        for parameter,value in [(1,143),(2,85),(3,40),(4,39)]:
            if self.lib.espeak_SetParameter(parameter,value,0):
                raise RuntimeError("Не удалось настроить голос.")
        self.callback = Callback(self._collect)
        self.lib.espeak_SetSynthCallback(self.callback)

    def _collect(self, buf, count, events):
        if buf and count: self.pcm.extend(ct.string_at(buf,count*2))
        if len(self.pcm)>self.rate*2*25:
            self.too_long = True
            return 1
        i = 0
        while events and events[i].type:
            event = events[i]
            if event.type == 1:
                start = max(0,event.pos-1)
                self.highlight = any(self.marked[start:start+event.length])
            elif event.type == 7:
                phone = event.id.phone.decode("utf-8","ignore")
                self.phones.append((event.ms/1000,phone,self.highlight))
            i += 1
        return 0

    def synthesize_pcm(self, text):
        plain,self.marked = marked_text(text)
        self.highlight = self.too_long = False
        self.pcm = bytearray()
        self.phones = []
        utf8 = plain.encode()
        result = self.lib.espeak_Synth(utf8,len(utf8)+1,0,1,0,1,None,None)
        if self.too_long: raise ValueError("Реплика длиннее 25 секунд.")
        if result or not self.pcm or not self.phones: raise ValueError("Не удалось озвучить реплику.")
        signal,track = dynamics(np.frombuffer(self.pcm,dtype="<i2"),self.rate,self.phones,self.marked)
        return signal.tobytes(), track

    def synthesize(self, text):
        pcm, track = self.synthesize_pcm(text)
        stream = io.BytesIO()
        with wave.open(stream,"wb") as out:
            out.setparams((1,2,self.rate,0,"NONE","PCM"))
            out.writeframes(pcm)
        return stream.getvalue(), track


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--ask",action="store_true",help="Подобрать ответ на вопрос из сценария")
    parser.add_argument("--out",type=Path,required=True,help="Путь без расширения для WAV и JSON")
    args = parser.parse_args()
    wav,track = SpeechCore().synthesize(choose_answer(args.text) if args.ask else args.text)
    args.out.with_suffix(".wav").write_bytes(wav)
    args.out.with_suffix(".json").write_text(json.dumps(track,ensure_ascii=False))
