import { AnimatePresence, motion } from "framer-motion";
import { AudioLines, Play, RotateCcw, Square, Zap } from "lucide-react";
import clsx from "clsx";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ProgressDots } from "../components/Bits";
import { DirectorNote } from "../components/DirectorNote";
import { LevelMeter } from "../components/Meters";
import { OnAir, RecordButton } from "../components/RecordButton";
import { ScriptLine, ScriptToggle } from "../components/Script";
import { useRecorder } from "../hooks/useRecorder";
import { api, LANG_GLYPH, LANG_NAME, type Baseline, type Card, type Entry, type ScoreResult } from "../lib/api";
import { describeMelody } from "../lib/melody";
import { progress } from "../lib/progress";
import { sfx } from "../lib/sound";
import { Result } from "./Result";

export interface Take {
  blob: Blob;
  start: number; // scene time when the recording began
  score: number;
}

type Step = "listen" | "turn" | "countdown" | "recording" | "scoring" | "result";

const LEAD = 0.6;   // scene time played before the line while dubbing
const GRACE = 1.0;  // recording keeps going this long after the line ends
const PAD = 0.3;    // scene time around the line when listening

export function Booth({ entry, baseline, startIndex = 0, onTake, onStreak, onFinish }: {
  entry: Entry; baseline: Baseline | null; startIndex?: number;
  onTake: (line: number, take: Take) => void; onStreak: (dub: boolean) => void; onFinish: () => void;
}) {
  const lines = useMemo(() => entry.lines.filter((l): l is Card & { shape: NonNullable<Card["shape"]> } => l.playable && !!l.shape), [entry]);
  const [index, setIndex] = useState(Math.min(startIndex, Math.max(0, lines.length - 1)));
  const line = lines[index];
  const [step, setStep] = useState<Step>("listen");
  const [listened, setListened] = useState(false);
  const [needsTap, setNeedsTap] = useState(false);
  const [melodyT, setMelodyT] = useState(0);
  const [count, setCount] = useState(3);
  const [roman, setRoman] = useState(() => localStorage.getItem("ddg.roman") === "1");
  const [result, setResult] = useState<{ r: ScoreResult; previous: number | null; url: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const rec = useRecorder(30);
  const video = useRef<HTMLVideoElement>(null);
  const voiceEl = useRef<HTMLAudioElement>(null);
  const raf = useRef(0);
  const stopAt = useRef<number | null>(null);
  const recStart = useRef(0);

  useEffect(() => localStorage.setItem("ddg.roman", roman ? "1" : "0"), [roman]);
  useEffect(() => {
    if (line) progress.setLast({ scene: entry.id, title: entry.title, line: index, total: lines.length });
  }, [entry, index, line, lines.length]);

  const lineZero = line ? line.start + line.onset : 0; // scene time of the melody line's t = 0

  /** Play part of the scene; `muted` while dubbing. Tracks the melody time for drawing. */
  const playScene = useCallback(async (from: number, until: number, muted: boolean) => {
    const v = video.current;
    if (!v) return false;
    voiceEl.current?.pause();                    // never over the example reading
    cancelAnimationFrame(raf.current);
    v.muted = muted;
    v.currentTime = Math.max(0, from);
    stopAt.current = until;
    const tick = () => {
      setMelodyT(v.currentTime - lineZero);
      if (stopAt.current !== null && v.currentTime >= stopAt.current) {
        v.pause();
        stopAt.current = null;
        v.dispatchEvent(new Event("segmentend"));
        return;
      }
      raf.current = requestAnimationFrame(tick);
    };
    try {
      await v.play();
    } catch {
      return false;
    }
    raf.current = requestAnimationFrame(tick);
    return true;
  }, [lineZero]);

  const listen = useCallback(async () => {
    if (!line) return;
    setStep((s) => (s === "result" ? "turn" : s === "turn" ? "turn" : "listen"));
    const ok = await playScene(line.start - PAD, line.end + PAD, false);
    setNeedsTap(!ok);
  }, [line, playScene]);

  // a new line starts by listening to the original
  useEffect(() => {
    setStep("listen");
    setListened(false);
    setResult(null);
    setError(null);
    setMelodyT(0);
    const t = setTimeout(listen, 350);
    return () => clearTimeout(t);
  }, [index]); // only when the line changes; `listen` is recreated with it

  // free the previous take's object URL when a new result replaces it
  useEffect(() => () => { if (result) URL.revokeObjectURL(result.url); }, [result]);

  useEffect(() => {
    const v = video.current;
    if (!v) return;
    const onEnd = () => setListened(true);
    v.addEventListener("segmentend", onEnd);
    return () => v.removeEventListener("segmentend", onEnd);
  }, []);

  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const finish = useCallback(async () => {
    if (rec.state !== "recording") return;
    stopAt.current = null;
    video.current?.pause();
    cancelAnimationFrame(raf.current);
    const blob = await rec.stop();
    setStep("scoring");
    try {
      const r = await api.score(blob, entry.id, line.line_id, baseline, recStart.current);
      const previous = progress.record(entry.id, line.line_id, r.score);
      onTake(line.line_id, { blob, start: recStart.current, score: r.score });
      setResult({ r, previous, url: URL.createObjectURL(blob) });
      setStep("result");
      r.verdict === "Dub" ? sfx.dub() : sfx.goose();
      onStreak(r.verdict === "Dub");
    } catch (e) {
      setError((e as Error).message);
      setStep("turn");
    }
  }, [rec, entry.id, line, baseline, onTake, onStreak]);

  // stop when the scene reaches the end of the line (+ grace), or the recorder hits its limit
  useEffect(() => {
    if (step === "recording" && line && melodyT >= line.end + GRACE - lineZero) finish();
  }, [step, melodyT, line, lineZero, finish]);
  useEffect(() => {
    if (step === "recording" && rec.state === "idle") finish();
  }, [step, rec.state, finish]);

  const dub = useCallback(async () => {
    if (!line) return;
    setError(null);
    setResult(null);
    const v = video.current;
    const from = Math.max(0, line.start - LEAD);
    if (v) {
      v.pause();
      v.muted = true;
      v.currentTime = from;
    }
    setMelodyT(from - lineZero);
    setStep("countdown");
    for (const n of [3, 2, 1]) {
      setCount(n);
      sfx.tick();
      await new Promise((r) => setTimeout(r, 650));
    }
    sfx.clap();
    if (!(await rec.start())) {
      setStep("turn");
      return;
    }
    recStart.current = from;
    setStep("recording");
    await playScene(from, line.end + GRACE + 0.2, true);
  }, [line, lineZero, rec, playScene]);

  // space bar: record / stop
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || (e.target as HTMLElement).closest("button,input,textarea,video")) return;
      e.preventDefault();
      if (step === "turn" || step === "result") dub();
      else if (step === "recording") finish();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!line) return <main className="grid flex-1 place-items-center text-dim">This scene has no playable lines.</main>;

  const now = melodyT + lineZero;                    // scene time
  const speaker = line.speaker_name ?? "the actor";
  const listener = line.addressee_name && line.addressee_name !== line.speaker_name ? line.addressee_name : null;
  const movement = describeMelody(line.shape).movement;
  const phase = now < line.start + line.onset - 0.1 ? "ready" : now <= line.end ? "speak" : "finish";
  const glyph = LANG_GLYPH[entry.target_language];
  const button = "flex items-center gap-2 rounded-md px-5 py-3 font-mono text-sm";
  const primary = `${button} bg-gold font-bold text-ink shadow-[0_5px_0_#a86c00] active:translate-y-1 disabled:opacity-30`;
  const secondary = `${button} border border-rule hover:border-paper`;
  // Stop the scene cleanly (pausing mid-line would leave the listen step waiting for the line's end
  // forever) and count the line as heard: the player has now heard it read in their own language.
  const hearVoice = () => {
    cancelAnimationFrame(raf.current);
    stopAt.current = null;
    video.current?.pause();
    setListened(true);
    const a = voiceEl.current;
    if (a) {
      a.currentTime = 0;
      a.play().catch(() => {});
    }
  };
  const hearIt = line.voice && (
    <button type="button" onClick={hearVoice}
      className={`${button} border border-mint/50 text-mint hover:bg-mint/10`}>
      <AudioLines className="h-4 w-4" /> HEAR IT IN {LANG_NAME[entry.target_language].toUpperCase()}
    </button>
  );

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 px-[max(16px,env(safe-area-inset-left))] py-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <ProgressDots total={lines.length} current={index} />
          <span className="font-mono text-xs tracking-widest text-dim">LINE {index + 1} OF {lines.length}</span>
        </div>
        {step === "recording" ? <OnAir on /> : <Stepper step={step} />}
      </div>

      {/* the scene, and who speaks to whom, like a film still with a caption */}
      <div className="relative mx-auto w-full overflow-hidden rounded-lg bg-black shadow-[0_20px_60px_-20px_rgba(0,0,0,.8)]">
        <video ref={video} src={api.videoUrl(entry.id)} playsInline preload="auto" className="aspect-video max-h-[40vh] w-full object-contain" />
        <p className="pointer-events-none absolute left-3 top-3 rounded bg-ink/70 px-2 py-1 text-xs text-paper/90 backdrop-blur">
          <span className="font-bold capitalize">{speaker}</span>
          {listener && <span className="text-dim"> → <span className="capitalize">{listener}</span></span>}
        </p>
        {step === "recording" && (
          <p className={clsx("pointer-events-none absolute right-3 top-3 rounded px-2 py-1 font-marquee text-sm",
            phase === "speak" ? "bg-onair text-paper" : "bg-ink/70 text-paper/80")}>
            {phase === "ready" ? "GET READY…" : phase === "speak" ? "SPEAK NOW" : "FINISH UP"}
          </p>
        )}
        {needsTap && step === "listen" && (
          <button type="button" onClick={listen} className="absolute inset-0 grid place-items-center bg-ink/60">
            <span className="flex items-center gap-2 rounded-lg bg-gold px-6 py-3 font-marquee text-lg text-ink"><Play className="h-5 w-5 fill-ink" /> PLAY THE LINE</span>
          </button>
        )}
        <AnimatePresence>
          {step === "countdown" && (
            <motion.span key={count} initial={{ scale: 2.2, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ opacity: 0 }}
              className="pointer-events-none absolute inset-0 grid place-items-center bg-ink/40 font-marquee text-8xl text-paper drop-shadow-[0_2px_8px_rgba(0,0,0,.8)]">
              {count}
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {line.voice && <audio ref={voiceEl} src={api.voiceUrl(entry.id, line.line_id)} preload="auto" />}
      <AnimatePresence mode="wait">
        {step === "listen" && (
          <motion.section key="listen" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-5">
            <DirectorNote emotion={line.emotion} intensity={line.intensity} duration={line.shape.duration}
              movement={movement === "rise" || movement === "fall" ? movement : null} />
            <LineToSay line={line} roman={roman} setRoman={setRoman} glyph={glyph} label="YOU'LL SAY" />
            <div className="flex flex-wrap justify-center gap-3">
              <button type="button" onClick={listen} className={secondary}><RotateCcw className="h-4 w-4" /> LISTEN AGAIN</button>
              {hearIt}
              <button type="button" disabled={!listened} onClick={() => { video.current?.pause(); setStep("turn"); }} className={primary}>
                {listened ? "MY TURN →" : "LISTENING…"}
              </button>
            </div>
          </motion.section>
        )}

        {(step === "turn" || step === "countdown") && (
          <motion.section key="turn" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-5">
            <DirectorNote emotion={line.emotion} intensity={line.intensity} duration={line.shape.duration}
              movement={movement === "rise" || movement === "fall" ? movement : null} />
            <LineToSay line={line} roman={roman} setRoman={setRoman} glyph={glyph} label="SAY THIS" />
            {step === "turn" && hearIt}
            <div className="flex items-center gap-5">
              <RecordButton recording={false} level={0} onClick={dub} disabled={step === "countdown"} />
              <div className="flex max-w-[16rem] flex-col gap-1 text-sm text-paper/80">
                <span>{step === "countdown" ? "Get ready…" : "Tap the mic (or press space)."}</span>
                <span className="text-dim">The scene plays muted. Speak when {speaker} does.</span>
                {step === "turn" && (
                  <button type="button" onClick={listen} className="flex items-center gap-1.5 self-start font-mono text-xs text-dim hover:text-paper">
                    <RotateCcw className="h-3 w-3" /> hear it again
                  </button>
                )}
              </div>
            </div>
            {rec.state === "denied" && <p className="text-goose">Microphone blocked. Allow it in the browser's address bar, then reload.</p>}
            {error && <p className="text-center text-goose">{error}</p>}
          </motion.section>
        )}

        {(step === "recording" || step === "scoring") && (
          <motion.section key="rec" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-5">
            {step === "recording" ? (
              <>
                <div className="w-full"><TimeBar from={line.start - LEAD} to={line.end + GRACE} speakFrom={line.start + line.onset} speakTo={line.end} now={now} /></div>
                <LineToSay line={line} roman={roman} glyph={glyph} />
                <div className="flex items-center gap-5">
                  <LevelMeter level={rec.level} />
                  <button type="button" onClick={finish} className={secondary}><Square className="h-3.5 w-3.5 fill-current" /> DONE</button>
                </div>
              </>
            ) : (
              <p className="animate-pulse py-10 text-center font-mono text-sm tracking-[0.3em] text-dim">CHECKING YOUR TAKE…</p>
            )}
          </motion.section>
        )}

        {step === "result" && result && (
          <Result key={`r${index}`} result={result.r} previous={result.previous}
            takeUrl={result.url} originalUrl={api.lineAudioUrl(entry.id, line.line_id)}
            isLast={index === lines.length - 1}
            onRetake={dub} onNext={() => (index === lines.length - 1 ? onFinish() : setIndex(index + 1))}
            onRate={(fair) => { api.feedback(entry.id, line.line_id, result.r, fair).catch(() => {}); }} />
        )}
      </AnimatePresence>
    </main>
  );
}

/** ① LISTEN → ② YOUR TURN, with the current step lit. */
function Stepper({ step }: { step: Step }) {
  const at = step === "listen" ? 0 : step === "result" ? 2 : 1;
  return (
    <ol className="flex items-center gap-2 font-mono text-[11px] tracking-[0.2em]">
      {["LISTEN", "YOUR TURN", "SCORE"].map((label, k) => (
        <li key={label} className={clsx("flex items-center gap-2", k === at ? "text-gold" : k < at ? "text-paper/60" : "text-dim/60")}>
          <span className={clsx("grid h-5 w-5 place-items-center rounded-full border text-[10px]", k === at ? "border-gold bg-gold text-ink" : "border-current")}>{k + 1}</span>
          {label}
          {k < 2 && <span className="text-dim/40">—</span>}
        </li>
      ))}
    </ol>
  );
}

/** The line in the player's language, centred like a subtitle, the punch word in gold. */
function LineToSay({ line, roman, setRoman, glyph, label }: {
  line: Card; roman: boolean; setRoman?: (r: boolean) => void; glyph: string; label?: string;
}) {
  return (
    <div className="flex w-full flex-col items-center gap-2">
      {(label || setRoman) && (
        <div className="flex items-center gap-3">
          {label && <span className="font-mono text-[11px] tracking-[0.25em] text-dim">{label}</span>}
          {setRoman && <ScriptToggle glyph={glyph} roman={roman} onChange={setRoman} />}
        </div>
      )}
      <ScriptLine text={line.translation} roman={line.romanized} stress={line.stress} showRoman={roman} center />
      {line.stress.length > 0 && (
        <p className="flex items-center gap-1.5 text-xs text-dim"><Zap className="h-3.5 w-3.5 text-gold" /> hit the <span className="text-gold">gold</span> word hardest</p>
      )}
    </div>
  );
}

/** Recording timeline: the zone where the actor speaks, and where the scene is now. */
function TimeBar({ from, to, speakFrom, speakTo, now }: { from: number; to: number; speakFrom: number; speakTo: number; now: number }) {
  const pct = (t: number) => `${Math.max(0, Math.min(100, ((t - from) / (to - from)) * 100))}%`;
  return (
    <div className="flex flex-col gap-1.5" aria-hidden>
      <div className="relative h-4 overflow-hidden rounded-full bg-ink-3">
        <div className="absolute inset-y-0 rounded-full bg-onair/35" style={{ left: pct(speakFrom), width: `calc(${pct(speakTo)} - ${pct(speakFrom)})` }} />
        <div className="absolute inset-y-0 left-0 bg-paper/25" style={{ width: pct(now) }} />
        <div className="absolute inset-y-[-2px] w-1 rounded-full bg-paper shadow-[0_0_8px_#fff]" style={{ left: pct(now) }} />
      </div>
      <div className="flex justify-between font-mono text-[10px] tracking-widest text-dim">
        <span>START</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm bg-onair/50" /> THE ACTOR SPEAKS HERE</span>
        <span>END</span>
      </div>
    </div>
  );
}
