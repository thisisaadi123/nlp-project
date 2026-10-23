import { AnimatePresence, motion } from "framer-motion";
import { Music2, Play, RotateCcw, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Hint, Karaoke, ProgressDots } from "../components/Bits";
import { EmotionBadge } from "../components/EmotionBadge";
import { describeMelody, MelodyLine } from "../components/MelodyLine";
import { LevelMeter } from "../components/Meters";
import { OnAir, RecordButton } from "../components/RecordButton";
import { ScriptLine, ScriptToggle } from "../components/Script";
import { useRecorder } from "../hooks/useRecorder";
import { api, LANG_GLYPH, type Baseline, type Card, type Entry, type ScoreResult } from "../lib/api";
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
  const [hints, setHints] = useState(() => !progress.walkthroughSeen());
  const [result, setResult] = useState<{ r: ScoreResult; previous: number | null; url: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const rec = useRecorder(30);
  const video = useRef<HTMLVideoElement>(null);
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
      const r = await api.score(blob, entry.id, line.line_id, baseline);
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

  const hintStep = step === "listen" ? "listen" : step === "result" ? "result" : "turn";
  const showHints = hints && index === 0;
  const listenProgress = Math.max(0, Math.min(1, melodyT / line.shape.duration));
  const live = rec.trace.map((p) => ({ ...p, t: p.t + recStart.current - lineZero }));
  const romanWords = line.romanized ? line.romanized.split(/\s+/).filter(Boolean) : [];

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-5 px-[max(16px,env(safe-area-inset-left))] py-5">
      {/* where we are, and who is speaking */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <ProgressDots total={lines.length} current={index} />
          <span className="font-mono text-xs tracking-widest text-dim">LINE {index + 1} OF {lines.length}</span>
        </div>
        {step === "recording" && <OnAir on />}
      </div>

      {showHints && <Hint step={hintStep} onDismiss={() => { setHints(false); progress.setWalkthroughSeen(true); }} />}

      <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr] lg:items-start lg:gap-10">
        {/* the scene (sticks while the step panel scrolls on desktop) */}
        <div className="flex min-w-0 flex-col gap-3 lg:sticky lg:top-20">
          <div className="relative overflow-hidden rounded-[3px] bg-black">
            <video ref={video} src={api.videoUrl(entry.id)} playsInline preload="auto" className="aspect-video max-h-[38vh] w-full lg:max-h-none" />
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

          {(line.speaker_name || line.addressee_name) && (
            <p className="text-sm text-dim">
              <span className="font-bold capitalize text-paper">{line.speaker_name ?? "Someone"}</span>
              {line.addressee_name && <> → <span className="capitalize">{line.addressee_name}</span></>}
            </p>
          )}

        </div>
        <div className="min-w-0">
          <AnimatePresence mode="wait">
            {step === "listen" && (
              <motion.section key="listen" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex flex-col gap-5">
                <p className="font-marquee text-2xl">LISTEN</p>
                <EmotionBadge emotion={line.emotion} intensity={line.intensity} tone={line.tone} duration={line.shape.duration} />
                <div className="flex flex-col gap-2">
                  <MelodyLine target={line.shape} progress={listenProgress} />
                  <p className="flex items-center gap-2 text-sm text-dim"><Music2 className="h-4 w-4 shrink-0 text-gold" /> {describeMelody(line.shape)}</p>
                </div>
                <Karaoke small shape={line.shape} t={listenProgress * line.shape.duration} />
                <div className="flex flex-wrap gap-3">
                  <button type="button" onClick={listen} className="flex items-center gap-2 rounded-md border border-rule px-5 py-3 font-mono text-sm hover:border-paper">
                    <RotateCcw className="h-4 w-4" /> LISTEN AGAIN
                  </button>
                  <button type="button" disabled={!listened} onClick={() => { video.current?.pause(); setStep("turn"); }}
                    className="rounded-md bg-gold px-6 py-3 font-mono text-sm font-bold text-ink shadow-[0_5px_0_#a86c00] transition-opacity active:translate-y-1 disabled:opacity-30">
                    MY TURN →
                  </button>
                </div>
              </motion.section>
            )}

            {(step === "turn" || step === "countdown" || step === "recording" || step === "scoring") && (
              <motion.section key="turn" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex flex-col gap-5">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-marquee text-2xl">{step === "recording" ? "GO!" : "YOUR TURN"}</p>
                  <ScriptToggle glyph={LANG_GLYPH[entry.target_language]} roman={roman} onChange={setRoman} />
                </div>

                {step === "recording" ? (
                  <Karaoke shape={line.shape} t={melodyT} roman={roman && romanWords.length === line.shape.words.length ? romanWords : undefined} />
                ) : (
                  <>
                    <ScriptLine text={line.translation} roman={line.romanized} stress={line.stress} showRoman={roman} />
                    <p className="text-lg text-paper/90">{line.coach}</p>
                  </>
                )}

                <MelodyLine target={line.shape} live={step === "recording" ? live : undefined}
                  progress={step === "recording" ? Math.max(0, Math.min(1, melodyT / line.shape.duration)) : undefined} />

                <div className="flex flex-col items-center gap-3">
                  {step === "recording" ? (
                    <>
                      <LevelMeter level={rec.level} />
                      <button type="button" onClick={finish} className="flex items-center gap-2 rounded-md border border-rule px-5 py-2.5 font-mono text-sm hover:border-paper">
                        <Square className="h-3.5 w-3.5 fill-current" /> STOP
                      </button>
                    </>
                  ) : step === "scoring" ? (
                    <p className="animate-pulse py-6 font-mono text-xs tracking-[0.3em] text-dim">LISTENING…</p>
                  ) : (
                    <>
                      <RecordButton recording={false} level={0} onClick={dub} disabled={step === "countdown"} />
                      <span className="font-mono text-[11px] tracking-[0.25em] text-dim">
                        {step === "countdown" ? "GET READY" : "TAP TO DUB OVER THE SCENE"}
                      </span>
                      {step === "turn" && (
                        <button type="button" onClick={listen} className="flex items-center gap-2 font-mono text-xs text-dim hover:text-paper">
                          <RotateCcw className="h-3.5 w-3.5" /> hear the original again
                        </button>
                      )}
                    </>
                  )}
                </div>
                {rec.state === "denied" && <p className="text-goose">Microphone blocked. Allow it in the browser's address bar, then reload.</p>}
                {error && <p className="text-center text-goose">{error}</p>}
              </motion.section>
            )}

            {step === "result" && result && (
              <Result key={`r${index}`} result={result.r} previous={result.previous} target={line.shape}
                takeUrl={result.url} originalUrl={api.lineAudioUrl(entry.id, line.line_id)}
                isLast={index === lines.length - 1}
                onRetake={dub} onNext={() => (index === lines.length - 1 ? onFinish() : setIndex(index + 1))} />
            )}
          </AnimatePresence>
        </div>
      </div>
    </main>
  );
}
