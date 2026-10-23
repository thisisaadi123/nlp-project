import { AnimatePresence, motion } from "framer-motion";
import { Check, Mic, Play, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { LevelMeter } from "../components/Meters";
import { OnAir, RecordButton } from "../components/RecordButton";
import { useRecorder } from "../hooks/useRecorder";
import { api, LANG_NAME, type Baseline, type Lang } from "../lib/api";
import { progress } from "../lib/progress";

type Step = "ask" | "ready" | "recorded" | "saving" | "done";

/** Read one sentence calmly: the player's normal voice, so intensity is judged against it. */
export function MicCheck({ lang, onDone }: { lang: Lang; onDone: (b: Baseline | null) => void }) {
  const [text, setText] = useState("");
  const [step, setStep] = useState<Step>("ask");
  const [take, setTake] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const rec = useRecorder(15);
  const player = useRef<HTMLAudioElement>(null);
  const url = useRef<string | null>(null);

  useEffect(() => {
    api.calibration(lang).then((r) => setText(r.text)).catch(() => setText(""));
  }, [lang]);
  useEffect(() => () => { if (url.current) URL.revokeObjectURL(url.current); }, []);

  const allow = async () => {
    if (await rec.requestPermission()) setStep("ready");
  };

  const toggle = async () => {
    setError(null);
    if (rec.state === "recording") {
      const blob = await rec.stop();
      if (url.current) URL.revokeObjectURL(url.current);
      url.current = URL.createObjectURL(blob);
      setTake(blob);
      setStep("recorded");
    } else {
      await rec.start();
    }
  };

  const save = async () => {
    if (!take) return;
    setStep("saving");
    try {
      const { baseline } = await api.calibrate(take);
      progress.setBaseline(lang, baseline);
      setStep("done");
      setTimeout(() => onDone(baseline), 900);
    } catch (e) {
      setError((e as Error).message);
      setStep("recorded");
    }
  };

  const levelWord = rec.level < 0.08 ? "too quiet" : rec.level > 0.85 ? "too loud" : "good";

  if (rec.state === "denied") {
    return (
      <main className="mx-auto flex max-w-xl flex-1 flex-col justify-center gap-5 px-6 py-12">
        <h2 className="font-marquee text-3xl">MIC BLOCKED</h2>
        <p>The game needs your microphone to hear your dub. To allow it:</p>
        <ul className="flex flex-col gap-2 border-l-2 border-rule pl-4 text-paper/90">
          <li><b>Chrome:</b> click the icon left of the address bar → Site settings → Microphone → Allow.</li>
          <li><b>Safari:</b> Safari menu → Settings for This Website → Microphone → Allow.</li>
        </ul>
        <p className="text-dim">Then reload the page.</p>
        <div className="flex gap-3">
          <button type="button" onClick={() => window.location.reload()} className="rounded-md bg-gold px-5 py-2.5 font-mono text-sm font-bold text-ink">RELOAD</button>
          <button type="button" onClick={() => onDone(null)} className="rounded-md border border-rule px-5 py-2.5 font-mono text-sm">SKIP</button>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto flex max-w-3xl flex-1 flex-col items-center justify-center gap-8 px-6 py-12 text-center">
      <div className="flex items-center gap-3">
        <h2 className="font-marquee text-3xl">MIC CHECK</h2>
        <OnAir on={rec.state === "recording"} />
      </div>

      {step === "ask" ? (
        <>
          <p className="max-w-md text-lg">We need your microphone to hear your dub.</p>
          <p className="max-w-md text-sm text-dim">Each take is scored on our server, and its words are checked by Sarvam's speech recognition. We don't keep your recordings.</p>
          <button type="button" onClick={allow}
            className="flex items-center gap-2 rounded-lg bg-gold px-8 py-3.5 font-marquee text-lg text-ink shadow-[0_6px_0_#a86c00] active:translate-y-1 active:shadow-[0_3px_0_#a86c00]">
            <Mic className="h-5 w-5" /> ALLOW MIC
          </button>
        </>
      ) : (
        <>
          <p className="max-w-md text-dim">Read this calmly once, so we know your normal voice. ({LANG_NAME[lang]})</p>
          <p className="font-script text-[clamp(1.5rem,3.5vw,2.3rem)] font-bold leading-relaxed">{text}</p>

          <AnimatePresence mode="wait">
            {step === "done" ? (
              <motion.div key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                className="grid h-28 w-28 place-items-center rounded-full bg-mint text-ink">
                <Check className="h-14 w-14" strokeWidth={3} />
              </motion.div>
            ) : step === "recorded" || step === "saving" ? (
              <motion.div key="recorded" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-wrap justify-center gap-3">
                <button type="button" onClick={() => player.current?.play()} className="flex items-center gap-2 rounded-md border border-rule px-5 py-3 font-mono text-sm hover:border-paper">
                  <Play className="h-4 w-4 fill-current" /> HEAR IT
                </button>
                <button type="button" onClick={() => { setTake(null); setStep("ready"); }} className="flex items-center gap-2 rounded-md border border-rule px-5 py-3 font-mono text-sm hover:border-paper">
                  <RotateCcw className="h-4 w-4" /> REDO
                </button>
                <button type="button" onClick={save} disabled={step === "saving"}
                  className="rounded-md bg-gold px-6 py-3 font-mono text-sm font-bold text-ink disabled:opacity-50">
                  {step === "saving" ? "SAVING…" : "SOUNDS GOOD →"}
                </button>
                <audio ref={player} src={url.current ?? undefined} />
              </motion.div>
            ) : (
              <motion.div key="rec" className="flex flex-col items-center gap-3">
                <RecordButton recording={rec.state === "recording"} level={rec.level} onClick={toggle} />
                <LevelMeter level={rec.level} />
                <span className="h-5 font-mono text-xs tracking-widest text-dim">
                  {rec.state === "recording" ? levelWord.toUpperCase() : "TAP TO RECORD, TAP AGAIN TO STOP"}
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}

      {error && <p className="text-goose">{error}</p>}
      {(step === "ask" || step === "ready") && rec.state !== "recording" && (
        <button type="button" onClick={() => onDone(null)} className="font-mono text-xs tracking-widest text-dim underline-offset-4 hover:underline">
          SKIP
        </button>
      )}
    </main>
  );
}
