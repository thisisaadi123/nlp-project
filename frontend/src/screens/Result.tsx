import { animate, motion } from "framer-motion";
import { ArrowRight, Check, Clock3, Flag, Music2, Play, RotateCcw, ThumbsDown, ThumbsUp, Type, Volume2, Zap } from "lucide-react";
import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import { Goose } from "../components/Goose";
import { Stars } from "../components/Script";
import type { ScoreResult } from "../lib/api";
import { stars } from "../lib/progress";

const PARTS = [
  { key: "timing", label: "TIMING", icon: Clock3 },
  { key: "volume", label: "VOLUME", icon: Volume2 },
  { key: "punch", label: "PUNCH", icon: Zap },
  { key: "tone", label: "TONE", icon: Music2 },
  { key: "words", label: "WORDS", icon: Type },
];

export function Result({ result, previous, takeUrl, originalUrl, isLast, onRetake, onNext, onRate }: {
  result: ScoreResult; previous: number | null; takeUrl: string; originalUrl: string;
  isLast: boolean; onRetake: () => void; onNext: () => void; onRate?: (fair: boolean) => void;
}) {
  const [rated, setRated] = useState<boolean | null>(null);
  const [shown, setShown] = useState(0);
  const dub = result.verdict === "Dub";
  const yours = useRef<HTMLAudioElement>(null);
  const original = useRef<HTMLAudioElement>(null);
  const play = (which: "yours" | "original") => {
    const [a, b] = which === "yours" ? [yours.current, original.current] : [original.current, yours.current];
    b?.pause();
    if (a) {
      a.currentTime = 0;
      a.play().catch(() => {});
    }
  };

  useEffect(() => {
    const c = animate(0, result.score, { duration: 1.1, ease: "easeOut", onUpdate: (v) => setShown(Math.round(v)) });
    return () => c.stop();
  }, [result.score]);

  return (
    <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex flex-col gap-6" aria-live="polite">
      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-col gap-2">
          <div className="flex items-baseline gap-2">
            <span className="font-marquee text-7xl tabular-nums leading-none">{shown}</span>
            <span className="font-mono text-dim">/100</span>
          </div>
          <div className="flex items-center gap-3">
            <Stars n={stars(result.score)} size={20} />
            {previous !== null && result.score > previous && <span className="font-mono text-xs tracking-widest text-mint">NEW BEST</span>}
          </div>
        </div>
        {dub ? (
          <motion.span initial={{ scale: 3, rotate: -25, opacity: 0 }} animate={{ scale: 1, rotate: -8, opacity: 1 }}
            transition={{ delay: 1.1, type: "spring", stiffness: 260, damping: 14 }}
            className="rounded-md border-4 border-gold px-4 py-1 font-marquee text-4xl text-gold shadow-[0_0_30px_rgba(255,194,75,.35)]">
            DUB!
          </motion.span>
        ) : (
          <motion.div initial={{ x: 140, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: 1.1, type: "spring" }} className="flex items-center gap-1">
            <Goose size={84} honking />
            <span className="font-marquee text-3xl text-goose">GOOSE</span>
          </motion.div>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <p className="font-mono text-[11px] tracking-[0.25em] text-dim">COMPARE BY EAR</p>
        <div className="flex gap-3">
          <button type="button" onClick={() => play("yours")} className="flex items-center gap-2 rounded-md border border-mint/60 px-4 py-2 font-mono text-sm text-mint hover:bg-mint/10">
            <Play className="h-3.5 w-3.5 fill-current" /> YOURS
          </button>
          <button type="button" onClick={() => play("original")} className="flex items-center gap-2 rounded-md border border-gold/60 px-4 py-2 font-mono text-sm text-gold hover:bg-gold/10">
            <Play className="h-3.5 w-3.5 fill-current" /> ORIGINAL
          </button>
          <audio ref={yours} src={takeUrl} preload="auto" />
          <audio ref={original} src={originalUrl} preload="auto" />
        </div>
      </div>

      {result.tips[0] && <p className="border-l-2 border-gold pl-4 text-lg text-paper/90">{result.tips[0]}</p>}

      {result.word_check && result.word_check.some((w) => !w.ok) && (
        <div className="flex flex-col gap-1">
          <p className="font-mono text-[11px] tracking-[0.25em] text-dim">YOUR WORDS</p>
          <p className="font-script text-xl leading-relaxed">
            {result.word_check.map((w, i) => (
              <span key={i} title={w.ok ? undefined : w.heard ? `we heard: ${w.heard}` : "skipped"}
                className={clsx(w.ok ? "text-paper/80" : "text-goose underline decoration-wavy decoration-goose/70 underline-offset-4")}>
                {w.word}{i < result.word_check!.length - 1 && " "}
              </span>
            ))}
          </p>
          {result.heard && <p className="text-sm text-dim">We heard: <span className="font-script text-paper/70">{result.heard}</span></p>}
        </div>
      )}

      {onRate && (
        <div className="flex items-center gap-3 text-sm text-dim">
          {rated === null ? (
            <>
              <span>Was this score fair?</span>
              {([true, false] as const).map((fair) => (
                <button key={String(fair)} type="button" aria-label={fair ? "yes, fair" : "no, unfair"}
                  onClick={() => { setRated(fair); onRate(fair); }}
                  className="grid h-8 w-8 place-items-center rounded-full border border-rule hover:border-paper hover:text-paper">
                  {fair ? <ThumbsUp className="h-4 w-4" /> : <ThumbsDown className="h-4 w-4" />}
                </button>
              ))}
            </>
          ) : (
            <span>Thanks, that helps us tune the score.</span>
          )}
        </div>
      )}

      <ul className="flex flex-col divide-y divide-rule/60 border-y border-rule/60">
        {PARTS.filter((p) => result.parts?.[p.key]).map(({ key, label, icon: Icon }, i) => {
          const part = result.parts[key];
          const good = part.score >= 70;
          return (
            <motion.li key={key} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 + i * 0.1 }}
              className="grid grid-cols-[1.5rem_5.5rem_1fr_auto] items-center gap-3 py-2.5">
              <Icon className={clsx("h-4 w-4", good ? "text-mint" : "text-goose")} />
              <span className="font-mono text-[11px] tracking-[0.18em] text-dim">{label}</span>
              <span className={clsx("flex items-center gap-1.5 font-medium", good ? "text-paper" : "text-goose")}>
                {good && <Check className="h-4 w-4 text-mint" />}{part.verdict}
              </span>
              <span className="h-1.5 w-16 overflow-hidden rounded-full bg-ink-3">
                <motion.span className={clsx("block h-full rounded-full", good ? "bg-mint" : "bg-goose")}
                  initial={{ width: 0 }} animate={{ width: `${part.score}%` }} transition={{ delay: 0.5 + i * 0.1, duration: 0.5 }} />
              </span>
            </motion.li>
          );
        })}
      </ul>

      <div className="flex flex-wrap gap-3">
        <button type="button" onClick={onRetake} className="flex items-center gap-2 rounded-md border border-rule px-5 py-3 font-mono text-sm tracking-widest hover:border-paper">
          <RotateCcw className="h-4 w-4" /> RETAKE
        </button>
        <button type="button" onClick={onNext}
          className="flex items-center gap-2 rounded-md bg-gold px-5 py-3 font-mono text-sm font-bold tracking-widest text-ink shadow-[0_5px_0_#a86c00] active:translate-y-1 active:shadow-[0_2px_0_#a86c00]">
          {isLast ? <>FINISH SCENE <Flag className="h-4 w-4" /></> : <>NEXT LINE <ArrowRight className="h-4 w-4" /></>}
        </button>
      </div>
    </motion.section>
  );
}
