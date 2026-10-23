import { animate, motion } from "framer-motion";
import { ArrowRight, Flag, Play, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Goose } from "../components/Goose";
import { LedBar } from "../components/Meters";
import { MelodyLine } from "../components/MelodyLine";
import { Stars } from "../components/Script";
import type { ScoreResult, Shape } from "../lib/api";
import { stars } from "../lib/progress";

const LABELS: Record<string, string> = { emotion: "FEEL", contour: "MELODY", timing: "TIMING", emphasis: "PUNCH", content: "WORDS" };
const ORDER = ["content", "emotion", "contour", "timing", "emphasis"];

export function Result({ result, previous, target, takeUrl, originalUrl, isLast, onRetake, onNext }: {
  result: ScoreResult; previous: number | null; target: Shape; takeUrl: string; originalUrl: string;
  isLast: boolean; onRetake: () => void; onNext: () => void;
}) {
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
        <div className="flex items-center gap-4 font-mono text-[11px] tracking-widest">
          <span className="flex items-center gap-1.5"><span className="h-1 w-5 rounded bg-gold" /> ORIGINAL</span>
          <span className="flex items-center gap-1.5"><span className="h-1 w-5 rounded bg-mint" /> YOU</span>
        </div>
        <MelodyLine target={target} compare={result.shape} />
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

      <div className="flex flex-col gap-2 opacity-90">
        {ORDER.filter((k) => k in result.components).map((k, i) => (
          <LedBar key={k} label={LABELS[k]} value={result.components[k]} delay={0.3 + i * 0.12} />
        ))}
      </div>

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
