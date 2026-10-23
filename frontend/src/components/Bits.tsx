import { AnimatePresence, motion } from "framer-motion";
import { Ear, Mic, Star, X } from "lucide-react";
import clsx from "clsx";
import { LANGS, LANG_NATIVE, LANG_NAME, type Lang, type Shape } from "../lib/api";

/** The line's words lighting up at the original's pace (time in the melody line's seconds). */
export function Karaoke({ shape, t, roman, small = false }: { shape: Shape; t: number; roman?: string[]; small?: boolean }) {
  return (
    <p className={clsx("font-script font-bold leading-snug", small ? "text-xl" : "text-center text-[clamp(1.6rem,4vw,2.6rem)]")}>
      {shape.words.map((w, i) => {
        const done = t >= w.end;
        const now = t >= w.start && t < w.end;
        return (
          <span key={i} className={clsx("transition-colors duration-150", w.stress ? (now ? "text-gold" : done ? "text-gold/70" : "text-gold/35") : now ? "text-paper" : done ? "text-paper/60" : "text-paper/25")}>
            {roman?.[i] ?? w.w}
            {i < shape.words.length - 1 && " "}
          </span>
        );
      })}
    </p>
  );
}

/** One dot per line: done ones filled, the current one wide. */
export function ProgressDots({ total, current }: { total: number; current: number }) {
  return (
    <div className="flex items-center gap-1.5" aria-label={`line ${current + 1} of ${total}`}>
      {Array.from({ length: total }, (_, i) => (
        <span key={i} className={clsx("h-2 rounded-full transition-all", i === current ? "w-6 bg-gold" : i < current ? "w-2 bg-paper/70" : "w-2 bg-ink-3")} />
      ))}
    </div>
  );
}

/** The five languages, each written in its own script. */
export function LanguagePicker({ value, onChange, size = "lg" }: { value: Lang | null; onChange: (l: Lang) => void; size?: "lg" | "sm" }) {
  return (
    <div role="radiogroup" aria-label="I'll dub into" className="flex flex-wrap justify-center gap-2">
      {LANGS.map((l) => (
        <button key={l} type="button" role="radio" aria-checked={value === l} onClick={() => onChange(l)}
          className={clsx(
            "flex flex-col items-center rounded-lg border transition-colors",
            size === "lg" ? "min-w-[5.5rem] px-4 py-2.5" : "min-w-[4.5rem] px-3 py-1.5",
            value === l ? "border-gold bg-gold text-ink" : "border-rule text-paper hover:border-paper",
          )}>
          <span className={clsx("font-script font-bold", size === "lg" ? "text-xl" : "text-base")}>{LANG_NATIVE[l]}</span>
          <span className={clsx("font-mono text-[10px] tracking-widest", value === l ? "text-ink/70" : "text-dim")}>{LANG_NAME[l].toUpperCase()}</span>
        </button>
      ))}
    </div>
  );
}

const HINTS: Record<"listen" | "turn" | "result", string> = {
  listen: "Listen to how they say it. Watch the melody line: thick means loud, the dot is the punch.",
  turn: "Your turn. Say your line over the scene and follow the melody line.",
  result: "Gold is the original, mint is you. The closer they look, the better.",
};

/** First-time walkthrough: one short hint per step, on the first line only. */
export function Hint({ step, onDismiss }: { step: keyof typeof HINTS; onDismiss: () => void }) {
  return (
    <AnimatePresence mode="wait">
      <motion.div key={step} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
        className="flex items-start gap-3 rounded-lg border border-gold/40 bg-gold/10 px-4 py-3 text-sm">
        <span className="font-marquee text-gold">{step === "listen" ? "1" : step === "turn" ? "2" : "3"}</span>
        <p className="flex-1 text-paper/90">{HINTS[step]}</p>
        <button type="button" onClick={onDismiss} aria-label="Got it, hide hints" className="text-dim hover:text-paper">
          <X className="h-4 w-4" />
        </button>
      </motion.div>
    </AnimatePresence>
  );
}

const STEPS = [
  { icon: Ear, title: "Listen", text: "to how they say it" },
  { icon: Mic, title: "Dub", text: "the line in your language" },
  { icon: Star, title: "Score", text: "Dub or Goose?" },
];

export function HowToPlay({ compact = false }: { compact?: boolean }) {
  return (
    <ol className={clsx("grid grid-cols-3", compact ? "gap-3" : "gap-4 sm:gap-10")}>
      {STEPS.map(({ icon: Icon, title, text }, i) => (
        <li key={title} className="flex flex-col items-center gap-2 text-center">
          <span className="relative grid h-14 w-14 place-items-center rounded-full border border-rule text-gold">
            <Icon className="h-6 w-6" />
            <span className="absolute -top-1 -left-1 grid h-5 w-5 place-items-center rounded-full bg-gold font-mono text-[11px] font-bold text-ink">{i + 1}</span>
          </span>
          <span className="font-marquee text-sm tracking-wide">{title.toUpperCase()}</span>
          <span className="text-xs text-dim">{text}</span>
        </li>
      ))}
    </ol>
  );
}

/** A small modal for "how to play" from the header. */
export function HelpOverlay({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-30 grid place-items-center bg-ink/80 p-6 backdrop-blur" initial={{ opacity: 0 }}
          animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} role="dialog" aria-modal="true" aria-label="How to play">
          <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }} className="flex w-full max-w-lg flex-col gap-6 rounded-xl border border-rule bg-ink-2 p-6"
            onClick={(e) => e.stopPropagation()}>
            <h2 className="font-marquee text-2xl">HOW TO PLAY</h2>
            <HowToPlay compact />
            <p className="text-sm text-dim">The melody line shows how the original is said: up and down is pitch, thick is loud, the dot is where they punch. Match it to earn a Dub.</p>
            <button type="button" onClick={onClose} className="self-end rounded-md bg-gold px-5 py-2 font-mono text-sm font-bold text-ink">GOT IT</button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
