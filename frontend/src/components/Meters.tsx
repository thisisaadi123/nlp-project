import { motion } from "framer-motion";
import clsx from "clsx";

/** A segmented LED bar, 0..100, lit one segment at a time. */
export function LedBar({ value, label, delay = 0 }: { value: number; label: string; delay?: number }) {
  const lit = Math.round(value / 10);
  const color = value >= 70 ? "bg-gold shadow-[0_0_8px_rgba(255,194,75,.7)]" : value >= 45 ? "bg-paper/80" : "bg-goose";
  return (
    <div className="grid grid-cols-[5.5rem_1fr_2.5rem] items-center gap-3">
      <span className="font-mono text-[11px] tracking-[0.18em] text-dim">{label}</span>
      <div className="flex gap-1" role="meter" aria-label={label} aria-valuenow={value} aria-valuemin={0} aria-valuemax={100}>
        {Array.from({ length: 10 }, (_, i) => (
          <motion.span
            key={i}
            className={clsx("h-3 flex-1 rounded-[2px]", i < lit ? color : "bg-ink-3")}
            initial={{ opacity: 0.25 }}
            animate={{ opacity: 1 }}
            transition={{ delay: delay + i * 0.045 }}
          />
        ))}
      </div>
      <span className="text-right font-mono text-sm tabular-nums">{value}</span>
    </div>
  );
}

/** Live microphone level as a row of vertical bars. */
export function LevelMeter({ level, bars = 18 }: { level: number; bars?: number }) {
  return (
    <div className="flex h-10 items-end gap-[3px]" aria-hidden>
      {Array.from({ length: bars }, (_, i) => {
        const on = i / bars < level;
        const hot = i / bars > 0.8;
        return (
          <span
            key={i}
            className={clsx("w-1.5 rounded-sm transition-[height,background-color] duration-75",
              on ? (hot ? "bg-onair" : "bg-gold") : "bg-ink-3")}
            style={{ height: `${30 + (i / bars) * 70}%` }}
          />
        );
      })}
    </div>
  );
}
