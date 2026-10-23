import clsx from "clsx";

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
