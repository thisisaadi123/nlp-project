import { motion } from "framer-motion";
import { Mic, Square } from "lucide-react";
import clsx from "clsx";

/** The big red button. A ring around it swells with the microphone level while recording. */
export function RecordButton({ recording, level, disabled, onClick, size = 112 }: {
  recording: boolean; level: number; disabled?: boolean; onClick: () => void; size?: number;
}) {
  return (
    <div className="relative grid place-items-center" style={{ width: size + 48, height: size + 48 }}>
      <motion.span
        className="absolute rounded-full border-2 border-onair/60"
        style={{ width: size + 12, height: size + 12 }}
        animate={{ scale: recording ? 1 + level * 0.35 : 1, opacity: recording ? 0.4 + level * 0.6 : 0.25 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
      />
      <motion.button
        type="button"
        onClick={onClick}
        disabled={disabled}
        aria-label={recording ? "Stop recording" : "Start recording"}
        whileTap={{ scale: 0.94 }}
        className={clsx(
          "relative grid place-items-center rounded-full text-paper transition-colors disabled:opacity-40",
          "bg-[radial-gradient(circle_at_35%_30%,#ff6b7d,#d81c37_60%,#9e1026)]",
          "shadow-[0_10px_0_#6d0b1a,0_14px_30px_rgba(255,61,85,.35)] active:translate-y-1 active:shadow-[0_6px_0_#6d0b1a]",
        )}
        style={{ width: size, height: size }}
      >
        {recording ? <Square className="h-9 w-9 fill-paper" /> : <Mic className="h-10 w-10" strokeWidth={2.2} />}
      </motion.button>
    </div>
  );
}

/** The ON AIR lamp. */
export function OnAir({ on }: { on: boolean }) {
  return (
    <span
      className={clsx(
        "rounded-sm border px-2 py-0.5 font-marquee text-xs tracking-widest transition-colors",
        on ? "animate-flicker border-onair bg-onair text-paper shadow-[0_0_18px_rgba(255,61,85,.6)]" : "border-rule text-dim",
      )}
    >
      ON AIR
    </span>
  );
}
