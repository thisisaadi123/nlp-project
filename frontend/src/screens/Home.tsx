import { motion } from "framer-motion";
import { Flame, Play, Star } from "lucide-react";
import { useEffect } from "react";
import { HowToPlay, LanguagePicker } from "../components/Bits";
import { Logo } from "../components/Logo";
import type { Lang } from "../lib/api";
import { progress } from "../lib/progress";
import { sfx } from "../lib/sound";

export function Home({ lang, onLang, onStart, onContinue }: {
  lang: Lang | null; onLang: (l: Lang) => void; onStart: () => void; onContinue: (scene: string) => void;
}) {
  const last = progress.last();
  const total = progress.totalStars();
  const best = progress.bestStreak();
  const start = () => {
    if (!lang) return;
    sfx.clap();
    onStart();
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Enter" && start();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col items-center justify-center gap-10 px-6 py-12 text-center">
      <motion.div initial={{ y: -30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ type: "spring", stiffness: 120 }}>
        <Logo big />
      </motion.div>
      <p className="max-w-md text-lg text-paper/90">
        Dub movie scenes in your language. Match the emotion. Earn a <span className="font-bold text-gold">Dub</span>, or get{" "}
        <span className="font-bold text-goose">Goosed</span>.
      </p>

      <div className="flex flex-col items-center gap-4">
        <span className="font-mono text-xs tracking-[0.25em] text-dim">I'LL DUB INTO</span>
        <LanguagePicker value={lang} onChange={onLang} />
      </div>

      <div className="flex flex-col items-center gap-3">
        <motion.button type="button" onClick={start} disabled={!lang}
          whileHover={lang ? { scale: 1.04 } : undefined} whileTap={lang ? { scale: 0.97, y: 4 } : undefined}
          className="rounded-lg bg-gold px-10 py-4 font-marquee text-xl text-ink shadow-[0_8px_0_#a86c00] transition-opacity active:shadow-[0_4px_0_#a86c00] disabled:opacity-30">
          START DUBBING
        </motion.button>
        {!lang && <span className="text-sm text-dim">Pick a language first.</span>}
        {last && (
          <button type="button" onClick={() => onContinue(last.scene)}
            className="flex items-center gap-2 font-mono text-sm text-dim hover:text-paper">
            <Play className="h-3.5 w-3.5 fill-current" /> Continue: {last.title} · line {last.line + 1}/{last.total}
          </button>
        )}
      </div>

      <HowToPlay />

      {(total > 0 || best > 0) && (
        <p className="flex items-center gap-5 font-mono text-sm text-dim">
          <span className="flex items-center gap-1.5"><Star className="h-4 w-4 fill-gold text-gold" /> {total} stars</span>
          <span className="flex items-center gap-1.5"><Flame className="h-4 w-4 text-goose" /> best streak {best}</span>
        </p>
      )}

      <p className="max-w-md text-xs text-dim/70">
        Clips are short excerpts used for research. Your voice stays on your device except when it is being scored.
      </p>
    </main>
  );
}
