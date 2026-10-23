import { AnimatePresence, motion } from "framer-motion";
import { Clapperboard, Film, Home, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { Goose } from "../components/Goose";
import { Stars } from "../components/Script";
import { api, type Entry } from "../lib/api";
import { DUB_AT, progress, stars } from "../lib/progress";
import type { Take } from "./Booth";

/** End of a scene: the scene score, each line's best, and the premiere of the player's dub. */
export function SceneEnd({ entry, takes, onRetry, onHome, onScenes }: {
  entry: Entry; takes: Record<number, Take>; onRetry: (index: number) => void; onHome: () => void; onScenes: () => void;
}) {
  const lines = entry.lines.filter((l) => l.playable && l.shape);
  const bests = lines.map((l) => takes[l.line_id]?.score ?? progress.best(entry.id, l.line_id));
  const scored = bests.filter((b): b is number => b !== null);
  const avg = scored.length ? Math.round(scored.reduce((a, b) => a + b, 0) / scored.length) : 0;
  const dub = avg >= DUB_AT;
  const [premiere, setPremiere] = useState<{ state: "idle" | "mixing" | "ready" | "error"; url?: string; error?: string }>({ state: "idle" });
  const [titleCard, setTitleCard] = useState(false);

  useEffect(() => () => { if (premiere.url) URL.revokeObjectURL(premiere.url); }, [premiere.url]);

  const sessionTakes = lines.filter((l) => takes[l.line_id]).map((l) => ({ line: l.line_id, ...takes[l.line_id] }));

  const watch = async () => {
    setPremiere({ state: "mixing" });
    try {
      const url = await api.premiere(entry.id, sessionTakes);
      setPremiere({ state: "ready", url });
      setTitleCard(true);
      setTimeout(() => setTitleCard(false), 1800);
    } catch (e) {
      setPremiere({ state: "error", error: (e as Error).message });
    }
  };

  return (
    <main className="mx-auto grid w-full max-w-5xl flex-1 content-center gap-10 px-[max(16px,env(safe-area-inset-left))] py-10 lg:grid-cols-[1fr_1.2fr]">
      <section className="flex flex-col gap-6">
        <p className="font-mono text-xs tracking-[0.25em] text-dim">SCENE COMPLETE · {entry.title.toUpperCase()}</p>
        <div className="flex items-center gap-5">
          <span className="font-marquee text-8xl leading-none tabular-nums">{avg}</span>
          {dub ? (
            <span className="-rotate-6 rounded-md border-4 border-gold px-3 py-1 font-marquee text-3xl text-gold">DUB!</span>
          ) : (
            <span className="flex items-center"><Goose size={72} honking /><span className="font-marquee text-2xl text-goose">GOOSE</span></span>
          )}
        </div>
        <Stars n={stars(avg)} size={24} />

        <ol className="flex flex-col">
          {lines.map((l, i) => (
            <li key={l.line_id}>
              <button type="button" onClick={() => onRetry(i)}
                className="grid w-full grid-cols-[2rem_1fr_auto] items-center gap-3 border-b border-rule/60 py-2.5 text-left text-dim hover:text-paper">
                <span className="font-mono text-xs">{String(i + 1).padStart(2, "0")}</span>
                <span className="truncate font-script">{l.translation}</span>
                {bests[i] !== null ? <span className="flex items-center gap-2 font-mono text-sm text-paper">{Math.round(bests[i] as number)} <Stars n={stars(bests[i] as number)} size={11} /></span>
                  : <span className="font-mono text-xs">not dubbed</span>}
              </button>
            </li>
          ))}
        </ol>
      </section>

      <section className="flex flex-col gap-4">
        <div className="relative overflow-hidden rounded-[3px] bg-black">
          {premiere.state === "ready" ? (
            <video src={premiere.url} controls autoPlay playsInline className="aspect-video w-full" />
          ) : (
            <img src={api.thumbnailUrl(entry.id)} alt="" className="aspect-video w-full object-cover opacity-50" />
          )}
          <AnimatePresence>
            {titleCard && (
              <motion.div initial={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.6 }}
                className="absolute inset-0 grid place-items-center bg-ink">
                <motion.p initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="text-center font-marquee text-4xl text-gold">
                  PREMIERE<span className="block text-lg text-paper">starring you</span>
                </motion.p>
              </motion.div>
            )}
          </AnimatePresence>
          {premiere.state !== "ready" && (
            <div className="absolute inset-0 grid place-items-center">
              <button type="button" onClick={watch} disabled={!sessionTakes.length || premiere.state === "mixing"}
                className="flex items-center gap-2 rounded-lg bg-gold px-7 py-3.5 font-marquee text-xl text-ink shadow-[0_6px_0_#a86c00] disabled:opacity-40">
                <Film className="h-5 w-5" /> {premiere.state === "mixing" ? "CUTTING…" : "WATCH YOUR PREMIERE"}
              </button>
            </div>
          )}
        </div>
        {!sessionTakes.length && <p className="text-sm text-dim">Dub at least one line in this session to see your premiere.</p>}
        {premiere.state === "error" && <p className="text-goose">{premiere.error}</p>}

        <div className="flex flex-wrap gap-3">
          <button type="button" onClick={() => onRetry(0)} className="flex items-center gap-2 rounded-md border border-rule px-4 py-2.5 font-mono text-sm hover:border-paper">
            <RotateCcw className="h-4 w-4" /> REPLAY SCENE
          </button>
          <button type="button" onClick={onScenes} className="flex items-center gap-2 rounded-md bg-gold px-4 py-2.5 font-mono text-sm font-bold text-ink">
            <Clapperboard className="h-4 w-4" /> NEXT SCENE
          </button>
          <button type="button" onClick={onHome} className="flex items-center gap-2 rounded-md px-3 py-2.5 font-mono text-sm text-dim hover:text-paper">
            <Home className="h-4 w-4" /> HOME
          </button>
        </div>
      </section>
    </main>
  );
}
