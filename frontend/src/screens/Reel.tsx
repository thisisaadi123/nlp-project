import { motion } from "framer-motion";
import { ArrowRight, Plus, Star, Trash2 } from "lucide-react";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { api, LANG_NAME, type Lang, type Scene } from "../lib/api";
import { progress } from "../lib/progress";

const LEVEL = { easy: "Easy", medium: "Medium", hard: "Hard" } as const;

/** Scene picker: a strip of film, one frame per scene dubbed into the player's language. */
export function Reel({ lang, onPick, onChangeLang, onUpload }: {
  lang: Lang; onPick: (scene: Scene) => void; onChangeLang: () => void; onUpload: () => void;
}) {
  const [scenes, setScenes] = useState<Scene[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setScenes(null);
    api.scenes(lang).then(setScenes).catch((e: Error) => setError(e.message));
  }, [lang]);

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-[max(16px,env(safe-area-inset-left))] py-12 sm:pt-20">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="font-marquee text-3xl">PICK A SCENE</h2>
        <span className="font-mono text-xs tracking-[0.2em] text-dim">DUBBING INTO {LANG_NAME[lang].toUpperCase()}</span>
      </div>

      {error && (
        <p className="text-goose">
          {error} <button type="button" className="underline" onClick={() => window.location.reload()}>Try again</button>
        </p>
      )}

      {scenes && scenes.length === 0 && (
        <p className="text-dim">
          No scenes in {LANG_NAME[lang]} yet: upload your own, or{" "}
          <button type="button" onClick={onChangeLang} className="text-paper underline underline-offset-4 hover:text-gold">change language</button>.
        </p>
      )}

      {!error && (
        <div className="-mx-4 overflow-x-auto pb-4">
          <div className="inline-flex flex-col bg-black/70 px-4">
            <div className="sprockets h-4" />
            <div className="flex gap-4 py-2">
              <button type="button" onClick={onUpload}
                className="group flex w-64 shrink-0 flex-col gap-2 text-left sm:w-72">
                <div className="grid aspect-video place-items-center rounded-[3px] border-2 border-dashed border-rule transition-colors group-hover:border-gold">
                  <span className="flex flex-col items-center gap-2 text-dim group-hover:text-gold">
                    <Plus className="h-8 w-8" />
                    <span className="font-marquee text-lg">YOUR SCENE</span>
                  </span>
                </div>
                <span className="px-0.5 text-sm text-dim">Upload a clip (up to 45 s) and dub it</span>
              </button>
              {scenes === null
                ? [0, 1, 2].map((i) => <div key={i} className="aspect-video w-64 animate-pulse rounded-[3px] bg-ink-3 sm:w-72" />)
                : scenes.map((s, i) => {
                    const earned = progress.sceneStars(s.id);
                    return (
                      <motion.button key={s.id} type="button" onClick={() => onPick(s)}
                        initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.08 }}
                        whileHover={{ y: -6 }} className="group flex w-64 shrink-0 flex-col gap-2 text-left sm:w-72">
                        <div className="relative aspect-video overflow-hidden rounded-[3px] bg-ink-3">
                          <img src={api.thumbnailUrl(s.id)} alt="" loading="lazy"
                            className="h-full w-full object-cover opacity-85 transition-opacity group-hover:opacity-100" />
                          <span className={clsx("absolute left-2 top-2 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-bold tracking-widest",
                            s.uploaded ? "bg-paper text-ink" : s.difficulty === "easy" ? "bg-mint text-ink" : s.difficulty === "medium" ? "bg-gold text-ink" : "bg-onair text-paper")}>
                            {s.uploaded ? "YOURS" : LEVEL[s.difficulty].toUpperCase()}
                          </span>
                          {s.uploaded && (
                            <span role="button" tabIndex={0} aria-label="Remove this scene"
                              onClick={(e) => { e.stopPropagation(); if (confirm("Remove this scene?")) api.removeScene(s.id).then(() => setScenes((all) => all?.filter((x) => x.id !== s.id) ?? null)).catch(() => {}); }}
                              className="absolute right-2 top-2 grid h-7 w-7 place-items-center rounded bg-ink/70 text-dim opacity-0 transition-opacity hover:text-goose group-hover:opacity-100">
                              <Trash2 className="h-3.5 w-3.5" />
                            </span>
                          )}
                          <span className="absolute bottom-2 right-2 grid h-9 w-9 place-items-center rounded-full bg-gold text-ink opacity-0 transition-opacity group-hover:opacity-100">
                            <ArrowRight className="h-5 w-5" />
                          </span>
                        </div>
                        <div className="flex items-baseline justify-between gap-2 px-0.5">
                          <span className="font-marquee text-lg leading-tight">{s.title}</span>
                          <span className="flex shrink-0 items-center gap-1 font-mono text-xs text-dim">
                            <Star className="h-3 w-3 fill-gold text-gold" /> {earned}/{s.lines * 3}
                          </span>
                        </div>
                        <p className="px-0.5 font-mono text-[11px] tracking-widest text-dim">
                          {LANG_NAME[s.source_language].toUpperCase()} <span className="text-gold">→</span> {LANG_NAME[s.target_language].toUpperCase()} · {s.lines} LINES
                        </p>
                        {s.characters.length > 0 && <p className="line-clamp-1 px-0.5 text-sm text-dim">{s.characters.join(", ")}</p>}
                      </motion.button>
                    );
                  })}
            </div>
            <div className="sprockets h-4" />
          </div>
        </div>
      )}
    </main>
  );
}
