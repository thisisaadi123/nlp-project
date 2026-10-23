import { motion } from "framer-motion";
import { Ear, Mic, Star } from "lucide-react";
import { api, LANG_NAME, type Entry } from "../lib/api";

/** Drop the pipeline's internal speaker labels, e.g. "(S1)", "(P2)", "SPEAKER_03". */
const clean = (t: string) => t.replace(/\s*\((?:S|P|SPEAKER_)\d+\)/g, "").replace(/\bSPEAKER_\d+\b/g, "someone").trim();

const AVATAR = ["#ffc24b", "#5be3b0", "#ff8a5c", "#6aa8ff", "#b98cff", "#ff5a5f"];

/** Before the first line, on one page: the scene, who's who, and what to do. */
export function Briefing({ entry, onStart }: { entry: Entry; onStart: () => void }) {
  const people = entry.characters.filter((c) => c.name);
  const playable = entry.lines.filter((l) => l.playable).length;
  const lang = LANG_NAME[entry.target_language];

  return (
    <motion.main initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
      className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-[max(16px,env(safe-area-inset-left))] py-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <p className="font-mono text-xs tracking-[0.25em] text-dim">THE SCENE</p>
          <h2 className="font-marquee text-4xl leading-tight">{entry.title}</h2>
        </div>
        {entry.mood && <p className="text-sm text-dim">{entry.mood}</p>}
      </div>

      <div className="overflow-hidden rounded-lg bg-black shadow-[0_20px_60px_-20px_rgba(0,0,0,.8)]">
        <video src={api.videoUrl(entry.id)} controls playsInline autoPlay className="aspect-video max-h-[42vh] w-full object-contain">
          <track kind="subtitles" src={api.subtitlesUrl(entry.id)} srcLang="und" label="Lines" default />
        </video>
      </div>

      {entry.scene && <p className="text-lg leading-relaxed text-paper/90">{clean(entry.scene)}</p>}

      {people.length > 0 && (
        <section className="flex flex-col gap-3">
          <p className="font-mono text-xs tracking-[0.25em] text-dim">WHO'S WHO</p>
          <ul className="grid gap-3 sm:grid-cols-2">
            {people.map((c, i) => {
              const name = c.name ?? "Someone";
              const facts = [c.age, c.gender].filter(Boolean).join(" · ");
              return (
                <li key={c.id} className="flex items-start gap-3">
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full font-marquee text-lg text-ink"
                    style={{ background: AVATAR[i % AVATAR.length] }} aria-hidden>
                    {name[0].toUpperCase()}
                  </span>
                  <div className="min-w-0">
                    <p className="font-bold capitalize leading-tight">{name}{facts && <span className="ml-2 font-mono text-xs font-normal text-dim">{facts}</span>}</p>
                    {c.description && c.description !== name && <p className="line-clamp-2 text-sm text-dim">{clean(c.description)}</p>}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      <div className="flex flex-col items-center gap-4 border-t border-rule/60 pt-6 sm:flex-row sm:justify-between">
        <p className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-paper/80">
          <span className="font-marquee text-paper">Dub {playable} lines into {lang}:</span>
          <span className="flex items-center gap-1.5"><Ear className="h-4 w-4 text-gold" /> listen</span>
          <span className="flex items-center gap-1.5"><Mic className="h-4 w-4 text-gold" /> say it</span>
          <span className="flex items-center gap-1.5"><Star className="h-4 w-4 text-gold" /> score</span>
        </p>
        <button type="button" onClick={onStart}
          className="rounded-lg bg-gold px-8 py-3.5 font-marquee text-lg text-ink shadow-[0_6px_0_#a86c00] active:translate-y-1 active:shadow-[0_3px_0_#a86c00]">
          START DUBBING
        </button>
      </div>
    </motion.main>
  );
}
