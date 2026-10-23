import { motion } from "framer-motion";
import { Users } from "lucide-react";
import { api, type Entry } from "../lib/api";

/** Before the first line: the whole scene with subtitles, and who's who. */
export function Briefing({ entry, onStart }: { entry: Entry; onStart: () => void }) {
  const people = entry.characters.filter((c) => c.name);
  return (
    <main className="mx-auto grid w-full max-w-6xl flex-1 content-center gap-8 px-[max(16px,env(safe-area-inset-left))] py-8 lg:grid-cols-[1.4fr_1fr] lg:gap-12">
      <div className="min-w-0 overflow-hidden rounded-[3px] bg-black">
        <video src={api.videoUrl(entry.id)} controls playsInline autoPlay className="aspect-video w-full">
          <track kind="subtitles" src={api.subtitlesUrl(entry.id)} srcLang="und" label="Lines" default />
        </video>
      </div>

      <motion.section initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="flex min-w-0 flex-col justify-center gap-6">
        <div>
          <p className="font-mono text-xs tracking-[0.25em] text-dim">THE SCENE</p>
          <h2 className="font-marquee text-4xl">{entry.title}</h2>
          {entry.mood && <p className="mt-1 text-dim">{entry.mood}</p>}
        </div>

        {people.length > 0 && (
          <div className="flex flex-col gap-3">
            <p className="flex items-center gap-2 font-mono text-xs tracking-[0.25em] text-dim"><Users className="h-4 w-4" /> WHO'S WHO</p>
            <ul className="flex flex-col gap-2.5">
              {people.map((c) => (
                <li key={c.id} className="border-l-2 border-gold/60 pl-3">
                  <p className="font-bold capitalize">{c.name}</p>
                  {c.description && c.description !== c.name && <p className="line-clamp-2 text-sm text-dim">{c.description}</p>}
                </li>
              ))}
            </ul>
            {entry.relationships.length > 0 && (
              <p className="text-sm text-paper/80">
                {entry.relationships.map((r) => `${r.between.join(" & ")}: ${r.relation}`).join(" · ")}
              </p>
            )}
          </div>
        )}

        <p className="text-sm text-dim">Watch it once. Then you'll dub it line by line, in your language.</p>
        <button type="button" onClick={onStart}
          className="self-start rounded-lg bg-gold px-8 py-3.5 font-marquee text-lg text-ink shadow-[0_6px_0_#a86c00] active:translate-y-1 active:shadow-[0_3px_0_#a86c00]">
          START DUBBING
        </button>
      </motion.section>
    </main>
  );
}
