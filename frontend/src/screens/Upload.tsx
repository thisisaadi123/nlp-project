import { AnimatePresence, motion } from "framer-motion";
import { AudioLines, Check, Clapperboard, Eye, Film, Languages, Loader2, Play, Upload as UploadIcon } from "lucide-react";
import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import { api, LANG_NAME, LANG_NATIVE, LANGS, type Lang, type UploadStatus } from "../lib/api";

const MAX_SECONDS = 45;   // config.yaml game.upload.max_seconds
const MAX_MB = 80;
const ICONS: Record<string, typeof Eye> = { check: Film, watch: Eye, hear: AudioLines, translate: Languages, voices: Clapperboard };

/** Turn the player's own clip into a scene: pick it, then watch it being made, then play it. */
export function Upload({ lang, onPlay }: { lang: Lang; onPlay: (entryId: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [seconds, setSeconds] = useState<number | null>(null);
  const [source, setSource] = useState<Lang | null>(null);
  const [title, setTitle] = useState("");
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<UploadStatus | null>(null);
  const [sending, setSending] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  // poll the job until it's ready or failed
  useEffect(() => {
    if (!job || job.state !== "working") return;
    const t = setTimeout(() => api.uploadStatus(job.id).then(setJob).catch(() => {}), 2000);
    return () => clearTimeout(t);
  }, [job]);

  const choose = (f: File | undefined) => {
    setError(null);
    if (!f) return;
    if (!f.type.startsWith("video/")) return setError("That isn't a video file.");
    if (f.size > MAX_MB * 2 ** 20) return setError(`Clips must be under ${MAX_MB} MB.`);
    setFile(f);
    setSeconds(null);
    setPreview(URL.createObjectURL(f));
    if (!title) setTitle(f.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").slice(0, 60));
  };

  const tooLong = seconds !== null && seconds > MAX_SECONDS;
  const ready = file && consent && !tooLong && !sending;

  const send = async () => {
    if (!file) return;
    setSending(true);
    setError(null);
    try {
      const { id } = await api.upload(file, lang, source, title);
      setJob(await api.uploadStatus(id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };

  if (job) return <Progress job={job} lang={lang} onPlay={onPlay} onAgain={() => { setJob(null); setFile(null); setPreview(null); }} />;

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-[max(16px,env(safe-area-inset-left))] py-6">
      <div>
        <p className="font-mono text-xs tracking-[0.25em] text-dim">YOUR OWN SCENE</p>
        <h2 className="font-marquee text-4xl leading-tight">Dub a clip you love</h2>
        <p className="mt-1 text-paper/80">Upload a movie scene (up to {MAX_SECONDS} s). We'll translate it into {LANG_NAME[lang]} and set it up for you to dub.</p>
      </div>

      {/* the clip: a drop zone, then the preview */}
      {preview ? (
        <div className="overflow-hidden rounded-lg bg-black shadow-[0_20px_60px_-20px_rgba(0,0,0,.8)]">
          <video src={preview} controls playsInline className="aspect-video max-h-[40vh] w-full object-contain"
            onLoadedMetadata={(e) => setSeconds(e.currentTarget.duration)} />
        </div>
      ) : (
        <button type="button" onClick={() => input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]); }}
          className={clsx("flex aspect-video max-h-[40vh] w-full flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed transition-colors",
            dragging ? "border-gold bg-gold/10" : "border-rule hover:border-paper/60")}>
          <UploadIcon className="h-10 w-10 text-gold" />
          <span className="font-marquee text-xl">DROP A CLIP HERE</span>
          <span className="text-sm text-dim">or click to choose · mp4, mov, webm · up to {MAX_SECONDS} s</span>
        </button>
      )}
      <input ref={input} type="file" accept="video/*" className="hidden" onChange={(e) => choose(e.target.files?.[0])} />

      {file && (
        <div className="flex flex-col gap-5">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <span className={clsx(tooLong ? "text-goose" : "text-dim")}>
              {file.name}{seconds !== null && ` · ${Math.round(seconds)} s`}{tooLong && ` · too long, trim it to ${MAX_SECONDS} s`}
            </span>
            <button type="button" onClick={() => input.current?.click()} className="font-mono text-xs text-dim hover:text-paper">CHOOSE ANOTHER</button>
          </div>

          <label className="flex flex-col gap-1.5">
            <span className="font-mono text-[11px] tracking-[0.25em] text-dim">TITLE</span>
            <input value={title} onChange={(e) => setTitle(e.target.value.slice(0, 60))} placeholder="e.g. Brothers' fight"
              className="rounded-md border border-rule bg-ink-2 px-3 py-2.5 text-paper outline-none focus:border-gold" />
          </label>

          <div className="flex flex-col gap-2">
            <span className="font-mono text-[11px] tracking-[0.25em] text-dim">SPOKEN IN</span>
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={() => setSource(null)}
                className={clsx("rounded-md border px-3 py-1.5 text-sm", source === null ? "border-gold bg-gold text-ink" : "border-rule hover:border-paper")}>
                Detect it
              </button>
              {LANGS.filter((l) => l !== lang).map((l) => (
                <button key={l} type="button" onClick={() => setSource(l)}
                  className={clsx("rounded-md border px-3 py-1.5 font-script text-sm", source === l ? "border-gold bg-gold text-ink" : "border-rule hover:border-paper")}>
                  {LANG_NATIVE[l]}
                </button>
              ))}
            </div>
          </div>

          <label className="flex cursor-pointer items-start gap-3 text-sm text-paper/80">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-0.5 h-4 w-4 accent-[#ffc24b]" />
            <span>This clip is family-friendly (no swearing, fights, blood, kissing, drinking or smoking) and I'm allowed to use it.</span>
          </label>

          <div className="flex flex-wrap items-center gap-4">
            <button type="button" disabled={!ready} onClick={send}
              className="flex items-center gap-2 rounded-lg bg-gold px-7 py-3.5 font-marquee text-lg text-ink shadow-[0_6px_0_#a86c00] active:translate-y-1 disabled:opacity-30">
              {sending ? <><Loader2 className="h-5 w-5 animate-spin" /> UPLOADING…</> : "MAKE MY SCENE"}
            </button>
            <span className="text-sm text-dim">Takes about 1-2 minutes.</span>
          </div>
        </div>
      )}
      {error && <p className="text-goose">{error}</p>}
    </main>
  );
}

/** The scene being made, step by step, then PLAY IT (or what went wrong). */
function Progress({ job, lang, onPlay, onAgain }: { job: UploadStatus; lang: Lang; onPlay: (id: string) => void; onAgain: () => void }) {
  const at = job.state === "ready" ? job.steps.length : Math.max(0, job.steps.indexOf(job.step));
  return (
    <main className="mx-auto flex w-full max-w-xl flex-1 flex-col justify-center gap-8 px-[max(16px,env(safe-area-inset-left))] py-8">
      <div>
        <p className="font-mono text-xs tracking-[0.25em] text-dim">MAKING YOUR SCENE</p>
        <h2 className="font-marquee text-4xl leading-tight">{job.title}</h2>
        <p className="mt-1 text-dim">Dubbed into {LANG_NAME[lang]}</p>
      </div>

      <ol className="flex flex-col gap-4">
        {job.steps.map((key, i) => {
          const Icon = ICONS[key] ?? Film;
          const done = i < at;
          const now = i === at && job.state === "working";
          const failed = i === at && job.state === "failed";
          return (
            <motion.li key={key} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.06 }}
              className="flex items-center gap-4">
              <span className={clsx("grid h-10 w-10 shrink-0 place-items-center rounded-full border",
                done ? "border-mint bg-mint/15 text-mint" : now ? "border-gold text-gold" : failed ? "border-goose text-goose" : "border-rule text-dim/60")}>
                {done ? <Check className="h-5 w-5" /> : now ? <Loader2 className="h-5 w-5 animate-spin" /> : <Icon className="h-5 w-5" />}
              </span>
              <span className={clsx("text-lg", done ? "text-paper/80" : now ? "text-paper" : failed ? "text-goose" : "text-dim/60")}>
                {job.labels[key] ?? key}
              </span>
            </motion.li>
          );
        })}
      </ol>

      <AnimatePresence>
        {job.state === "ready" && job.entry && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap items-center gap-4">
            <button type="button" onClick={() => onPlay(job.entry!)}
              className="flex items-center gap-2 rounded-lg bg-gold px-8 py-3.5 font-marquee text-lg text-ink shadow-[0_6px_0_#a86c00] active:translate-y-1">
              <Play className="h-5 w-5 fill-ink" /> PLAY IT
            </button>
            <span className="text-sm text-dim">It's also in your scene list now.</span>
          </motion.div>
        )}
        {job.state === "failed" && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-start gap-3">
            <p className="text-goose">{job.error}</p>
            <button type="button" onClick={onAgain} className="rounded-md border border-rule px-5 py-2.5 font-mono text-sm hover:border-paper">TRY ANOTHER CLIP</button>
          </motion.div>
        )}
      </AnimatePresence>
      {job.state === "working" && <p className="text-sm text-dim">This takes about 1-2 minutes. You can keep this page open.</p>}
    </main>
  );
}
