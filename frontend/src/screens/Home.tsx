import { animate, AnimatePresence, motion } from "framer-motion";
import { AudioLines, Clock3, Ear, Flame, Mic, Music2, Play, Star, Type, Upload, Volume2, Zap } from "lucide-react";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { LanguagePicker } from "../components/Bits";
import { Goose } from "../components/Goose";
import { Logo } from "../components/Logo";
import { Stars } from "../components/Script";
import { LANG_NATIVE, LANGS, type Lang } from "../lib/api";
import { DUB_AT, progress, stars } from "../lib/progress";
import { sfx } from "../lib/sound";

const reveal = { initial: { opacity: 0, y: 30 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, margin: "-80px" },
  transition: { duration: 0.6 } };

/** The home page: pick a language and start, and learn the game by playing with it. */
export function Home({ lang, onLang, onStart, onContinue }: {
  lang: Lang | null; onLang: (l: Lang) => void; onStart: () => void; onContinue: (scene: string) => void;
}) {
  const last = progress.last();
  const total = progress.totalStars();
  const best = progress.bestStreak();
  const start = () => {
    if (!lang) return document.getElementById("pick-language")?.scrollIntoView({ behavior: "smooth", block: "center" });
    sfx.clap();
    onStart();
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Enter" && (e.target as HTMLElement).tagName !== "INPUT" && start();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <main className="flex w-full flex-1 flex-col items-center overflow-x-hidden">
      {/* ---------- hero ---------- */}
      <section className="relative flex min-h-[calc(100vh-4rem)] w-full flex-col items-center justify-center gap-9 px-6 py-14 text-center">
        <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-24 mx-auto h-[70vh] w-[60vw] animate-sweep rounded-full bg-gold/10 blur-3xl" />
        <motion.div initial={{ y: -30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ type: "spring", stiffness: 120 }}>
          <Logo big />
        </motion.div>
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="max-w-xl text-xl leading-relaxed text-paper/90">
          Real movie scenes. Your voice. Your language.<br />
          Nail the line and it's a <span className="font-bold text-gold">Dub</span>. Miss it and you get{" "}
          <span className="font-bold text-goose">Goosed</span>.
        </motion.p>

        <div id="pick-language" className="flex flex-col items-center gap-4">
          <span className="font-mono text-xs tracking-[0.25em] text-dim">I'LL DUB INTO</span>
          <LanguagePicker value={lang} onChange={onLang} />
        </div>

        <div className="flex flex-col items-center gap-3">
          <motion.button type="button" onClick={start} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97, y: 4 }}
            className={clsx("rounded-lg bg-gold px-10 py-4 font-marquee text-xl text-ink shadow-[0_8px_0_#a86c00] active:shadow-[0_4px_0_#a86c00]",
              !lang && "opacity-40")}>
            START DUBBING
          </motion.button>
          {!lang && <span className="text-sm text-dim">Pick a language first.</span>}
          {last && (
            <button type="button" onClick={() => onContinue(last.scene)} className="flex items-center gap-2 font-mono text-sm text-dim hover:text-paper">
              <Play className="h-3.5 w-3.5 fill-current" /> Continue: {last.title} · line {last.line + 1}/{last.total}
            </button>
          )}
          {(total > 0 || best > 0) && (
            <p className="flex items-center gap-5 font-mono text-sm text-dim">
              <span className="flex items-center gap-1.5"><Star className="h-4 w-4 fill-gold text-gold" /> {total} stars</span>
              <span className="flex items-center gap-1.5"><Flame className="h-4 w-4 text-goose" /> best streak {best}</span>
            </p>
          )}
        </div>
        <motion.a href="#how" animate={{ y: [0, 8, 0] }} transition={{ repeat: Infinity, duration: 2 }}
          className="absolute bottom-6 font-mono text-[11px] tracking-[0.3em] text-dim hover:text-paper">HOW IT WORKS ↓</motion.a>
      </section>

      <Ribbon />

      <div className="flex w-full max-w-5xl flex-col gap-28 px-6 py-24">
        <RoundDemo />
        <DubOrGoose />
        <Listening />
        <Context />
        <Extras />

        {/* ---------- last call ---------- */}
        <motion.section {...reveal} className="flex flex-col items-center gap-5 text-center">
          <Goose size={90} honking />
          <h2 className="font-marquee text-4xl">Ready for your first take?</h2>
          <motion.button type="button" onClick={start} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97, y: 4 }}
            className="rounded-lg bg-gold px-10 py-4 font-marquee text-xl text-ink shadow-[0_8px_0_#a86c00]">
            {lang ? "START DUBBING" : "PICK A LANGUAGE"}
          </motion.button>
          <p className="max-w-md text-xs text-dim/70">
            Scenes are short excerpts used for research. Your takes are scored on our server and their words checked
            by Sarvam's speech recognition; recordings aren't kept.
          </p>
        </motion.section>
      </div>
    </main>
  );
}

function SectionTitle({ kicker, title, children }: { kicker: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-mono text-xs tracking-[0.3em] text-gold">{kicker}</p>
      <h2 className="font-marquee text-4xl leading-tight sm:text-5xl">{title}</h2>
      {children && <p className="max-w-2xl text-lg text-paper/80">{children}</p>}
    </div>
  );
}

/** The five languages scrolling past in their own scripts. */
function Ribbon() {
  const words = [...LANGS.map((l) => LANG_NATIVE[l]), "DUB", "GOOSE"];
  const row = [...words, ...words, ...words];
  return (
    <div className="w-full overflow-hidden border-y border-rule/60 bg-ink-2/70 py-4" aria-hidden>
      <div className="flex w-max animate-ribbon gap-10 whitespace-nowrap">
        {[...row, ...row].map((w, i) => (
          <span key={i} className={clsx("font-script text-2xl font-bold", w === "DUB" ? "font-marquee text-gold" : w === "GOOSE" ? "font-marquee text-goose" : "text-paper/40")}>
            {w}
          </span>
        ))}
      </div>
    </div>
  );
}

const BEATS = [
  { icon: Ear, title: "Listen", text: "The scene plays the original line. A note tells you how it's said: the feeling, how loud, the word to punch. Tap \"hear it in Tamil\" to hear your line read aloud." },
  { icon: Mic, title: "Say it", text: "3, 2, 1: the scene plays again without sound. Say your line when the actor speaks, as if you were dubbing the film." },
  { icon: Star, title: "Score", text: "In a few seconds you see your score out of 100, what you got right, and the one thing to fix next take." },
];

/** A round in three beats, playing itself; click a beat to jump to it. */
function RoundDemo() {
  const [beat, setBeat] = useState(0);
  const [auto, setAuto] = useState(true);
  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => setBeat((b) => (b + 1) % 3), 3600);
    return () => clearInterval(t);
  }, [auto]);
  return (
    <motion.section id="how" {...reveal} className="flex flex-col gap-10">
      <SectionTitle kicker="A ROUND, IN 3 BEATS" title="Listen. Say it. Score." />
      <div className="grid items-center gap-10 lg:grid-cols-[1.1fr_1fr]">
        {/* the mock booth */}
        <div className="relative aspect-video overflow-hidden rounded-xl border border-rule bg-gradient-to-br from-ink-3 to-ink shadow-[0_30px_80px_-30px_rgba(0,0,0,.9)]">
          <AnimatePresence mode="wait">
            <motion.div key={beat} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
              className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6">
              {beat === 0 && (
                <>
                  <div className="flex h-16 items-center gap-1.5">
                    {Array.from({ length: 18 }, (_, i) => (
                      <span key={i} className="w-1.5 origin-center animate-bars rounded-full bg-gold" style={{ height: "100%", animationDelay: `${(i % 6) * 0.12}s` }} />
                    ))}
                  </div>
                  <p className="font-marquee text-lg">😠 Say it <span className="text-[#ff5a5f]">angrily</span> · 🔊 Loud · ⏱ ~2 s</p>
                  <p className="font-script text-2xl font-bold"><span className="text-gold">அண்ணா</span>, நீ என்ன பண்ணிட்ட?</p>
                </>
              )}
              {beat === 1 && (
                <>
                  <motion.span animate={{ scale: [1, 1.12, 1] }} transition={{ repeat: Infinity, duration: 1 }}
                    className="grid h-20 w-20 place-items-center rounded-full bg-onair shadow-[0_0_40px_rgba(255,61,85,.6)]">
                    <Mic className="h-9 w-9" />
                  </motion.span>
                  <p className="rounded bg-onair px-2 py-1 font-marquee text-sm">SPEAK NOW</p>
                  <div className="relative h-3 w-3/4 overflow-hidden rounded-full bg-ink-3">
                    <div className="absolute inset-y-0 left-[20%] w-[55%] rounded-full bg-onair/40" />
                    <motion.div initial={{ width: "0%" }} animate={{ width: "100%" }} transition={{ duration: 3.4, ease: "linear" }}
                      className="absolute inset-y-0 left-0 bg-paper/30" />
                  </div>
                  <p className="font-mono text-[10px] tracking-widest text-dim">THE ACTOR SPEAKS IN THE RED PART</p>
                </>
              )}
              {beat === 2 && <DemoScore />}
            </motion.div>
          </AnimatePresence>
        </div>
        {/* the beats */}
        <ol className="flex flex-col gap-2">
          {BEATS.map(({ icon: Icon, title, text }, i) => (
            <li key={title}>
              <button type="button" onClick={() => { setAuto(false); setBeat(i); }}
                className={clsx("flex w-full items-start gap-4 rounded-lg p-4 text-left transition-colors", beat === i ? "bg-ink-2" : "hover:bg-ink-2/50")}>
                <span className={clsx("grid h-11 w-11 shrink-0 place-items-center rounded-full border transition-colors",
                  beat === i ? "border-gold bg-gold text-ink" : "border-rule text-gold")}>
                  <Icon className="h-5 w-5" />
                </span>
                <span className="flex flex-col gap-1">
                  <span className="font-marquee text-xl">{i + 1}. {title.toUpperCase()}</span>
                  <span className={clsx("text-paper/75 transition-all", beat === i ? "max-h-40 opacity-100" : "max-h-0 overflow-hidden opacity-0 lg:max-h-40 lg:opacity-60")}>{text}</span>
                </span>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </motion.section>
  );
}

function DemoScore() {
  const [n, setN] = useState(0);
  useEffect(() => {
    const c = animate(0, 84, { duration: 1.2, ease: "easeOut", onUpdate: (v) => setN(Math.round(v)) });
    return () => c.stop();
  }, []);
  return (
    <div className="flex items-center gap-6">
      <span className="font-marquee text-7xl tabular-nums">{n}</span>
      <motion.span initial={{ scale: 3, rotate: -25, opacity: 0 }} animate={{ scale: 1, rotate: -8, opacity: 1 }} transition={{ delay: 1.2, type: "spring" }}
        className="rounded-md border-4 border-gold px-3 py-1 font-marquee text-3xl text-gold">DUB!</motion.span>
    </div>
  );
}

/** Drag the score and watch the verdict change. */
function DubOrGoose() {
  const [score, setScore] = useState(62);
  const dub = score >= DUB_AT;
  return (
    <motion.section {...reveal} className="flex flex-col gap-10">
      <SectionTitle kicker="THE VERDICT" title="Dub or Goose?">
        Every take gets a score out of 100. Drag the slider to see what each score means.
      </SectionTitle>
      <div className="grid items-center gap-10 rounded-2xl border border-rule bg-ink-2/60 p-6 sm:p-10 lg:grid-cols-2">
        <div className="flex flex-col items-center gap-6">
          <div className="flex h-36 items-center gap-6">
            <span className="font-marquee text-8xl tabular-nums">{score}</span>
            <AnimatePresence mode="wait">
              {dub ? (
                <motion.div key="dub" initial={{ scale: 2.5, rotate: -25, opacity: 0 }} animate={{ scale: 1, rotate: -8, opacity: 1 }} exit={{ opacity: 0, scale: 0.6 }}
                  transition={{ type: "spring", stiffness: 260, damping: 14 }} className="flex flex-col items-center gap-2">
                  <span className="rounded-md border-4 border-gold px-4 py-1 font-marquee text-4xl text-gold shadow-[0_0_30px_rgba(255,194,75,.35)]">DUB!</span>
                  <Stars n={stars(score)} size={22} />
                </motion.div>
              ) : (
                <motion.div key="goose" initial={{ x: 80, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center">
                  <Goose size={96} honking />
                  <span className="font-marquee text-3xl text-goose">GOOSE</span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <div className="w-full">
            <input type="range" min={0} max={100} value={score} onChange={(e) => setScore(+e.target.value)} aria-label="score"
              className="w-full accent-[#ffc24b]" />
            <div className="relative mt-1 h-5 font-mono text-[10px] tracking-widest text-dim">
              <span className="absolute left-0">0</span>
              {[DUB_AT, 80, 90].map((v) => (
                <span key={v} className="absolute -translate-x-1/2 text-gold" style={{ left: `${v}%` }}>{v}</span>
              ))}
              <span className="absolute right-0">100</span>
            </div>
          </div>
        </div>
        <ul className="flex flex-col gap-5 text-lg">
          <li className="flex gap-4">
            <span className="w-24 shrink-0 font-marquee text-gold">DUB!</span>
            <span className="text-paper/85"><b>{DUB_AT} or more.</b> The line is yours: it sounds like it belongs in the film.
              <span className="mt-1 flex items-center gap-3 text-base text-dim"><Stars n={1} size={14} /> {DUB_AT}+ <Stars n={2} size={14} /> 80+ <Stars n={3} size={14} /> 90+</span></span>
          </li>
          <li className="flex gap-4">
            <span className="w-24 shrink-0 font-marquee text-goose">GOOSE</span>
            <span className="text-paper/85"><b>Under {DUB_AT}.</b> The goose got you. Retake it as often as you like: your best take counts, and every Dub in a row grows your streak.</span>
          </li>
          <li className="flex gap-4">
            <span className="w-24 shrink-0 font-marquee text-paper">SCENE</span>
            <span className="text-paper/85">Finish every line and the scene's score is your average. Then watch your <b>premiere</b>: the scene with your voice in it.</span>
          </li>
        </ul>
      </div>
    </motion.section>
  );
}

const PARTS = [
  { icon: Clock3, name: "Timing", what: "Start and stop when the actor does, so your voice fits their lips.", good: "on time", bad: "0.8 s late" },
  { icon: Volume2, name: "Volume", what: "Soft, normal, loud or shouting, as the scene asks. Measured against your own calm voice from the mic check.", good: "loud", bad: "too soft: needs SHOUTING" },
  { icon: Zap, name: "Punch", what: "Hit the gold word harder than the rest, where the actor did.", good: "on the word", bad: "too early: put it on அண்ணா" },
  { icon: Music2, name: "Tone", what: "Let your voice rise, drop or stay level at the end, like the actor's.", good: "matched", bad: "ends going up; the actor's drops" },
  { icon: Type, name: "Words", what: "Say the line as written, in the right language. Wrong or skipped words are marked so you can fix them.", good: "every word", bad: "we heard புள்ள for பொண்ணு" },
];

function Listening() {
  const [open, setOpen] = useState(0);
  return (
    <motion.section {...reveal} className="flex flex-col gap-10">
      <SectionTitle kicker="HOW YOU'RE SCORED" title="What we listen for">
        Five things, each one you can hear and fix. After every take you see how each went, and one tip.
      </SectionTitle>
      <ul className="flex flex-col divide-y divide-rule/60 border-y border-rule/60">
        {PARTS.map(({ icon: Icon, name, what, good, bad }, i) => (
          <li key={name}>
            <button type="button" onClick={() => setOpen(i)} onMouseEnter={() => setOpen(i)}
              className="grid w-full grid-cols-[2.5rem_7rem_1fr] items-start gap-4 py-5 text-left sm:grid-cols-[2.5rem_8rem_1fr_16rem]">
              <Icon className={clsx("mt-1 h-6 w-6 transition-colors", open === i ? "text-gold" : "text-dim")} />
              <span className="font-marquee text-xl">{name.toUpperCase()}</span>
              <span className="text-paper/80">{what}</span>
              <span className={clsx("col-span-3 flex flex-col gap-1 font-mono text-sm transition-opacity sm:col-span-1", open === i ? "opacity-100" : "opacity-40")}>
                <span className="text-mint">✓ {good}</span>
                <span className="text-goose">✗ {bad}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </motion.section>
  );
}

const EXAMPLES = [
  { who: "to a friend", hindi: "भाई, चल ना!", tamil: "மச்சான், வா டா!", note: "bhai between friends is machan, never anna" },
  { who: "to an elder brother", hindi: "भैया, चलिए ना!", tamil: "அண்ணா, வாங்க!", note: "bhaiya is anna, with the respectful vaanga" },
  { who: "angry, to an elder brother", hindi: "भैया! ये क्या किया?!", tamil: "அண்ணா! என்ன பண்ணிட்ட?!", note: "same word, said hard: the note says shouting" },
];

/** The same Hindi word, three relationships, three Tamil lines. */
function Context() {
  const [i, setI] = useState(0);
  const ex = EXAMPLES[i];
  return (
    <motion.section {...reveal} className="flex flex-col gap-10">
      <SectionTitle kicker="NOT GOOGLE TRANSLATE" title="Who's talking to whom matters">
        Every scene is translated with its context: the characters, how they're related, the mood. So the words and the
        politeness change with the relationship, and the lines sound spoken, not written.
      </SectionTitle>
      <div className="flex flex-col gap-6 rounded-2xl border border-rule bg-ink-2/60 p-6 sm:p-10">
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((e, k) => (
            <button key={e.who} type="button" onClick={() => setI(k)}
              className={clsx("rounded-md border px-4 py-2 text-sm transition-colors", k === i ? "border-gold bg-gold text-ink" : "border-rule hover:border-paper")}>
              {e.who}
            </button>
          ))}
        </div>
        <AnimatePresence mode="wait">
          <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="grid items-center gap-4 sm:grid-cols-[1fr_auto_1fr]">
            <div><p className="font-mono text-[11px] tracking-widest text-dim">HINDI</p><p className="font-script text-3xl font-bold">{ex.hindi}</p></div>
            <span className="font-marquee text-3xl text-gold">→</span>
            <div><p className="font-mono text-[11px] tracking-widest text-dim">TAMIL DUB</p><p className="font-script text-3xl font-bold text-gold">{ex.tamil}</p></div>
            <p className="text-dim sm:col-span-3">{ex.note}</p>
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.section>
  );
}

function Extras() {
  const items = [
    { icon: AudioLines, title: "Hear it in your language", text: "Every line has a spoken example in your language, timed to the actor, so you know how it should sound before you try." },
    { icon: Upload, title: "Bring your own scene", text: "Upload a family-friendly clip (up to 45 seconds). We find who says what, translate it and set it up for you to dub in a minute or two." },
    { icon: Star, title: "Stars, streaks, premieres", text: "Earn up to three stars a line, keep a streak of Dubs going, and watch the finished scene with your voice in it." },
  ];
  return (
    <motion.section {...reveal} className="grid gap-10 sm:grid-cols-3">
      {items.map(({ icon: Icon, title, text }, i) => (
        <motion.div key={title} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
          transition={{ delay: i * 0.12 }} className="flex flex-col gap-3">
          <Icon className="h-8 w-8 text-gold" />
          <h3 className="font-marquee text-xl">{title.toUpperCase()}</h3>
          <p className="text-paper/75">{text}</p>
        </motion.div>
      ))}
    </motion.section>
  );
}
