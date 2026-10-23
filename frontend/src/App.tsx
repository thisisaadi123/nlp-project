import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, CircleHelp, Flame, Star, Volume2, VolumeX } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { HelpOverlay, LanguagePicker } from "./components/Bits";
import { Logo } from "./components/Logo";
import { api, LANG_NATIVE, type Baseline, type Entry, type Lang, type Scene } from "./lib/api";
import { progress } from "./lib/progress";
import { isMuted, setMuted } from "./lib/sound";
import { Booth, type Take } from "./screens/Booth";
import { Briefing } from "./screens/Briefing";
import { Home } from "./screens/Home";
import { MicCheck } from "./screens/MicCheck";
import { Reel } from "./screens/Reel";
import { SceneEnd } from "./screens/SceneEnd";

type Screen =
  | { name: "home" }
  | { name: "reel" }
  | { name: "mic"; entry: Entry; next: "briefing" | "booth"; index: number }
  | { name: "briefing"; entry: Entry }
  | { name: "booth"; entry: Entry; index: number }
  | { name: "end"; entry: Entry };

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: "home" });
  const [lang, setLangState] = useState<Lang | null>(progress.lang());
  const [streak, setStreak] = useState(progress.streak());
  const [muted, setMute] = useState(isMuted());
  const [help, setHelp] = useState(false);
  const [langMenu, setLangMenu] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [takes, setTakes] = useState<Record<string, Record<number, Take>>>({});
  const menu = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const close = (e: MouseEvent) => menu.current && !menu.current.contains(e.target as Node) && setLangMenu(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const setLang = (l: Lang) => {
    setLangState(l);
    progress.setLang(l);
    setLangMenu(false);
    if (screen.name !== "home") setScreen({ name: "reel" });
  };

  const baselineFor = (entry: Entry): Baseline | null => progress.baseline(entry.target_language);

  /** Open a scene: mic check first if this language has no voice baseline yet. */
  const open = async (sceneId: string, index: number | null) => {
    setError(null);
    try {
      const entry = await api.entry(sceneId);
      const next = index === null ? "briefing" : "booth";
      if (!baselineFor(entry)) setScreen({ name: "mic", entry, next, index: index ?? 0 });
      else setScreen(next === "briefing" ? { name: "briefing", entry } : { name: "booth", entry, index: index ?? 0 });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const onStreak = (dub: boolean) => {
    const n = dub ? streak + 1 : 0;
    setStreak(n);
    progress.setStreak(n);
  };

  const keepTake = (entryId: string) => (line: number, take: Take) =>
    setTakes((all) => {
      const scene = all[entryId] ?? {};
      if (scene[line] && scene[line].score >= take.score) return all;
      return { ...all, [entryId]: { ...scene, [line]: take } };
    });

  const back = () => setScreen(screen.name === "reel" ? { name: "home" } : { name: "reel" });

  return (
    <div className="flex min-h-full flex-col">
      {screen.name !== "home" && (
        <header className="sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-rule/60 bg-ink/85 px-[max(16px,env(safe-area-inset-left))] py-3 backdrop-blur">
          <div className="flex items-center gap-2">
            <button type="button" aria-label="Back" onClick={back} className="grid h-8 w-8 place-items-center rounded-full text-dim hover:bg-ink-3 hover:text-paper">
              <ChevronLeft className="h-5 w-5" />
            </button>
            <button type="button" onClick={() => setScreen({ name: "home" })} aria-label="Home"><Logo /></button>
          </div>
          <div className="flex items-center gap-3 font-mono text-sm sm:gap-4">
            <div className="relative" ref={menu}>
              <button type="button" onClick={() => setLangMenu(!langMenu)} aria-label="Change language" aria-expanded={langMenu}
                className="rounded-md border border-rule px-2 py-0.5 font-script text-sm hover:border-paper">
                {lang ? LANG_NATIVE[lang] : "Language"}
              </button>
              <AnimatePresence>
                {langMenu && (
                  <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="absolute right-0 top-10 z-30 w-[min(92vw,26rem)] rounded-lg border border-rule bg-ink-2 p-3 shadow-2xl">
                    <LanguagePicker value={lang} onChange={setLang} size="sm" />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <span className={streak ? "flex items-center gap-1 text-goose" : "flex items-center gap-1 text-dim"} aria-label={`streak ${streak}`}>
              <Flame className="h-4 w-4" /> {streak}
            </span>
            <span className="flex items-center gap-1" aria-label="stars"><Star className="h-4 w-4 fill-gold text-gold" /> {progress.totalStars()}</span>
            <button type="button" aria-label={muted ? "Sound on" : "Sound off"} className="text-dim hover:text-paper"
              onClick={() => { setMuted(!muted); setMute(!muted); }}>
              {muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
            </button>
            <button type="button" aria-label="How to play" className="text-dim hover:text-paper"
              onClick={() => { setHelp(true); progress.setWalkthroughSeen(false); }}>
              <CircleHelp className="h-4 w-4" />
            </button>
          </div>
        </header>
      )}
      {error && <p className="px-6 pt-4 text-goose">{error}</p>}
      <HelpOverlay open={help} onClose={() => setHelp(false)} />

      <AnimatePresence mode="wait">
        <motion.div key={screen.name + ("entry" in screen ? screen.entry.id : "") + ("index" in screen ? screen.index : "")}
          className="flex flex-1 flex-col" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
          {screen.name === "home" && (
            <Home lang={lang} onLang={(l) => { setLangState(l); progress.setLang(l); }}
              onStart={() => setScreen({ name: "reel" })} onContinue={(scene) => open(scene, progress.last()?.line ?? 0)} />
          )}
          {screen.name === "reel" && lang && (
            <Reel lang={lang} onPick={(s: Scene) => open(s.id, null)} onChangeLang={() => setLangMenu(true)} />
          )}
          {screen.name === "mic" && (
            <MicCheck lang={screen.entry.target_language} onDone={() =>
              setScreen(screen.next === "briefing" ? { name: "briefing", entry: screen.entry } : { name: "booth", entry: screen.entry, index: screen.index })} />
          )}
          {screen.name === "briefing" && <Briefing entry={screen.entry} onStart={() => setScreen({ name: "booth", entry: screen.entry, index: 0 })} />}
          {screen.name === "booth" && (
            <Booth entry={screen.entry} baseline={baselineFor(screen.entry)} startIndex={screen.index}
              onTake={keepTake(screen.entry.id)} onStreak={onStreak} onFinish={() => setScreen({ name: "end", entry: screen.entry })} />
          )}
          {screen.name === "end" && (
            <SceneEnd entry={screen.entry} takes={takes[screen.entry.id] ?? {}}
              onRetry={(i) => setScreen({ name: "booth", entry: screen.entry, index: i })}
              onHome={() => setScreen({ name: "home" })} onScenes={() => setScreen({ name: "reel" })} />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
