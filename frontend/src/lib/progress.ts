// Per-player progress, kept in this browser only.
import type { Baseline, Lang } from "./api";

const read = <T,>(key: string, fallback: T): T => {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
};
const write = (key: string, value: unknown) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: progress just isn't kept */
  }
};

export const stars = (score: number) => (score >= 85 ? 3 : score >= 70 ? 2 : score >= 50 ? 1 : 0);

export interface LastPlayed {
  scene: string;
  title: string;
  line: number;
  total: number;
}

export const progress = {
  best: (scene: string, line: number) => read<Record<string, number>>("ddg.best", {})[`${scene}#${line}`] ?? null,
  /** Saves the score if it beats the previous best. Returns the previous best (null if none). */
  record(scene: string, line: number, score: number): number | null {
    const all = read<Record<string, number>>("ddg.best", {});
    const key = `${scene}#${line}`;
    const previous = key in all ? all[key] : null;
    if (previous === null || score > previous) all[key] = score;
    write("ddg.best", all);
    return previous;
  },
  sceneStars(scene: string) {
    const all = read<Record<string, number>>("ddg.best", {});
    return Object.entries(all).reduce((s, [k, v]) => s + (k.startsWith(`${scene}#`) ? stars(v) : 0), 0);
  },
  totalStars() {
    return Object.values(read<Record<string, number>>("ddg.best", {})).reduce((s, v) => s + stars(v), 0);
  },
  streak: () => read<number>("ddg.streak", 0),
  bestStreak: () => read<number>("ddg.bestStreak", 0),
  setStreak(n: number) {
    write("ddg.streak", n);
    if (n > read<number>("ddg.bestStreak", 0)) write("ddg.bestStreak", n);
  },
  baseline: (lang: Lang) => read<Baseline | null>(`ddg.baseline.${lang}`, null),
  setBaseline: (lang: Lang, b: Baseline) => write(`ddg.baseline.${lang}`, b),
  lang: () => read<Lang | null>("ddg.lang", null),
  setLang: (lang: Lang) => write("ddg.lang", lang),
  last: () => read<LastPlayed | null>("ddg.last", null),
  setLast: (l: LastPlayed) => write("ddg.last", l),
  walkthroughSeen: () => read<boolean>("ddg.walkthrough", false),
  setWalkthroughSeen: (seen: boolean) => write("ddg.walkthrough", seen),
};
