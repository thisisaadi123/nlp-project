// Calls to the backend (backend/api/app.py).

export type Lang = "hi" | "ta" | "bn" | "mr" | "te";
export const LANGS: Lang[] = ["hi", "ta", "bn", "mr", "te"];

export interface Scene {
  id: string;
  clip: string;
  title: string;
  source_language: Lang;
  target_language: Lang;
  scene: string | null;
  mood: string | null;
  difficulty: "easy" | "medium" | "hard";
  lines: number;
  characters: string[];
  uploaded: boolean;   // the player's own clip
}

/** Progress of turning a player's clip into a scene (backend/game/upload.py). */
export interface UploadStatus {
  id: string;
  state: "working" | "ready" | "failed";
  step: string;
  steps: string[];
  labels: Record<string, string>;
  title: string;
  error?: string;
  entry?: string;
}

/** The melody line of one delivery (backend/game/shape.py). Time 0 = first voiced sound. */
export interface Shape {
  duration: number;
  step: number;
  pitch: number[];
  loud: number[];
  voiced: number[];
  stress_t: number;
  words: { w: string; start: number; end: number; stress: boolean }[];
}

export interface Card {
  line_id: number;
  speaker: string | null;
  addressee: string | null;
  speaker_name: string | null;
  addressee_name: string | null;
  start: number;
  end: number;
  duration: number;
  onset: number;
  source_text: string;
  source_roman: string;
  translation: string;
  romanized: string;
  emotion: string | null;
  intensity: "low" | "medium" | "high" | null;
  tone: string | null;
  stress: string[];
  delivery: string;
  playable: boolean;
  voice?: boolean;   // a spoken example of the dubbed line exists (Sarvam Bulbul)
  shape: Shape | null;
}

export interface Character {
  id: string;
  name: string | null;
  description: string | null;
  gender?: string | null;
  age?: string | null;
}

export interface Entry {
  id: string;
  clip: string;
  title: string;
  source_language: Lang;
  target_language: Lang;
  scene: string | null;
  mood: string | null;
  difficulty: Scene["difficulty"];
  characters: Character[];
  relationships: { between: string[]; relation: string | null; closeness: string | null }[];
  calibration_text: string;
  lines: Card[];
}

export type Baseline = Record<string, number>;

/** One part of the director's-note score (backend/scoring/director.py). */
export interface ScorePart {
  score: number;    // 0..100
  verdict: string;  // e.g. "on time", "too soft"
  tip: string;      // what to fix; "" when it was right
}

export interface ScoreResult {
  score: number;
  verdict: "Dub" | "Goose";
  parts: Record<string, ScorePart>;
  components: Record<string, number>;
  tips: string[];
  heard: string | null;
  /** The line word by word: said right, or what was heard instead (null when words weren't checked). */
  word_check: { word: string; ok: boolean; heard: string | null }[] | null;
  heard_roman: string | null;
  duration: { yours: number; original: number };
  calibrated: boolean;
  shape: Shape;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Something went wrong (${res.status}).`;
    try {
      message = (await res.json()).detail ?? message;
    } catch {
      /* not JSON */
    }
    throw new Error(message);
  }
  return res.json();
}

const enc = encodeURIComponent;

export const api = {
  scenes: (lang?: Lang) => fetch(`/api/clips${lang ? `?lang=${lang}` : ""}`).then((r) => json<Scene[]>(r)),
  entry: (id: string) => fetch(`/api/clips/${enc(id)}`).then((r) => json<Entry>(r)),
  videoUrl: (id: string) => `/api/clips/${enc(id)}/video`,
  thumbnailUrl: (id: string) => `/api/clips/${enc(id)}/thumbnail`,
  subtitlesUrl: (id: string) => `/api/clips/${enc(id)}/subtitles.vtt`,
  lineAudioUrl: (id: string, line: number) => `/api/clips/${enc(id)}/lines/${line}/audio`,
  voiceUrl: (id: string, line: number) => `/api/clips/${enc(id)}/lines/${line}/voice`,
  upload: (video: File, target: Lang, source: Lang | null, title: string) => {
    const form = new FormData();
    form.append("video", video, video.name);
    form.append("target", target);
    if (source) form.append("source", source);
    if (title.trim()) form.append("title", title.trim());
    form.append("consent", "true");
    return fetch("/api/uploads", { method: "POST", body: form }).then((r) => json<{ id: string }>(r));
  },
  uploadStatus: (id: string) => fetch(`/api/uploads/${enc(id)}`).then((r) => json<UploadStatus>(r)),
  removeScene: (id: string) => fetch(`/api/clips/${enc(id)}`, { method: "DELETE" }).then((r) => json<{ ok: boolean }>(r)),
  /** "Was this score fair?" (only the numbers are sent; never audio). */
  feedback: (clip: string, line: number, r: ScoreResult, fair: boolean) => {
    const form = new FormData();
    form.append("clip", clip);
    form.append("line", String(line));
    form.append("score", String(r.score));
    form.append("fair", String(fair));
    form.append("parts", JSON.stringify(r.components ?? {}));
    return fetch("/api/feedback", { method: "POST", body: form }).then((res) => json<{ ok: boolean }>(res));
  },
  calibration: (lang: Lang) => fetch(`/api/calibration/${lang}`).then((r) => json<{ text: string }>(r)),

  calibrate: (audio: Blob) => {
    const form = new FormData();
    form.append("audio", audio, fileName(audio, "calibration"));
    return fetch("/api/calibrate", { method: "POST", body: form }).then((r) => json<{ baseline: Baseline }>(r));
  },

  /** `offset`: the scene time at which the recording started, for the timing check. */
  score: (audio: Blob, clip: string, line: number, baseline: Baseline | null, offset?: number) => {
    const form = new FormData();
    form.append("audio", audio, fileName(audio, "take"));
    form.append("clip", clip);
    form.append("line", String(line));
    if (baseline) form.append("baseline", JSON.stringify(baseline));
    if (offset !== undefined) form.append("offset", offset.toFixed(3));
    return fetch("/api/score", { method: "POST", body: form }).then((r) => json<ScoreResult>(r));
  },

  /** The scene with the player's takes mixed in; resolves to a playable object URL. */
  premiere: async (clip: string, takes: { line: number; blob: Blob; start: number }[]) => {
    const form = new FormData();
    form.append("clip", clip);
    form.append("starts", JSON.stringify(Object.fromEntries(takes.map((t) => [t.line, t.start]))));
    for (const t of takes) form.append("takes", t.blob, fileName(t.blob, String(t.line)));
    const res = await fetch("/api/premiere", { method: "POST", body: form });
    if (!res.ok) await json(res);
    return URL.createObjectURL(await res.blob());
  },
};

function fileName(blob: Blob, base: string) {
  return `${base}.${blob.type.includes("mp4") ? "mp4" : blob.type.includes("ogg") ? "ogg" : blob.type.includes("wav") ? "wav" : "webm"}`;
}

export const LANG_NAME: Record<Lang, string> = { hi: "Hindi", ta: "Tamil", bn: "Bengali", mr: "Marathi", te: "Telugu" };
export const LANG_NATIVE: Record<Lang, string> = { hi: "हिन्दी", ta: "தமிழ்", bn: "বাংলা", mr: "मराठी", te: "తెలుగు" };
export const LANG_GLYPH: Record<Lang, string> = { hi: "अ", ta: "அ", bn: "অ", mr: "अ", te: "అ" };
