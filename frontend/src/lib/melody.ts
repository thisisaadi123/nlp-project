import type { Shape } from "./api";

export type Movement = "flat" | "rise" | "fall" | "wave" | "arch" | "short";

/** How the original's voice moves over the line, in a word and a short sentence. */
export function describeMelody(s: Shape): { movement: Movement; text: string } {
  const v = s.pitch.filter((_, i) => s.voiced[i]);
  if (v.length < 4) return { movement: "short", text: "Short and punchy." };
  const q = Math.max(1, Math.floor(v.length / 4));
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  const change = mean(v.slice(-q)) - mean(v.slice(0, q));
  const spread = Math.max(...v) - Math.min(...v);
  if (spread < 3) return { movement: "flat", text: "Keep your voice level, almost flat." };
  if (change > 2) return { movement: "rise", text: "Let your voice go up at the end, like a question or a call." };
  if (change < -2) return { movement: "fall", text: "Start high and let your voice drop at the end." };
  if (spread > 7) return { movement: "wave", text: "Lots of ups and downs: say it with feeling." };
  return { movement: "arch", text: "Rise in the middle, then settle." };
}
