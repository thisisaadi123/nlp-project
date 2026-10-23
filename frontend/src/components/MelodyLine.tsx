import { useId } from "react";
import type { PitchPoint } from "../hooks/useRecorder";
import type { Shape } from "../lib/api";

const W = 640;
const H = 140;
const PAD_X = 16;
const AXIS = 44; // room for the HIGH / LOW labels
const TOP = 18;
const BOTTOM = 128;

interface Props {
  target: Shape;
  /** listen: fraction of the line played so far (0..1), the curve draws up to it */
  progress?: number;
  /** recording: the player's live pitch, time in the target's seconds */
  live?: PitchPoint[];
  /** after scoring: the player's whole take */
  compare?: Shape;
  color?: string;
}

/** The original's pitch over time (thicker = louder, gaps = pauses) and its stressed moment;
 *  optionally the player's pitch drawn over it in mint. */
export function MelodyLine({ target, progress, live, compare, color = "#ffc24b" }: Props) {
  const clip = useId();
  const span = Math.max(target.duration, compare?.duration ?? 0, 0.5);
  const all = [...target.pitch, ...(compare?.pitch ?? []), ...(live?.flatMap((p) => (p.st === null ? [] : [p.st])) ?? [])];
  const range = Math.max(5, ...all.map(Math.abs)) * 1.1;
  const x = (t: number) => AXIS + (Math.max(0, Math.min(t, span)) / span) * (W - AXIS - PAD_X);
  const y = (st: number) => TOP + ((range - st) / (2 * range)) * (BOTTOM - TOP);

  const stressWord = target.words.find((w) => w.stress)?.w;
  const progressX = progress === undefined ? W : x(progress * target.duration);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" role="img"
      aria-label={`Melody of the line, ${target.duration.toFixed(1)} seconds${stressWord ? `, stress on ${stressWord}` : ""}`}>
      <defs>
        <clipPath id={clip}><rect x="0" y="0" width={progressX} height={H} /></clipPath>
        <filter id={`${clip}-glow`}><feGaussianBlur stdDeviation="3" /></filter>
      </defs>
      {/* the voice's middle */}
      <line x1={AXIS} x2={W - PAD_X} y1={y(0)} y2={y(0)} stroke="#36313f" strokeDasharray="3 6" />
      <text x={0} y={TOP + 8} fontSize={11} letterSpacing={1.5} fill="#a39cad" className="font-mono">HIGH</text>
      <text x={0} y={BOTTOM} fontSize={11} letterSpacing={1.5} fill="#a39cad" className="font-mono">LOW</text>

      {/* faint full curve, then the played part in colour */}
      <Curve shape={target} x={x} y={y} color={color} opacity={progress === undefined ? 1 : 0.22} />
      {progress !== undefined && <g clipPath={`url(#${clip})`}><Curve shape={target} x={x} y={y} color={color} opacity={1} /></g>}

      {compare && <Curve shape={compare} x={x} y={y} color="#5be3b0" opacity={0.95} thin />}

      {live && live.length > 1 && (
        <polyline
          fill="none" stroke="#5be3b0" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round"
          points={live.filter((p) => p.st !== null && p.t >= 0).map((p) => `${x(p.t)},${y(p.st as number)}`).join(" ")}
        />
      )}

      {/* the stressed moment */}
      <circle cx={x(target.stress_t)} cy={y(pitchAt(target, target.stress_t))} r={9} fill={color} opacity={0.35} filter={`url(#${clip}-glow)`} />
      <circle cx={x(target.stress_t)} cy={y(pitchAt(target, target.stress_t))} r={5} fill={color} />

      {progress !== undefined && progress > 0 && progress < 1 && (
        <line x1={progressX} x2={progressX} y1={TOP - 6} y2={BOTTOM + 6} stroke="#f4eee3" strokeWidth={1.5} opacity={0.7} />
      )}
    </svg>
  );
}

/** The melody in plain words, e.g. "Your voice stays level, then drops at the end." */
export function describeMelody(s: Shape) {
  const v = s.pitch.filter((_, i) => s.voiced[i]);
  if (v.length < 4) return "Short and punchy.";
  const q = Math.max(1, Math.floor(v.length / 4));
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  const change = mean(v.slice(-q)) - mean(v.slice(0, q));
  const spread = Math.max(...v) - Math.min(...v);
  if (spread < 3) return "Your voice stays level, almost flat.";
  if (change > 2) return "Your voice goes up at the end, like a question or a call.";
  if (change < -2) return "Your voice starts high and drops at the end.";
  if (spread > 7) return "Your voice jumps up and down a lot.";
  return "Your voice rises in the middle, then settles.";
}

function pitchAt(s: Shape, t: number) {
  const i = Math.min(s.pitch.length - 1, Math.max(0, Math.round(t / s.step)));
  return s.pitch[i] ?? 0;
}

/** The curve as short segments: thickness follows loudness, unvoiced stretches are left out. */
function Curve({ shape, x, y, color, opacity, thin = false }: {
  shape: Shape; x: (t: number) => number; y: (st: number) => number; color: string; opacity: number; thin?: boolean;
}) {
  const segs = [];
  for (let i = 1; i < shape.pitch.length; i++) {
    if (!shape.voiced[i] || !shape.voiced[i - 1]) continue;
    const w = thin ? 3 : 2 + shape.loud[i] * 9;
    segs.push(
      <line key={i} x1={x((i - 1) * shape.step)} y1={y(shape.pitch[i - 1])} x2={x(i * shape.step)} y2={y(shape.pitch[i])}
        stroke={color} strokeWidth={w} strokeLinecap="round" opacity={opacity} />,
    );
  }
  return <g>{segs}</g>;
}
