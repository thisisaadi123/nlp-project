import { Clock3, TrendingDown, TrendingUp, Volume2 } from "lucide-react";

const EMOTIONS: Record<string, { face: string; word: string; how: string; color: string }> = {
  angry: { face: "😠", word: "Angry", how: "angrily", color: "#ff5a5f" },
  sad: { face: "😢", word: "Sad", how: "sadly", color: "#6aa8ff" },
  happy: { face: "😄", word: "Happy", how: "happily", color: "#ffc24b" },
  fearful: { face: "😨", word: "Scared", how: "like you're scared", color: "#4fd1c5" },
  surprised: { face: "😲", word: "Surprised", how: "surprised", color: "#ff9f43" },
  disgust: { face: "🤢", word: "Disgusted", how: "in disgust", color: "#9bd35a" },
  sarcastic: { face: "😏", word: "Sarcastic", how: "sarcastically", color: "#b98cff" },
  comic: { face: "😜", word: "Playful", how: "playfully", color: "#ffd166" },
  calm: { face: "😌", word: "Calm", how: "calmly", color: "#a3b1c2" },
  neutral: { face: "😐", word: "Matter-of-fact", how: "plainly", color: "#a3b1c2" },
};

const VOLUME = ["Whisper", "Soft", "Normal", "Loud", "Shouting"];
const LEVEL: Record<string, number> = { low: 1, medium: 2, high: 3 };

export function emotionStyle(emotion: string | null) {
  return EMOTIONS[emotion ?? ""] ?? {
    face: "🎭", word: emotion ? emotion[0].toUpperCase() + emotion.slice(1) : "Natural", how: "naturally", color: "#a3b1c2",
  };
}

/** The whole note in one row: feeling, volume, length, and the pitch movement when it's clear. */
export function DirectorNote({ emotion, intensity, duration, movement }: {
  emotion: string | null; intensity: "low" | "medium" | "high" | null; duration: number;
  movement?: "rise" | "fall" | null;
}) {
  const e = emotionStyle(emotion);
  const level = intensity ? LEVEL[intensity] : 1;
  const volume = level === 3 && (emotion === "angry" || emotion === "surprised") ? 4 : level;
  const secs = Math.max(1, Math.round(duration));
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-paper/90">
      <span className="flex items-center gap-2">
        <span className="text-2xl" aria-hidden>{e.face}</span>
        <span className="font-marquee text-lg">Say it <span style={{ color: e.color }}>{e.how}</span></span>
      </span>
      <span className="flex items-center gap-2" aria-label={`volume: ${VOLUME[volume]}`}>
        <Volume2 className="h-4 w-4 text-dim" />
        <span className="flex items-end gap-[2px]" aria-hidden>
          {VOLUME.map((_, i) => (
            <span key={i} className="w-1 rounded-sm" style={{ height: 5 + i * 3, background: i <= volume ? e.color : "#2e2a38" }} />
          ))}
        </span>
        <span>{VOLUME[volume]}</span>
      </span>
      <span className="flex items-center gap-1.5"><Clock3 className="h-4 w-4 text-dim" /> ~{secs} s</span>
      {movement && (
        <span className="flex items-center gap-1.5">
          {movement === "rise" ? <TrendingUp className="h-4 w-4 text-dim" /> : <TrendingDown className="h-4 w-4 text-dim" />}
          {movement === "rise" ? "ends going up" : "ends going down"}
        </span>
      )}
    </div>
  );
}
