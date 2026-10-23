import { Clock3, Volume2 } from "lucide-react";

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

/** The director's note for a line: how to say it, how loud, how long, in plain words. */
export function EmotionBadge({ emotion, intensity, tone, duration }: {
  emotion: string | null; intensity: "low" | "medium" | "high" | null; tone?: string | null; duration?: number;
}) {
  const e = emotionStyle(emotion);
  const level = intensity ? LEVEL[intensity] : null;
  // a strong feeling said at full force reads as shouting, not just loud
  const volume = level === 3 && (emotion === "angry" || emotion === "surprised") ? 4 : level;
  const secs = duration === undefined ? null : Math.max(1, Math.round(duration));
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4">
        <span className="grid h-14 w-14 shrink-0 place-items-center rounded-full text-3xl" style={{ background: `${e.color}22`, boxShadow: `inset 0 0 0 2px ${e.color}` }} aria-hidden>
          {e.face}
        </span>
        <div className="leading-tight">
          <p className="font-marquee text-2xl">Say it <span style={{ color: e.color }}>{e.how}</span></p>
          {tone && <p className="mt-1 text-dim first-letter:uppercase">{tone}</p>}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-8 gap-y-2 text-paper/90">
        {volume !== null && (
          <span className="flex items-center gap-2.5" aria-label={`volume: ${VOLUME[volume]}`}>
            <Volume2 className="h-5 w-5 text-dim" />
            <span className="flex items-end gap-[3px]" aria-hidden>
              {VOLUME.map((_, i) => (
                <span key={i} className="w-1.5 rounded-sm" style={{ height: 6 + i * 4, background: i <= volume ? e.color : "#2e2a38" }} />
              ))}
            </span>
            <span className="font-medium">{VOLUME[volume]}</span>
          </span>
        )}
        {secs !== null && (
          <span className="flex items-center gap-2.5">
            <Clock3 className="h-5 w-5 text-dim" />
            <span className="font-medium">About {secs} second{secs > 1 ? "s" : ""}</span>
          </span>
        )}
      </div>
    </div>
  );
}
