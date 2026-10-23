import * as ToggleGroup from "@radix-ui/react-toggle-group";
import clsx from "clsx";
import { Star } from "lucide-react";

const clean = (w: string) => w.replace(/[^\p{L}\p{M}\p{N}]/gu, "");

/** The line to dub, in the target script or romanized, with the stressed word lit up. */
export function ScriptLine({ text, roman, stress, showRoman, center = false }: {
  text: string; roman: string; stress: string[]; showRoman: boolean; center?: boolean;
}) {
  const words = text.split(/\s+/).filter(Boolean);
  const romanWords = roman.split(/\s+/).filter(Boolean);
  const stressed = new Set(
    words.map((w, i) => (stress.some((s) => clean(s) && clean(w).includes(clean(s))) ? i : -1)).filter((i) => i >= 0),
  );
  // romanized words line up with the script's words only when the counts match
  const shown = showRoman && romanWords.length ? romanWords : words;
  const canMark = !showRoman || romanWords.length === words.length;
  return (
    <p
      lang={showRoman ? "en" : undefined}
      className={clsx("leading-snug text-paper", center && "text-center", showRoman ? "font-body font-medium" : "font-script font-bold",
        // long lines get smaller type so the whole line fits without pushing the buttons off screen
        (showRoman ? roman : text).length > 60 ? "text-xl sm:text-2xl" : (showRoman ? roman : text).length > 30 ? "text-2xl sm:text-3xl" : "text-3xl sm:text-4xl")}
    >
      {shown.map((w, i) => (
        <span key={i} className={clsx(canMark && stressed.has(i) && "relative text-gold")}>
          {w}
          {i < shown.length - 1 && " "}
        </span>
      ))}
    </p>
  );
}

/** Script / English-letters switch, styled as a two-position studio switch. */
export function ScriptToggle({ glyph, roman, onChange }: { glyph: string; roman: boolean; onChange: (r: boolean) => void }) {
  return (
    <ToggleGroup.Root
      type="single"
      value={roman ? "roman" : "script"}
      onValueChange={(v) => v && onChange(v === "roman")}
      aria-label="Show the line in"
      className="inline-flex rounded-md border border-rule bg-ink-2 p-0.5"
    >
      {[["script", glyph], ["roman", "Aa"]].map(([value, label]) => (
        <ToggleGroup.Item
          key={value}
          value={value}
          aria-label={value === "script" ? "original script" : "English letters"}
          className="min-w-10 rounded px-2.5 py-1 font-script text-sm text-dim transition-colors data-[state=on]:bg-paper data-[state=on]:text-ink"
        >
          {label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

export function Stars({ n, size = 14 }: { n: number; size?: number }) {
  return (
    <span className="inline-flex gap-0.5" aria-label={`${n} of 3 stars`}>
      {[0, 1, 2].map((i) => (
        <Star key={i} style={{ width: size, height: size }} className={i < n ? "fill-gold text-gold" : "text-rule"} />
      ))}
    </span>
  );
}
