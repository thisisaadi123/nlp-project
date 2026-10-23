/** Night-time dubbing studio: warm stage light on dark ink, a red ON AIR lamp, gold for a Dub. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#13111a", 2: "#1b1823", 3: "#26222f" },
        rule: "#36313f",
        paper: "#f4eee3",
        dim: "#a39cad",
        gold: { DEFAULT: "#ffc24b", deep: "#e09a1a" },
        onair: "#ff3d55",
        goose: "#ff7b54",
        mint: "#5be3b0",
      },
      fontFamily: {
        marquee: ['"Bungee"', "Impact", "sans-serif"],
        body: ['"Figtree"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
        script: ['"Noto Sans Devanagari"', '"Noto Sans Tamil"', '"Noto Sans Bengali"', '"Noto Sans Telugu"', "sans-serif"],
      },
      keyframes: {
        flicker: { "0%,100%": { opacity: "1" }, "45%": { opacity: ".55" }, "50%": { opacity: ".9" } },
      },
      animation: { flicker: "flicker 1.4s ease-in-out infinite" },
    },
  },
  plugins: [],
};
