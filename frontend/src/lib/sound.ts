// Tiny synthesized sound effects (no audio files): a clapper, a chime for a Dub, a honk for a Goose.

let ctx: AudioContext | null = null;
let muted = localStorage.getItem("ddg.muted") === "1";

export const isMuted = () => muted;
export function setMuted(m: boolean) {
  muted = m;
  localStorage.setItem("ddg.muted", m ? "1" : "0");
}

function audio() {
  ctx ??= new AudioContext();
  return ctx;
}

function tone(freq: number, start: number, dur: number, type: OscillatorType, gain = 0.18, slideTo?: number) {
  const a = audio();
  const osc = a.createOscillator();
  const g = a.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, a.currentTime + start);
  if (slideTo) osc.frequency.exponentialRampToValueAtTime(slideTo, a.currentTime + start + dur);
  g.gain.setValueAtTime(0, a.currentTime + start);
  g.gain.linearRampToValueAtTime(gain, a.currentTime + start + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + start + dur);
  osc.connect(g).connect(a.destination);
  osc.start(a.currentTime + start);
  osc.stop(a.currentTime + start + dur + 0.02);
}

export const sfx = {
  clap() {
    if (muted) return;
    const a = audio();
    const len = Math.floor(a.sampleRate * 0.06);
    const buf = a.createBuffer(1, len, a.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len) ** 3;
    const src = a.createBufferSource();
    src.buffer = buf;
    const g = a.createGain();
    g.gain.value = 0.5;
    src.connect(g).connect(a.destination);
    src.start();
  },
  tick() {
    if (!muted) tone(880, 0, 0.06, "square", 0.05);
  },
  dub() {
    if (muted) return;
    [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tone(f, i * 0.09, 0.5, "triangle", 0.16));
  },
  goose() {
    if (muted) return;
    tone(420, 0, 0.16, "sawtooth", 0.12, 300);
    tone(440, 0.22, 0.24, "sawtooth", 0.12, 260);
  },
};
