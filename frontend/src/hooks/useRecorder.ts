import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderState = "idle" | "starting" | "recording" | "denied";

/** One live pitch reading: seconds since recording started, semitones vs the take's median. */
export interface PitchPoint {
  t: number;
  st: number | null;
  loud: number;
}

const PITCH_EVERY_S = 0.05; // ~20 readings per second

/** Fundamental frequency of one buffer with the YIN method, or null when there is no clear voice. */
export function detectPitch(buf: Float32Array, sampleRate: number, minHz = 70, maxHz = 600): number | null {
  let energy = 0;
  for (const v of buf) energy += v * v;
  if (Math.sqrt(energy / buf.length) < 0.01) return null;
  const minLag = Math.floor(sampleRate / maxHz);
  const maxLag = Math.min(Math.floor(sampleRate / minHz), Math.floor(buf.length / 2));
  const d = new Float32Array(maxLag + 1);
  for (let lag = 1; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < buf.length - maxLag; i++) {
      const diff = buf[i] - buf[i + lag];
      sum += diff * diff;
    }
    d[lag] = sum;
  }
  // cumulative mean normalized difference; the first dip under the threshold is the period
  let running = 0;
  for (let lag = 1; lag <= maxLag; lag++) {
    running += d[lag];
    d[lag] = running ? (d[lag] * lag) / running : 1;
  }
  for (let lag = minLag; lag < maxLag; lag++) {
    if (d[lag] < 0.15) {
      while (lag + 1 < maxLag && d[lag + 1] < d[lag]) lag++;
      // parabolic interpolation around the minimum
      const a = d[lag - 1], b = d[lag], c = d[lag + 1];
      const shift = (a - c) / (2 * (a - 2 * b + c) || 1);
      return sampleRate / (lag + (Number.isFinite(shift) ? shift : 0));
    }
  }
  return null;
}

/** Microphone recording with a live input level (0..1) and a live pitch trace. */
export function useRecorder(maxSeconds = 15) {
  const [state, setState] = useState<RecorderState>("idle");
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [trace, setTrace] = useState<PitchPoint[]>([]);
  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const ctx = useRef<AudioContext | null>(null);
  const raf = useRef(0);
  const started = useRef(0);
  const lastPitch = useRef(0);
  const hz = useRef<number[]>([]);
  const resolveStop = useRef<((b: Blob) => void) | null>(null);

  const cleanup = useCallback(() => {
    cancelAnimationFrame(raf.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    ctx.current?.close().catch(() => {});
    ctx.current = null;
    setLevel(0);
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const stop = useCallback((): Promise<Blob> => {
    return new Promise((resolve) => {
      const r = recorder.current;
      if (!r || r.state === "inactive") return resolve(new Blob());
      resolveStop.current = resolve;
      r.stop();
    });
  }, []);

  /** Ask for the microphone without recording (to show the permission prompt at a calm moment). */
  const requestPermission = useCallback(async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      s.getTracks().forEach((t) => t.stop());
      return true;
    } catch {
      setState("denied");
      return false;
    }
  }, []);

  const start = useCallback(async () => {
    setState("starting");
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false },
      });
    } catch {
      setState("denied");
      return false;
    }
    const mime = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg"].find((m) => MediaRecorder.isTypeSupported(m));
    const r = new MediaRecorder(stream.current, mime ? { mimeType: mime } : undefined);
    chunks.current = [];
    r.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    r.onstop = () => {
      const blob = new Blob(chunks.current, { type: r.mimeType });
      cleanup();
      setState("idle");
      resolveStop.current?.(blob);
      resolveStop.current = null;
    };
    recorder.current = r;

    ctx.current = new AudioContext();
    const analyser = ctx.current.createAnalyser();
    analyser.fftSize = 2048;
    ctx.current.createMediaStreamSource(stream.current).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const sampleRate = ctx.current.sampleRate;
    hz.current = [];
    lastPitch.current = 0;
    setTrace([]);
    started.current = performance.now();
    const tick = () => {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += v * v;
      const rms = Math.sqrt(sum / buf.length);
      const lvl = Math.min(1, rms * 6);
      setLevel((prev) => prev * 0.6 + lvl * 0.4);
      const t = (performance.now() - started.current) / 1000;
      setElapsed(t);
      if (t - lastPitch.current >= PITCH_EVERY_S) {
        lastPitch.current = t;
        const f = detectPitch(buf, sampleRate);
        if (f) hz.current.push(f);
        // semitones relative to this take's own median, like the server's melody line
        const sorted = [...hz.current].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        const st = f && median ? 12 * Math.log2(f / median) : null;
        setTrace((prev) => [...prev, { t, st: st === null ? null : Math.max(-12, Math.min(12, st)), loud: lvl }]);
      }
      if (t >= maxSeconds) r.state !== "inactive" && r.stop();
      else raf.current = requestAnimationFrame(tick);
    };
    r.start();
    setState("recording");
    setElapsed(0);
    raf.current = requestAnimationFrame(tick);
    return true;
  }, [cleanup, maxSeconds]);

  return { state, level, elapsed, trace, start, stop, requestPermission };
}
