import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderState = "idle" | "starting" | "recording" | "denied";

/** Microphone recording with a live input level (0..1). */
export function useRecorder(maxSeconds = 15) {
  const [state, setState] = useState<RecorderState>("idle");
  const [level, setLevel] = useState(0);
  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const ctx = useRef<AudioContext | null>(null);
  const raf = useRef(0);
  const started = useRef(0);
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
    analyser.fftSize = 1024;
    ctx.current.createMediaStreamSource(stream.current).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    started.current = performance.now();
    const tick = () => {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += v * v;
      const rms = Math.sqrt(sum / buf.length);
      const lvl = Math.min(1, rms * 6);
      setLevel((prev) => prev * 0.6 + lvl * 0.4);
      const t = (performance.now() - started.current) / 1000;
      if (t >= maxSeconds) r.state !== "inactive" && r.stop();
      else raf.current = requestAnimationFrame(tick);
    };
    r.start();
    setState("recording");
    raf.current = requestAnimationFrame(tick);
    return true;
  }, [cleanup, maxSeconds]);

  return { state, level, start, stop, requestPermission };
}
