import { Goose } from "./Goose";

export function Logo({ big = false }: { big?: boolean }) {
  if (!big) {
    return (
      <span className="flex items-center gap-2 whitespace-nowrap font-marquee text-sm tracking-wide">
        <span className="h-2.5 w-2.5 rounded-full bg-onair shadow-[0_0_10px_#ff3d55]" />
        <span className="hidden sm:inline">DUB DUB <span className="text-goose">GOOSE</span></span>
      </span>
    );
  }
  return (
    <div className="relative mt-14 select-none text-center sm:mt-0">
      <h1 className="font-marquee leading-[0.85] tracking-tight text-[clamp(3.2rem,11vw,8rem)]">
        <span className="block text-paper">DUB DUB</span>
        <span className="block text-goose drop-shadow-[0_6px_0_rgba(0,0,0,0.35)]">GOOSE</span>
      </h1>
      <div className="absolute -top-16 right-1/2 translate-x-1/2 scale-75 sm:-right-14 sm:-top-10 sm:translate-x-0 sm:scale-100">
        <Goose size={96} className="rotate-12" />
      </div>
    </div>
  );
}
