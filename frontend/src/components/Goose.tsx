import { motion } from "framer-motion";

/** The goose. `honking` opens the beak and bobs the head. */
export function Goose({ size = 120, honking = false, className = "" }: { size?: number; honking?: boolean; className?: string }) {
  return (
    <motion.svg
      viewBox="0 0 120 120"
      width={size}
      height={size}
      className={className}
      aria-hidden
      animate={honking ? { rotate: [0, -6, 4, -6, 0], y: [0, -4, 0, -3, 0] } : { rotate: 0, y: 0 }}
      transition={{ duration: 0.7, repeat: honking ? Infinity : 0, repeatDelay: 0.6 }}
    >
      {/* body */}
      <ellipse cx="62" cy="86" rx="38" ry="24" fill="#f4eee3" />
      <path d="M30 84 Q18 72 26 64 Q34 76 44 78 Z" fill="#e4dccd" />
      {/* neck + head */}
      <path d="M70 72 Q66 46 74 30 Q80 18 92 20 Q102 23 100 34 Q98 42 90 46 Q84 52 86 72 Z" fill="#f4eee3" />
      <circle cx="91" cy="30" r="3.2" fill="#13111a" />
      <circle cx="92" cy="29" r="1" fill="#f4eee3" />
      {/* beak: upper stays, lower drops when honking */}
      <path d="M99 30 L116 33 L100 36 Z" fill="#ff7b54" />
      <motion.path
        d="M100 36 L114 36 L99 39 Z"
        fill="#e0623d"
        style={{ originX: "100px", originY: "36px" }}
        animate={honking ? { rotate: [0, 18, 0, 18, 0] } : { rotate: 0 }}
        transition={{ duration: 0.7, repeat: honking ? Infinity : 0, repeatDelay: 0.6 }}
      />
      {/* feet */}
      <path d="M52 108 l-6 6 h12 Z M70 108 l-6 6 h12 Z" fill="#ff7b54" />
    </motion.svg>
  );
}
