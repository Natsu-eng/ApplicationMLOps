import type { ReactNode } from "react";

type Variant = "neutral" | "accent" | "success" | "warning" | "danger";

const VARIANT_CLASSES: Record<Variant, string> = {
  neutral: "bg-slate-800 text-slate-400 border border-slate-700",
  accent: "bg-teal-500/15 text-teal-300 border border-teal-500/30",
  success: "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30",
  warning: "bg-amber-500/15 text-amber-300 border border-amber-500/30",
  danger: "bg-rose-500/15 text-rose-300 border border-rose-500/30",
};

export function Badge({ variant = "neutral", children }: { variant?: Variant; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center whitespace-nowrap text-[11px] font-medium px-2 py-0.5 rounded-full ${VARIANT_CLASSES[variant]}`}
    >
      {children}
    </span>
  );
}
