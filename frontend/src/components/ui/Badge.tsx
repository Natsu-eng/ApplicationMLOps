import type { ReactNode } from "react";

type Variant = "neutral" | "accent" | "success" | "warning" | "danger";

const VARIANT_CLASSES: Record<Variant, string> = {
  neutral: "bg-slate-100 text-slate-600 border border-slate-200",
  accent: "bg-teal-50 text-teal-700 border border-teal-200",
  success: "bg-emerald-50 text-emerald-700 border border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border border-amber-200",
  danger: "bg-rose-50 text-rose-700 border border-rose-200",
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
