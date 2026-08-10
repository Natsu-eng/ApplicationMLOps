import { useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { ColorIconBadge, accentSurfaceClass, type AccentColor } from "../ui/ColorIconBadge";

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

/** Anime un compteur de 0 (ou de son ancienne valeur) jusqu'à `value` — animation
 * désactivée si l'utilisateur a demandé moins de mouvement au niveau système
 * (même garde que `.hero-term` dans index.css, ici en JS car la valeur cible
 * est dynamique). `undefined` tant que la donnée n'est pas encore chargée :
 * le tiret d'attente reste géré par l'appelant. */
function useCountUp(value: number | undefined, durationMs = 800): number | undefined {
  const [display, setDisplay] = useState(value);
  const previousValue = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (value === undefined) return;
    const from = previousValue.current ?? 0;
    previousValue.current = value;

    if (from === value || window.matchMedia(REDUCED_MOTION_QUERY).matches) {
      setDisplay(value);
      return;
    }

    const start = performance.now();
    let frame: number;
    const tick = (now: number) => {
      const progress = Math.min((now - start) / durationMs, 1);
      const eased = 1 - (1 - progress) ** 3; // ease-out cubic
      setDisplay(Math.round(from + (value - from) * eased));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, durationMs]);

  return display;
}

/** Tuile de statistique du dashboard (Lot E1-ter) — icône colorée
 * (`ColorIconBadge`, palette partagée avec le reste de l'app), compteur
 * animé à l'entrée, légère élévation au survol. Une seule fois par page :
 * les keyframes d'entrée sont injectées par `StatTileRow`, pas ici, pour ne
 * pas dupliquer la balise `<style>` par tuile. */
export function StatTile({
  icon,
  label,
  value,
  color,
  delayMs = 0,
}: {
  icon: LucideIcon;
  label: string;
  value: number | undefined;
  color: AccentColor;
  delayMs?: number;
}) {
  const displayValue = useCountUp(value);

  return (
    <div
      className={`group stat-tile-enter rounded-2xl border shadow-sm p-4 flex items-center gap-3 transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 ${accentSurfaceClass(color)}`}
      style={{ animationDelay: `${delayMs}ms` }}
    >
      <ColorIconBadge icon={icon} color={color} />
      <div className="min-w-0">
        <p className="text-xl font-semibold text-slate-900 tabular-nums">{displayValue ?? "—"}</p>
        <p className="text-xs text-slate-600 truncate">{label}</p>
      </div>
    </div>
  );
}

/** Conteneur de la rangée de tuiles — porte les keyframes d'entrée (une
 * seule injection pour toute la rangée) et l'échelonnement visuel via
 * `--stat-tile-index` sur chaque enfant. */
export function StatTileRow({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-8">
      <style>{`
        @keyframes stat-tile-fade-in {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .stat-tile-enter { animation: stat-tile-fade-in 0.5s ease-out backwards; }
        @media (prefers-reduced-motion: reduce) {
          .stat-tile-enter { animation: none; }
        }
      `}</style>
      {children}
    </div>
  );
}
