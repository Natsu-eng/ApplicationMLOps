import { useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import { Minus, TrendingDown, TrendingUp } from "lucide-react";
import { ColorIconBadge, accentValueTextClass, type AccentColor } from "../ui/ColorIconBadge";

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
interface StatTileSplitPart {
  label: string;
  value: number | undefined;
  /** Couleur d'identité (ex. couleur de pilier, `pillarColor()`) — teinte le
   * chiffre de CETTE colonne uniquement (Lot 2A correctif 3 : "supervisé,
   * non supervisé et Vision gardent la couleur de leur pilier"). Optionnel :
   * une colonne sans identité propre (aucun cas actuel) resterait neutre. */
  color?: AccentColor;
}

/** Tendance vs période précédente (Lot 2A, AUDIT_DATALAB_2026-08-16.md
 * §J.4) — jamais calculée par le composant : la valeur vient du serveur
 * (aucun historique client-side ici), le composant ne fait que l'afficher. */
export interface StatTileTrend {
  direction: "up" | "down" | "flat";
  /** Déjà formaté par l'appelant (ex. "+12 %", "−3") — le composant ne
   * devine jamais si une hausse est "bonne" ou "mauvaise" (dépend de la
   * métrique), donc pas de couleur sémantique automatique ici. */
  label: string;
}

const TREND_ICON = { up: TrendingUp, down: TrendingDown, flat: Minus } as const;

export function StatTile({
  icon,
  label,
  value,
  color,
  delayMs = 0,
  split,
  trend,
  loading = false,
  href,
}: {
  icon: LucideIcon;
  label: string;
  value: number | undefined;
  color: AccentColor;
  delayMs?: number;
  /** Scinde la valeur en sous-compteurs côte à côte (ex. supervisé / non
   * supervisé / vision) dans la MÊME carte — retour utilisateur direct : un
   * total fusionné est moins parlant qu'un total ventilé par pilier. Le
   * chrome de la tuile (bord, ombre, barre d'accent, icône) reste
   * identique ; `label`/`value` deviennent alors purement le fallback tant
   * que `split` n'est pas fourni. 3 parts fixes (pas un tableau générique) :
   * les hooks `useCountUp` ci-dessous doivent rester appelés un nombre fixe
   * de fois à chaque rendu (règle des hooks), pas dans une boucle de
   * longueur variable — étendu de 2 à 3 au Lot 16 (pilier Vision). Le trend
   * ne s'affiche que sur la tuile simple (non scindée) — combiner les deux
   * dans le même espace restreint serait illisible. */
  split?: [StatTileSplitPart, StatTileSplitPart, StatTileSplitPart];
  trend?: StatTileTrend;
  /** État de chargement (Lot 2A) — squelette plutôt que "—", distinct d'une
   * valeur réellement absente. */
  loading?: boolean;
  /** Lien vers la vue détaillée (Lot 2A) — la tuile entière devient
   * cliquable/focusable au clavier, jamais une zone de clic invisible sans
   * affordance. */
  href?: string;
}) {
  const displayValue = useCountUp(value);
  const displaySplitA = useCountUp(split?.[0]?.value);
  const displaySplitB = useCountUp(split?.[1]?.value);
  const displaySplitC = useCountUp(split?.[2]?.value);

  // Composition refaite (retour utilisateur, Lot 2A) — l'ancienne mise en
  // page (icône + bloc chiffre/libellé côte à côte sur UNE ligne, plus une
  // barre colorée en haut de tuile) cassait sous contrainte réelle : le
  // libellé tronquait ("Modèles entr…"), le delta passait à la ligne à
  // côté du chiffre faute de place. Refaite en 3 lignes empilées, chacune
  // sur toute la largeur de la tuile — plus aucune compétition d'espace
  // entre icône/chiffre/delta/libellé : (1) icône seule, (2) chiffre
  // dominant + delta aligné à côté, (3) libellé sur sa propre ligne
  // complète, jamais tronqué. Barre colorée supprimée (motif daté) :
  // l'identité de couleur reste portée par l'icône seule.
  const content = loading ? (
    <div className="p-4 space-y-2.5">
      <div className="h-8 w-8 rounded-control bg-muted animate-pulse" />
      <div className="h-7 w-20 rounded bg-muted animate-pulse" />
      <div className="h-3.5 w-28 rounded bg-muted animate-pulse" />
    </div>
  ) : split ? (
    <div className="p-4 space-y-3">
      <ColorIconBadge icon={icon} color={color} size="sm" />
      <div className="flex items-stretch divide-x divide-border">
        <div className="pr-3 min-w-0">
          <p className={`text-title tabular-nums leading-none ${split[0].color ? accentValueTextClass(split[0].color) : "text-foreground"}`}>
            {displaySplitA ?? "—"}
          </p>
          <p className="text-caption text-muted-foreground mt-1.5">{split[0].label}</p>
        </div>
        <div className="px-3 min-w-0">
          <p className={`text-title tabular-nums leading-none ${split[1].color ? accentValueTextClass(split[1].color) : "text-foreground"}`}>
            {displaySplitB ?? "—"}
          </p>
          <p className="text-caption text-muted-foreground mt-1.5">{split[1].label}</p>
        </div>
        <div className="pl-3 min-w-0">
          <p className={`text-title tabular-nums leading-none ${split[2].color ? accentValueTextClass(split[2].color) : "text-foreground"}`}>
            {displaySplitC ?? "—"}
          </p>
          <p className="text-caption text-muted-foreground mt-1.5">{split[2].label}</p>
        </div>
      </div>
    </div>
  ) : (
    <div className="p-4 space-y-2">
      <ColorIconBadge icon={icon} color={color} size="sm" />
      <div className="flex items-baseline gap-2">
        <p className="text-display text-foreground tabular-nums leading-none">{displayValue ?? "—"}</p>
        {trend && (
          <span
            className={`inline-flex items-center gap-0.5 text-caption tabular-nums flex-shrink-0 ${
              trend.direction === "flat" ? "text-muted-foreground" : "text-foreground/70"
            }`}
          >
            {(() => {
              const TrendIcon = TREND_ICON[trend.direction];
              return <TrendIcon size={12} aria-hidden="true" />;
            })()}
            {trend.label}
          </span>
        )}
      </div>
      <p className="text-caption text-muted-foreground">{label}</p>
    </div>
  );

  const chromeClass =
    "group stat-tile-enter overflow-hidden rounded-card border border-border/70 bg-card shadow-card transition-all duration-200 hover:shadow-overlay hover:-translate-y-0.5";

  if (href) {
    return (
      <Link
        to={href}
        className={`${chromeClass} block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/50`}
        style={{ animationDelay: `${delayMs}ms` }}
      >
        {content}
      </Link>
    );
  }

  return (
    <div className={chromeClass} style={{ animationDelay: `${delayMs}ms` }}>
      {content}
    </div>
  );
}

/** Conteneur de la rangée de tuiles — porte les keyframes d'entrée (une
 * seule injection pour toute la rangée) et l'échelonnement visuel via
 * `--stat-tile-index` sur chaque enfant. `wide` : la 2ᵉ tuile (celle qui
 * porte un `split`, voir Dashboard.tsx) reçoit plus de largeur, les 3 autres
 * (une seule valeur, pas besoin de place) se resserrent en retour — retour
 * utilisateur direct : les libellés "Supervisé"/"Non supervisé" tronquaient
 * dans une grille à 4 colonnes strictement égales. Part élargie à nouveau
 * (Lot 16) : le split est passé de 2 à 3 sous-compteurs (pilier Vision
 * ajouté). */
export function StatTileRow({ children, wide = false }: { children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={`grid gap-4 sm:grid-cols-2 mb-8 ${wide ? "lg:grid-cols-[0.9fr_1.35fr_0.75fr_0.9fr]" : "lg:grid-cols-4"}`}>
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
