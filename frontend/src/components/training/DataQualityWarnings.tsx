import { useEffect, useState } from "react";
import { AlertTriangle, ChevronDown, Info, ShieldAlert, ShieldCheck, XCircle } from "lucide-react";
import { ApiError, api, type DataWarning } from "../../api/client";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";

/** Garde-fous de données (Lot B) — avertissements affichés au moment du
 * choix dataset+cible, AVANT le lancement de l'entraînement. Toujours
 * informatif, jamais bloquant : même une fuite critique n'empêche pas de
 * lancer l'entraînement, elle est juste signalée clairement (on propose,
 * on n'impose pas). */

const LEVEL_CONFIG: Record<
  DataWarning["level"],
  { badge: "danger" | "warning" | "accent"; icon: typeof ShieldAlert; border: string; iconColor: string }
> = {
  critique: { badge: "danger", icon: ShieldAlert, border: "border-destructive/20", iconColor: "text-destructive" },
  attention: { badge: "warning", icon: AlertTriangle, border: "border-warning/20", iconColor: "text-warning" },
  info: { badge: "accent", icon: Info, border: "border-primary/20", iconColor: "text-primary" },
};

const LEVEL_LABEL: Record<DataWarning["level"], string> = {
  critique: "Critique",
  attention: "Attention",
  info: "Info",
};

/** Codes de garde-fous (Lot B) pour lesquels la colonne signalée n'a aucune
 * valeur prédictive et peut être exclue en un clic (Lot Nettoyage guidé des
 * variables) — miroir de `_EXCLUSION_WARNING_CODES` côté backend
 * (`services/feature_engineering.py`), sans nouvel appel réseau : on lit
 * directement `warning.columns`, déjà disponible. */
const EXCLUDABLE_CODES = new Set(["colonne_constante", "cardinalite_excessive", "colonnes_dupliquees"]);

/** Colonne(s) qu'un warning donné rend excluables — pour un doublon exact
 * (deux colonnes signalées), seule la SECONDE est proposée à l'exclusion :
 * garder au moins une des deux, contenu strictement identique (même règle
 * que `_suggest_column_exclusion` côté backend). */
function excludableColumnsOf(warning: DataWarning): string[] {
  if (!EXCLUDABLE_CODES.has(warning.code)) return [];
  if (warning.code === "colonnes_dupliquees" && warning.columns.length === 2) return [warning.columns[1]];
  return warning.columns;
}

export function DataQualityWarnings({
  datasetId,
  targetColumn,
  groupColumn,
  selectedFeatures,
  onExcludeColumns,
}: {
  datasetId: number;
  /** Optionnel (Lot Nettoyage guidé des variables) : sans cible, seules les
   * détections structurelles reviennent (constantes, cardinalité, doublons,
   * numérique mal typé, valeurs manquantes, colinéarité) — utilisable dès
   * l'exploration d'un dataset (voir EdaModal.tsx), pas seulement depuis le
   * formulaire d'entraînement. */
  targetColumn?: string;
  groupColumn?: string;
  /** Variables actuellement retenues pour l'entraînement (étape 1 du
   * formulaire) — sert uniquement à savoir si une colonne excluable est déjà
   * exclue, pour afficher l'action "Exclure" ou une confirmation. Omis
   * (contexte EDA, hors formulaire d'entraînement) : les actions d'exclusion
   * ne s'affichent simplement pas, seules les alertes restent visibles. */
  selectedFeatures?: Set<string>;
  /** Retire une ou plusieurs colonnes de la sélection (étape 1) — jamais un
   * simple toggle : approuver une suggestion d'exclusion doit toujours
   * exclure, jamais réintégrer une colonne déjà exclue. */
  onExcludeColumns?: (columns: string[]) => void;
}) {
  const [warnings, setWarnings] = useState<DataWarning[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  useEffect(() => {
    setLoading(true);
    setError(null);
    setExpanded(new Set());
    api.datasets
      .qualityCheck(datasetId, targetColumn || undefined, groupColumn)
      .then((data) => setWarnings(data.warnings))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Analyse des données indisponible"))
      .finally(() => setLoading(false));
  }, [datasetId, targetColumn, groupColumn]);

  function toggle(index: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  const canExclude = Boolean(selectedFeatures && onExcludeColumns);
  const allExcludable = warnings
    ? Array.from(new Set(warnings.flatMap(excludableColumnsOf)))
    : [];
  const stillIncluded = selectedFeatures
    ? allExcludable.filter((col) => selectedFeatures.has(col))
    : [];

  return (
    <div className="space-y-2">
      {loading && <p className="text-xs text-muted-foreground">Analyse des données…</p>}
      {error && <p className="text-xs text-destructive">{error}</p>}

      {!loading && !error && warnings && warnings.length === 0 && (
        <div className="flex items-center gap-2 text-xs text-success bg-success/10 border border-success/20 rounded-lg px-3 py-2">
          <ShieldCheck size={14} className="flex-shrink-0" />
          Aucune alerte détectée sur ce dataset pour cette cible.
        </div>
      )}

      {canExclude && stillIncluded.length > 0 && (
        <div className="flex items-center justify-between gap-3 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2">
          <p className="text-xs text-muted-foreground">
            {stillIncluded.length} variable{stillIncluded.length > 1 ? "s" : ""} sans valeur prédictive
            détectée{stillIncluded.length > 1 ? "s" : ""} (identifiant, constante, doublon).
          </p>
          <Button type="button" variant="secondary" size="sm" onClick={() => onExcludeColumns?.(stillIncluded)}>
            <XCircle size={13} />
            Tout exclure
          </Button>
        </div>
      )}

      {!loading &&
        !error &&
        warnings &&
        warnings.length > 0 &&
        warnings.map((warning, index) => {
          const config = LEVEL_CONFIG[warning.level];
          const Icon = config.icon;
          const isOpen = expanded.has(index);
          const excludable = canExclude ? excludableColumnsOf(warning) : [];
          return (
            <div
              key={`${warning.code}-${warning.columns.join(",")}-${index}`}
              className={`rounded-lg border ${config.border} bg-muted px-3 py-2.5`}
            >
              <button
                type="button"
                onClick={() => toggle(index)}
                className="w-full flex items-start gap-2 text-left"
              >
                <Icon size={15} className={`flex-shrink-0 mt-0.5 ${config.iconColor}`} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant={config.badge}>{LEVEL_LABEL[warning.level]}</Badge>
                    <p className="text-sm text-foreground font-medium">{warning.title}</p>
                  </div>
                </div>
                <ChevronDown
                  size={14}
                  className={`flex-shrink-0 text-muted-foreground transition-transform mt-1 ${isOpen ? "rotate-180" : ""}`}
                />
              </button>

              {isOpen && <p className="text-xs text-muted-foreground mt-2 pl-6">{warning.explanation}</p>}

              <p className="text-xs text-muted-foreground mt-2 pl-6">
                <span className="text-muted-foreground">Action recommandée : </span>
                {warning.action}
              </p>

              {excludable.length > 0 && (
                <div className="flex flex-wrap items-center gap-2 mt-2 pl-6">
                  {excludable.map((col) =>
                    selectedFeatures?.has(col) ? (
                      <button
                        key={col}
                        type="button"
                        onClick={() => onExcludeColumns?.([col])}
                        className="inline-flex items-center gap-1 rounded-full border border-destructive/20 bg-card px-2.5 py-1 text-xs font-medium text-destructive hover:bg-destructive/10 transition-colors"
                      >
                        <XCircle size={12} />
                        Exclure « {col} »
                      </button>
                    ) : (
                      <span
                        key={col}
                        className="inline-flex items-center gap-1 rounded-full border border-success/20 bg-success/10 px-2.5 py-1 text-xs font-medium text-success"
                      >
                        <ShieldCheck size={12} />
                        « {col} » exclue
                      </span>
                    ),
                  )}
                </div>
              )}
            </div>
          );
        })}
    </div>
  );
}
