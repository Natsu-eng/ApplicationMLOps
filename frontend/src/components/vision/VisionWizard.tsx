import { ImageOff } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { ApiError, api, type AugmentationPreset, type AugmentationPreviewResult } from "../../api/client";
import { Badge } from "../ui/Badge";

// Lot 6A (correctif I9) — labels/descriptions des 4 presets, partagés par
// les deux wizards Vision (classification ET anomalies, même presets
// depuis la parité d'étapes de ce lot).
export const AUGMENTATION_PRESET_INFO: Record<AugmentationPreset, { label: string; description: string }> = {
  aucune: { label: "Aucune", description: "Images utilisées telles quelles, sans transformation." },
  legere: { label: "Légère", description: "Retournement horizontal seulement." },
  standard: { label: "Standard", description: "Retournement + légère rotation + variation de luminosité/contraste." },
  forte: { label: "Forte", description: "Standard, en plus marqué, + décalage et mise à l'échelle aléatoires." },
};

// Barre d'étapes : voir components/ui/WizardStepper.tsx — partagée avec
// Training.tsx (avant cette refonte, ce fichier avait déjà sa propre copie
// pour les 2 wizards Vision, malgré le commentaire ci-dessus revendiquant
// l'avoir "extrait" — l'extraction ne couvrait que 2 des 3 wizards).

/** Contenu d'une étape du wizard — titre + description en langage clair,
 * puis les champs propres à l'étape. */
export function StepContent({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium text-foreground">{title}</h3>
        {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
      </div>
      {children}
    </div>
  );
}

export function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="text-sm text-foreground">{value}</dd>
    </div>
  );
}

/** Répartition personnalisée (Lot 6A) — un curseur par split CONTRÔLÉ
 * (val/test pour la classification, val seul pour les anomalies, où test/
 * est un dossier séparé du dataset, jamais issu d'un split), le reste
 * (train) est TOUJOURS dérivé, jamais lui-même un curseur — impossible de
 * faire dépasser 100 % par construction, contrairement à 3 curseurs
 * indépendants qu'il faudrait ensuite valider/renormaliser. */
export function SplitRatioControl({
  totalImages,
  splits,
}: {
  totalImages: number | null;
  splits: { key: string; label: string; ratio: number; onChange: (ratio: number) => void; min: number; max: number }[];
}) {
  const heldOut = splits.reduce((sum, s) => sum + s.ratio, 0);
  const trainRatio = Math.max(0, 1 - heldOut);
  const countFor = (ratio: number) => (totalImages != null ? Math.round(totalImages * ratio) : null);

  return (
    <div className="space-y-3">
      {splits.map((s) => (
        <div key={s.key}>
          <label htmlFor={`split-${s.key}`} className="flex items-center justify-between text-sm text-muted-foreground mb-1">
            <span>{s.label}</span>
            <span className="tabular-nums text-foreground">
              {Math.round(s.ratio * 100)} %{countFor(s.ratio) != null && ` · ${countFor(s.ratio)} images`}
            </span>
          </label>
          <input
            id={`split-${s.key}`}
            type="range"
            min={s.min}
            max={s.max}
            step={0.01}
            value={s.ratio}
            onChange={(e) => s.onChange(Number(e.target.value))}
            className="w-full accent-primary"
          />
        </div>
      ))}
      <div className="flex items-center justify-between text-sm rounded-lg border border-border bg-muted px-3 py-2">
        <span className="text-muted-foreground">Entraînement (reste)</span>
        <span className="tabular-nums text-foreground font-medium">
          {Math.round(trainRatio * 100)} %{countFor(trainRatio) != null && ` · ${countFor(trainRatio)} images`}
        </span>
      </div>
    </div>
  );
}

/** Déséquilibre de classes (Lot 6A) — signal factuel calculé côté client à
 * partir de `class_distribution` (déjà renvoyé par `VisionDatasetDetail`,
 * aucun appel réseau supplémentaire), jamais un jugement : affiche le
 * ratio classe majoritaire/minoritaire et pointe vers le réglage qui aide
 * réellement (pondération de classes, déjà implémentée — I8) plutôt que de
 * juste signaler le problème sans action possible. */
export function classImbalanceRatio(classDistribution: Record<string, number> | undefined): number | null {
  if (!classDistribution) return null;
  const counts = Object.values(classDistribution).filter((n) => n > 0);
  if (counts.length < 2) return null;
  return Math.max(...counts) / Math.min(...counts);
}

const IMBALANCE_WARN_THRESHOLD = 3;

export function ClassImbalanceBanner({ classDistribution }: { classDistribution: Record<string, number> | undefined }) {
  const ratio = classImbalanceRatio(classDistribution);
  if (ratio === null || ratio < IMBALANCE_WARN_THRESHOLD) return null;
  const entries = Object.entries(classDistribution ?? {}).sort((a, b) => b[1] - a[1]);
  const [majorityName, majorityCount] = entries[0];
  const [minorityName, minorityCount] = entries[entries.length - 1];
  return (
    <div className="rounded-lg border border-warning/20 bg-warning/10 p-3 text-sm text-warning">
      <p>
        Classes déséquilibrées — « {majorityName} » ({majorityCount} images) contre « {minorityName} » (
        {minorityCount} images), soit {ratio.toFixed(1)}× plus d'exemples pour la classe majoritaire.
      </p>
      <p className="mt-1 text-xs">
        La pondération de classes (Mode expert, activée par défaut) compense ce déséquilibre pendant
        l'entraînement — sans elle, le modèle peut apprendre à toujours prédire la classe majoritaire.
      </p>
    </div>
  );
}

/** Grille de sélection des 4 presets d'augmentation (Lot 6A) — extrait de
 * VisionClassification.tsx pour être partagé avec VisionAnomalies.tsx
 * (parité des étapes, même lot). `recommendedPreset` reste optionnel : les
 * anomalies n'ont pas de recommandation calculée côté backend (contrairement
 * à la classification, `recommend_augmentation_preset`), le badge "Recommandé"
 * ne s'affiche simplement jamais dans ce cas. */
export function AugmentationPresetPicker({
  value,
  onChange,
  recommendedPreset,
}: {
  value: AugmentationPreset;
  onChange: (preset: AugmentationPreset) => void;
  recommendedPreset?: AugmentationPreset | null;
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {(Object.keys(AUGMENTATION_PRESET_INFO) as AugmentationPreset[]).map((preset) => {
        const info = AUGMENTATION_PRESET_INFO[preset];
        const isSelected = value === preset;
        const isRecommended = recommendedPreset === preset;
        return (
          <button
            key={preset}
            type="button"
            onClick={() => onChange(preset)}
            aria-pressed={isSelected}
            className={`text-left rounded-xl border p-3 transition-colors ${
              isSelected ? "border-primary/50 bg-primary/5" : "border-border hover:bg-muted"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <p className="text-sm font-medium text-foreground">{info.label}</p>
              {isRecommended && <Badge variant="primary">Recommandé</Badge>}
            </div>
            <p className="text-xs text-muted-foreground mt-1">{info.description}</p>
          </button>
        );
      })}
    </div>
  );
}

/** Aperçu avant/après d'un preset d'augmentation (Lot 6A) — appelle le
 * backend (voir GET /vision/datasets/{id}/augmentation-preview) pour
 * appliquer la VRAIE transformation d'entraînement à quelques images
 * réelles du dataset, jamais une approximation CSS côté client (qui
 * divergerait silencieusement des transformations torchvision réelles).
 * Partagé par les deux wizards Vision. */
export function AugmentationPreviewGallery({ datasetId, preset }: { datasetId: number | ""; preset: AugmentationPreset }) {
  const [result, setResult] = useState<AugmentationPreviewResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!datasetId) {
      setResult(null);
      return;
    }
    setLoading(true);
    setError(null);
    api.visionDatasets
      .augmentationPreview(datasetId, preset)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Aperçu indisponible"))
      .finally(() => setLoading(false));
  }, [datasetId, preset]);

  if (!datasetId) return null;

  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        Aperçu sur {result?.pairs.length ?? "…"} image{result && result.pairs.length > 1 ? "s" : ""} réelle
        {result && result.pairs.length > 1 ? "s" : ""} du dataset — à gauche l'original, à droite après
        transformation.
      </p>
      {loading && <p className="text-xs text-muted-foreground">Génération de l'aperçu…</p>}
      {error && (
        <div className="flex items-center gap-2 text-xs text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
          <ImageOff size={13} className="flex-shrink-0" />
          {error}
        </div>
      )}
      {result && result.pairs.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {result.pairs.map((pair, i) => (
            <div key={i} className="grid grid-cols-2 gap-1 rounded-lg overflow-hidden border border-border">
              <img src={pair.original_png} alt="Image originale" className="w-full aspect-square object-cover" />
              <img src={pair.augmented_png} alt="Image après augmentation" className="w-full aspect-square object-cover" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
