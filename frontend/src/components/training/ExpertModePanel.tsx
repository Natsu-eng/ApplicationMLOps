import { useEffect, useState } from "react";
import { AlertTriangle, ChevronDown, SlidersHorizontal } from "lucide-react";
import { api, type HyperparameterOverrides, type HyperparamMeta, type ModelCatalogEntry } from "../../api/client";
import { Badge } from "../ui/Badge";
import { Switch } from "../ui/Switch";
import { LabelWithHelp } from "../ui/Tooltip";

/** Défauts du mode expert (Lot E2) — STRICTEMENT ceux du mode guidé
 * d'aujourd'hui (même valeur envoyée que si le mode expert n'existait pas) :
 * activer le mode expert sans rien changer doit produire le même
 * entraînement qu'en mode guidé (exigence du cadrage, testée en vitest). */
export const DEFAULT_CV_FOLDS = 4; // `api.core.config.Settings.cv_folds_default`
export const DEFAULT_SEED = 42; // `api.core.config.Settings.model_seed`
export const DEFAULT_CQR_ALPHA = 0.2; // `api.core.config.Settings.cqr_alpha`

const FAMILY_LABELS: Record<string, string> = {
  arbre_ensemble: "Arbres & ensembles",
  lineaire: "Modèles linéaires",
  distance_noyau: "Distance & voisinage",
};

/** Panneau de manettes expertes (Lot E2) — replié par défaut, ignoré par
 * l'utilisateur non-DS. Chaque manette a une valeur par défaut identique au
 * mode guidé : l'activer sans rien toucher ne change aucun résultat. */
export function ExpertModePanel({
  expertMode,
  onExpertModeChange,
  optunaTrials,
  onOptunaTrialsChange,
  cvFolds,
  onCvFoldsChange,
  seed,
  onSeedChange,
  cqrAlpha,
  onCqrAlphaChange,
  selectedModelIds,
  onSelectedModelIdsChange,
  classRebalancing,
  onClassRebalancingChange,
  hyperparameterOverrides,
  onHyperparameterOverridesChange,
}: {
  expertMode: boolean;
  onExpertModeChange: (value: boolean) => void;
  optunaTrials: number;
  onOptunaTrialsChange: (value: number) => void;
  cvFolds: number;
  onCvFoldsChange: (value: number) => void;
  seed: number;
  onSeedChange: (value: number) => void;
  cqrAlpha: number;
  onCqrAlphaChange: (value: number) => void;
  selectedModelIds: Set<string>;
  onSelectedModelIdsChange: (ids: Set<string>) => void;
  classRebalancing: boolean;
  onClassRebalancingChange: (value: boolean) => void;
  // Mode expert hyperparamètres (retour utilisateur direct : "laisser le
  // choix sur les hyperparamètres, profondeur des arbres etc.").
  hyperparameterOverrides: HyperparameterOverrides;
  onHyperparameterOverridesChange: (overrides: HyperparameterOverrides) => void;
}) {
  const [catalog, setCatalog] = useState<ModelCatalogEntry[]>([]);
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  // Un seul panneau de réglages avancés ouvert à la fois (retour
  // utilisateur direct : "y'a trop d'éléments et ça part jusqu'en bas" —
  // même discipline que la pagination des tableaux : jamais tout déplié en
  // même temps par défaut, même ici sans liste à proprement parler).
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);

  // Chargé seulement à la première ouverture du mode expert — un utilisateur
  // guidé qui ne l'ouvre jamais ne déclenche aucun appel supplémentaire.
  useEffect(() => {
    if (!expertMode || catalogLoaded) return;
    api.training
      .modelsCatalog()
      .then(({ models }) => {
        setCatalog(models);
        // Pré-coche le sous-ensemble par défaut — cohérent avec le mode
        // guidé si l'utilisateur ne touche à rien.
        if (selectedModelIds.size === 0) {
          onSelectedModelIdsChange(new Set(models.filter((m) => m.is_default).map((m) => m.id)));
        }
      })
      .catch(() => setCatalog([]))
      .finally(() => setCatalogLoaded(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expertMode, catalogLoaded]);

  function toggleModel(id: string) {
    const next = new Set(selectedModelIds);
    if (next.has(id)) {
      if (next.size === 1) return; // toujours au moins un modèle sélectionné
      next.delete(id);
    } else {
      next.add(id);
    }
    onSelectedModelIdsChange(next);
  }

  // Fixe/libère UN hyperparamètre d'UN modèle — `value === undefined` le
  // retire de la surcharge (retour à la recherche automatique), jamais un
  // objet vide laissé pour ce modèle (nettoyé s'il ne reste plus rien).
  function setOverride(modelId: string, paramName: string, value: number | string | undefined) {
    const modelOverrides = { ...(hyperparameterOverrides[modelId] ?? {}) };
    if (value === undefined) {
      delete modelOverrides[paramName];
    } else {
      modelOverrides[paramName] = value;
    }
    const next = { ...hyperparameterOverrides };
    if (Object.keys(modelOverrides).length > 0) {
      next[modelId] = modelOverrides;
    } else {
      delete next[modelId];
    }
    onHyperparameterOverridesChange(next);
  }

  const families = Array.from(new Set(catalog.map((m) => m.family)));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card px-3 py-2.5">
        <div>
          <p className="text-sm font-medium text-foreground">Mode expert</p>
          <p className="text-xs text-muted-foreground">
            Affiche les réglages techniques du moteur (modèles comparés, validation croisée, graine
            aléatoire…). Inutile pour un usage courant — les valeurs par défaut du mode guidé sont déjà
            de bons choix.
          </p>
        </div>
        <Switch checked={expertMode} onChange={onExpertModeChange} label="Activer le mode expert" />
      </div>

      {expertMode && (
        <div className="space-y-5 rounded-lg border border-primary/20 bg-primary/5 p-3.5">
          <div>
            <label htmlFor="expert-optuna-trials" className="block text-sm text-muted-foreground mb-1">
              <LabelWithHelp
                label={`Recherche d'hyperparamètres — ${optunaTrials} essais`}
                help="À chaque essai, l'outil teste une combinaison de réglages internes pour chaque modèle et garde la meilleure. Plus élevé = recherche plus fine, mais entraînement plus long."
              />
            </label>
            <input
              id="expert-optuna-trials"
              type="range"
              min={5}
              max={60}
              step={5}
              value={optunaTrials}
              onChange={(e) => onOptunaTrialsChange(Number(e.target.value))}
              className="w-full accent-primary"
            />
          </div>

          <div>
            <label htmlFor="expert-cv-folds" className="block text-sm text-muted-foreground mb-1">
              <LabelWithHelp
                label={`Nombre de blocs de validation croisée — ${cvFolds}`}
                help="Le jeu d'entraînement est découpé en ce nombre de blocs : chaque modèle est évalué plusieurs fois en tournant le bloc de test. Plus de blocs = évaluation plus fiable, mais plus lente."
              />
            </label>
            <input
              id="expert-cv-folds"
              type="range"
              min={2}
              max={10}
              step={1}
              value={cvFolds}
              onChange={(e) => onCvFoldsChange(Number(e.target.value))}
              className="w-full accent-primary"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="expert-seed" className="block text-sm text-muted-foreground mb-1">
                <LabelWithHelp
                  label="Graine aléatoire"
                  help="Fixe le hasard utilisé pendant l'entraînement (découpage, initialisation des modèles…). Deux entraînements avec la même graine et les mêmes données donnent le même résultat — utile pour reproduire un résultat exact."
                />
              </label>
              <input
                id="expert-seed"
                type="number"
                min={0}
                max={99999}
                value={seed}
                onChange={(e) => onSeedChange(Number(e.target.value))}
                className="w-full rounded-lg border border-input bg-card px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
              />
            </div>

            <div>
              <label htmlFor="expert-cqr-alpha" className="block text-sm text-muted-foreground mb-1">
                <LabelWithHelp
                  label={`Confiance des intervalles — ${Math.round((1 - cqrAlpha) * 100)} %`}
                  help="Régression uniquement. Le modèle donne, en plus de sa prédiction, une fourchette de valeurs probables. Une confiance plus haute élargit cette fourchette pour être sûr d'avoir raison plus souvent."
                />
              </label>
              <input
                id="expert-cqr-alpha"
                type="range"
                min={0.05}
                max={0.5}
                step={0.05}
                value={cqrAlpha}
                onChange={(e) => onCqrAlphaChange(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
          </div>

          <div className="flex items-center justify-between gap-3">
            <LabelWithHelp
              label="Rééquilibrer les classes (classification)"
              help="Donne plus de poids aux classes rares pendant l'entraînement, au prix de plus de fausses alertes sur la classe majoritaire. Sans effet en régression. Si un déséquilibre est détecté, une suggestion contextuelle apparaît aussi à l'étape précédente — cette case permet de forcer ou d'annuler ce choix manuellement."
            />
            <Switch
              checked={classRebalancing}
              onChange={onClassRebalancingChange}
              label="Rééquilibrer les classes"
            />
          </div>

          <div>
            <p className="text-sm text-muted-foreground mb-2">
              <LabelWithHelp
                label={`Modèles comparés — ${selectedModelIds.size} sélectionné${selectedModelIds.size > 1 ? "s" : ""}`}
                help="L'outil entraîne chaque modèle coché et garde le meilleur sur la validation croisée. En cocher plus augmente les chances de trouver un meilleur modèle, mais rallonge l'entraînement."
              />
            </p>
            {!catalogLoaded && <p className="text-xs text-muted-foreground">Chargement du catalogue…</p>}
            <div className="space-y-3">
              {families.map((family) => (
                <div key={family}>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1.5">
                    {FAMILY_LABELS[family] ?? family}
                  </p>
                  <div className="space-y-1.5">
                    {catalog
                      .filter((m) => m.family === family)
                      .map((model) => (
                        <ModelRow
                          key={model.id}
                          model={model}
                          checked={selectedModelIds.has(model.id)}
                          onToggle={() => toggleModel(model.id)}
                          expanded={expandedModelId === model.id}
                          onToggleExpanded={() => setExpandedModelId((prev) => (prev === model.id ? null : model.id))}
                          overrides={hyperparameterOverrides[model.id] ?? {}}
                          onSetOverride={(name, value) => setOverride(model.id, name, value)}
                        />
                      ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ModelRow({
  model,
  checked,
  onToggle,
  expanded,
  onToggleExpanded,
  overrides,
  onSetOverride,
}: {
  model: ModelCatalogEntry;
  checked: boolean;
  onToggle: () => void;
  expanded: boolean;
  onToggleExpanded: () => void;
  overrides: Record<string, number | string>;
  onSetOverride: (name: string, value: number | string | undefined) => void;
}) {
  const hasTunableParams = model.tunable_hyperparameters.length > 0;
  const fixedCount = Object.keys(overrides).length;

  return (
    <div
      className={`rounded-lg border transition-colors ${
        checked ? "border-primary/40 bg-primary/10" : "border-border bg-card"
      }`}
    >
      <div className="flex items-center gap-2 px-2.5 py-2">
        <label className="flex items-center gap-2 text-xs flex-1 min-w-0 cursor-pointer">
          <input type="checkbox" className="accent-primary" checked={checked} onChange={onToggle} />
          <span className={`flex-1 min-w-0 ${checked ? "text-primary" : "text-foreground/90"}`}>{model.label}</span>
        </label>
        {model.slow && (
          <Badge variant="warning">
            <AlertTriangle size={10} className="mr-0.5" />
            Lent
          </Badge>
        )}
        {model.supported_tasks.length === 1 && (
          <Badge variant="neutral">{model.supported_tasks[0] === "classification" ? "Classif." : "Régression"}</Badge>
        )}
        {checked && hasTunableParams && (
          <button
            type="button"
            onClick={onToggleExpanded}
            className={`flex items-center gap-1 text-xs rounded-full px-2 py-0.5 border transition-colors flex-shrink-0 ${
              fixedCount > 0
                ? "border-primary/40 bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <SlidersHorizontal size={11} />
            {fixedCount > 0 ? `${fixedCount} fixé${fixedCount > 1 ? "s" : ""}` : "Réglages avancés"}
            <ChevronDown size={11} className={`transition-transform ${expanded ? "rotate-180" : ""}`} />
          </button>
        )}
      </div>

      {checked && expanded && hasTunableParams && (
        <div className="space-y-3 border-t border-border/60 px-2.5 py-3">
          {model.tunable_hyperparameters.map((meta) => (
            <HyperparamControl
              key={meta.name}
              meta={meta}
              value={overrides[meta.name]}
              onChange={(v) => onSetOverride(meta.name, v)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** Un seul hyperparamètre réglable — "Auto" (recherché par Optuna, comme
 * aujourd'hui) tant que l'utilisateur n'a pas explicitement coché "Fixer" ;
 * jamais une valeur imposée par accident (curseur touché sans l'avoir
 * voulu) puisque le contrôle reste désactivé tant que "Fixer" n'est pas
 * coché. */
function HyperparamControl({
  meta,
  value,
  onChange,
}: {
  meta: HyperparamMeta;
  value: number | string | undefined;
  onChange: (value: number | string | undefined) => void;
}) {
  const isFixed = value !== undefined;
  const defaultValue: number | string =
    meta.kind === "categorical" ? meta.choices?.[0] ?? "" : ((meta.low ?? 0) + (meta.high ?? 0)) / 2;

  return (
    <div>
      <div className="flex items-center justify-between gap-2 mb-1">
        <LabelWithHelp label={meta.label} help={meta.help} />
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer flex-shrink-0">
          <input
            type="checkbox"
            className="accent-primary"
            checked={isFixed}
            onChange={(e) => onChange(e.target.checked ? defaultValue : undefined)}
          />
          Fixer
        </label>
      </div>
      {meta.kind === "categorical" ? (
        <select
          disabled={!isFixed}
          value={isFixed ? String(value) : String(defaultValue)}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-lg border border-input bg-card px-2.5 py-1.5 text-xs text-foreground disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-primary/40"
        >
          {(meta.choices ?? []).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      ) : (
        <div className="flex items-center gap-2">
          <input
            type="range"
            disabled={!isFixed}
            min={meta.low ?? 0}
            max={meta.high ?? 1}
            step={meta.kind === "int" ? 1 : ((meta.high ?? 1) - (meta.low ?? 0)) / 100}
            value={isFixed ? Number(value) : Number(defaultValue)}
            onChange={(e) => onChange(meta.kind === "int" ? Math.round(Number(e.target.value)) : Number(e.target.value))}
            className="w-full accent-primary disabled:opacity-50"
          />
          <span className="text-xs text-muted-foreground tabular-nums w-14 text-right flex-shrink-0">
            {isFixed ? (meta.kind === "int" ? Number(value) : Number(value).toPrecision(3)) : "auto"}
          </span>
        </div>
      )}
    </div>
  );
}
