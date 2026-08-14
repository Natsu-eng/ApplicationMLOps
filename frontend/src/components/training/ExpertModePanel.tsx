import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { api, type ModelCatalogEntry } from "../../api/client";
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
}) {
  const [catalog, setCatalog] = useState<ModelCatalogEntry[]>([]);
  const [catalogLoaded, setCatalogLoaded] = useState(false);

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

  const families = Array.from(new Set(catalog.map((m) => m.family)));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-white px-3 py-2.5">
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
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                    {catalog
                      .filter((m) => m.family === family)
                      .map((model) => {
                        const checked = selectedModelIds.has(model.id);
                        return (
                          <label
                            key={model.id}
                            className={`flex items-center gap-2 text-xs rounded-lg border px-2.5 py-2 cursor-pointer transition-colors ${
                              checked
                                ? "border-primary/40 bg-primary/10 text-primary"
                                : "border-border bg-white text-foreground/90 hover:border-input"
                            }`}
                          >
                          <input
                            type="checkbox"
                            className="accent-primary"
                            checked={checked}
                            onChange={() => toggleModel(model.id)}
                          />
                          <span className="flex-1 min-w-0">{model.label}</span>
                          {model.slow && (
                            <Badge variant="warning">
                              <AlertTriangle size={10} className="mr-0.5" />
                              Lent
                            </Badge>
                          )}
                          {model.supported_tasks.length === 1 && (
                            <Badge variant="neutral">
                              {model.supported_tasks[0] === "classification" ? "Classif." : "Régression"}
                            </Badge>
                          )}
                          </label>
                        );
                      })}
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
