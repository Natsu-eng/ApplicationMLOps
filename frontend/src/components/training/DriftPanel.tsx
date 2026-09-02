import { useEffect, useState } from "react";
import { Waves } from "lucide-react";
import { ApiError, api, apiErrorReference, type DriftReport } from "../../api/client";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { ErrorNote } from "../ui/ErrorNote";
import { SectionHeader } from "../ui/SectionHeader";

const SEVERITY_LABEL: Record<DriftReport["features"][number]["severity"], string> = {
  stable: "Stable",
  modere: "Modérée",
  significatif: "Significative",
};

const SEVERITY_VARIANT: Record<DriftReport["features"][number]["severity"], "success" | "warning" | "danger"> = {
  stable: "success",
  modere: "warning",
  significatif: "danger",
};

/** Dérive des données — première fonctionnalité de suivi POST-déploiement
 * du produit (verdict/seuil/fiabilité s'arrêtent tous à l'instant du
 * déploiement). Compare la distribution des variables réellement envoyées
 * en production à celle du dataset d'entraînement (PSI, voir
 * backend/domains/shared/drift.py) — répond à la question que le tableau
 * de bord pose déjà sans jamais permettre d'y répondre ("le plus ancien
 * n'a pas été revérifié — contrôler sa dérive"). Chargée à part (pas dans
 * `MLModelDetail`) : nécessite de relire le dataset d'entraînement, plus
 * coûteux que le reste de la fiche modèle. */
export function DriftPanel({ jobId }: { jobId: number }) {
  const [report, setReport] = useState<DriftReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorRef, setErrorRef] = useState<string | undefined>(undefined);

  useEffect(() => {
    api.training
      .getDrift(jobId)
      .then(setReport)
      .catch((err) => {
        setError(err instanceof ApiError ? err.message : "Impossible de calculer la dérive");
        setErrorRef(apiErrorReference(err));
      });
  }, [jobId]);

  if (error) return <ErrorNote message={error} reference={errorRef} />;
  if (report === null) return <p className="text-sm text-muted-foreground">Calcul en cours…</p>;

  if (report.insufficient_data) {
    return (
      <Card className="p-5">
        <SectionHeader
          icon={Waves}
          color="blue"
          label="Dérive des données"
          help="Compare les variables envoyées en production à celles du dataset d'entraînement (PSI). Nécessite un minimum de prédictions récentes pour être fiable."
        />
        <p className="text-sm text-muted-foreground mt-2">
          {report.n_predictions_analyzed === 0
            ? "Aucune prédiction enregistrée pour ce modèle pour l'instant."
            : `Seulement ${report.n_predictions_analyzed} prédiction${report.n_predictions_analyzed > 1 ? "s" : ""} enregistrée${report.n_predictions_analyzed > 1 ? "s" : ""}.`}{" "}
          Au moins {report.min_predictions_required} sont nécessaires pour un calcul fiable — revenez une fois ce
          modèle davantage utilisé en prédiction.
        </p>
      </Card>
    );
  }

  const hasIssue = report.n_significant > 0 || report.n_moderate > 0;

  return (
    <Card className="p-5">
      <SectionHeader
        icon={Waves}
        color="blue"
        label="Dérive des données"
        help="Compare la distribution de chaque variable, telle qu'envoyée récemment en prédiction, à celle du dataset d'entraînement — un score élevé signale un écart qui peut expliquer une baisse de qualité, même sans connaître le résultat réel."
      />
      <p className="text-sm text-muted-foreground mt-2 mb-4">
        Basé sur les {report.n_predictions_analyzed} prédictions les plus récentes.{" "}
        {hasIssue ? (
          <span className="text-foreground">
            {report.n_significant > 0 &&
              `${report.n_significant} variable${report.n_significant > 1 ? "s" : ""} en dérive significative`}
            {report.n_significant > 0 && report.n_moderate > 0 && ", "}
            {report.n_moderate > 0 && `${report.n_moderate} en dérive modérée`}.
          </span>
        ) : (
          "Aucune dérive notable détectée."
        )}
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="py-2 pr-4 font-normal">Variable</th>
              <th className="py-2 pr-4 font-normal">PSI</th>
              <th className="py-2 font-normal">Sévérité</th>
            </tr>
          </thead>
          <tbody>
            {report.features.map((f) => (
              <tr key={f.feature} className="border-b border-border/60 last:border-0">
                <td className="py-2 pr-4 text-foreground/90">{f.feature}</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-foreground/90">{f.psi.toFixed(3)}</td>
                <td className="py-2">
                  <Badge variant={SEVERITY_VARIANT[f.severity]}>{SEVERITY_LABEL[f.severity]}</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
