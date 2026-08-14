import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { Activity, BrainCircuit, Database, FileSpreadsheet, ScrollText, Trash2, Users } from "lucide-react";
import {
  ApiError,
  api,
  type AuditLogEntry,
  type DatasetSummary,
  type TeamMember,
  type TrainingJobSummary,
} from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import AppShell from "../components/AppShell";
import { StatTile, StatTileRow } from "../components/dashboard/StatTile";
import ModelResultModal from "../components/training/ModelResultModal";
import { Avatar } from "../components/ui/Avatar";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentColorForId } from "../components/ui/ColorIconBadge";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Input } from "../components/ui/Input";
import { formatDateTime } from "../utils/format";

const AUDIT_ACTION_LABELS: Record<string, string> = {
  "member.added": "Membre ajouté",
  "dataset.deleted": "Dataset supprimé",
  "training_job.deleted": "Entraînement supprimé",
  "model.promoted": "Modèle promu",
};

function auditActionLabel(entry: AuditLogEntry): string {
  const base = AUDIT_ACTION_LABELS[entry.action] ?? entry.action;
  if (entry.action === "member.added" && entry.details?.email) return `${base} — ${entry.details.email}`;
  if (entry.action === "dataset.deleted" && entry.details?.name) return `${base} — ${entry.details.name}`;
  if (entry.action === "training_job.deleted" && entry.details?.target_column) {
    return `${base} — cible « ${entry.details.target_column} »`;
  }
  if (entry.action === "model.promoted" && entry.details?.stage) {
    const stage = entry.details.stage;
    const stageLabel = stage === "production" ? "production" : stage === "staging" ? "validation" : "retiré";
    return `${base} (${entry.details.algorithm ?? ""}) → ${stageLabel}`;
  }
  return base;
}

/** Journal d'audit (Lot 10, owner uniquement) — actions sensibles de
 * l'équipe (ajout de membre, suppression de dataset/entraînement,
 * promotion de modèle), pas un log applicatif complet (déjà couvert côté
 * serveur) : juste ce qu'un owner voudrait pouvoir vérifier après coup. */
function AuditLogPanel() {
  const [entries, setEntries] = useState<AuditLogEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.team
      .auditLog()
      .then(setEntries)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Journal indisponible"));
  }, []);

  return (
    <>
      <SectionHeader
        icon={ScrollText}
        color="amber"
        label="Journal d'audit"
        help="Actions sensibles de l'équipe — ajout de membre, suppression de dataset/entraînement, promotion de modèle. Visible uniquement par le propriétaire de l'organisation."
      />
      {error && <p className="text-sm text-rose-600">{error}</p>}
      {entries === null && !error && <p className="text-sm text-slate-500">Chargement…</p>}
      {entries && entries.length === 0 && <p className="text-sm text-slate-500">Aucune action enregistrée pour l'instant.</p>}
      {entries && entries.length > 0 && (
        <ul className="divide-y divide-slate-200 max-h-72 overflow-y-auto">
          {entries.slice(0, 20).map((entry) => (
            <li key={entry.id} className="py-2 flex items-center justify-between gap-3">
              <p className="text-sm text-slate-700 truncate">{auditActionLabel(entry)}</p>
              <span className="text-xs text-slate-400 flex-shrink-0">
                {entry.actor_name ?? "—"} · {formatDateTime(entry.created_at)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

const ACTIVE_STATUSES = new Set(["queued", "running"]);

/** Salutation contextuelle à l'heure réelle du navigateur — jamais un texte
 * figé. Convention FR usuelle : "Bonjour" en journée, "Bonsoir" le soir/la
 * nuit (5h–18h / 18h–5h). */
function greeting(hour: number): string {
  return hour >= 5 && hour < 18 ? "Bonjour" : "Bonsoir";
}

function jobStatusBadge(job: TrainingJobSummary) {
  switch (job.status) {
    case "completed":
      return <Badge variant="success" dot>Terminé</Badge>;
    case "failed":
      return <Badge variant="danger" dot>Échec</Badge>;
    case "running":
      return <Badge variant="primary" dot pulse>En cours</Badge>;
    default:
      return <Badge variant="neutral" dot>En file</Badge>;
  }
}

function datasetStatusBadge(status: DatasetSummary["status"]) {
  if (status === "ready") return <Badge variant="success" dot>Prêt</Badge>;
  if (status === "error") return <Badge variant="danger" dot>Erreur</Badge>;
  return <Badge variant="primary" dot pulse>Analyse…</Badge>;
}


/** Page protégée du Lot 1, enrichie au Lot E1-ter : vue d'ensemble de
 * l'activité (datasets, entraînements récents) au-dessus de la gestion
 * d'équipe — le dashboard doit d'abord montrer ce qui se passe, pas
 * seulement qui a accès. */
export default function Dashboard() {
  const { user } = useAuth();

  const [members, setMembers] = useState<TeamMember[] | null>(null);
  const [membersError, setMembersError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [jobs, setJobs] = useState<TrainingJobSummary[] | null>(null);
  const [viewingJob, setViewingJob] = useState<TrainingJobSummary | null>(null);
  const [confirmingDeleteJobId, setConfirmingDeleteJobId] = useState<number | null>(null);

  const loadMembers = useCallback(async () => {
    try {
      setMembers(await api.team.members());
      setMembersError(null);
    } catch (err) {
      setMembersError(err instanceof ApiError ? err.message : "Impossible de charger l'équipe");
    }
  }, []);

  const loadJobs = useCallback(() => {
    api.training.listJobs().then(setJobs).catch(() => setJobs([]));
  }, []);

  useEffect(() => {
    loadMembers();
    api.datasets.list().then(setDatasets).catch(() => setDatasets([]));
    loadJobs();
  }, [loadMembers, loadJobs]);

  // Deux clics pour confirmer (même motif que la page Entraînement) — pas de
  // window.confirm natif, pour rester cohérent avec le reste de l'app.
  async function handleDeleteJob(id: number) {
    if (confirmingDeleteJobId !== id) {
      setConfirmingDeleteJobId(id);
      return;
    }
    setConfirmingDeleteJobId(null);
    try {
      await api.training.remove(id);
      loadJobs();
    } catch {
      // best-effort — la liste se resynchronisera au prochain chargement
    }
  }

  if (!user) return null;

  const recentJobs = jobs?.slice(0, 5) ?? [];
  const recentDatasets = datasets?.slice(0, 5) ?? [];
  const activeJobsCount = jobs?.filter((j) => ACTIVE_STATUSES.has(j.status)).length ?? 0;

  return (
    <AppShell pillarId="supervised">
      <div className="mb-8 flex items-center justify-between flex-wrap gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-primary font-semibold mb-1">
            Vue d'ensemble
          </p>
          <h1 className="text-2xl font-serif text-slate-900">
            {greeting(new Date().getHours())}, {user.nom.split(" ")[0]}
          </h1>
        </div>
        <Link to="/training">
          <Button>
            <BrainCircuit size={15} />
            Nouvel entraînement
          </Button>
        </Link>
      </div>

      <StatTileRow>
        <StatTile icon={Database} label="Datasets" value={datasets?.length} color="blue" delayMs={0} />
        <StatTile icon={BrainCircuit} label="Entraînements" value={jobs?.length} color="teal" delayMs={60} />
        <StatTile
          icon={Activity}
          label="En cours"
          value={jobs ? activeJobsCount : undefined}
          color="amber"
          delayMs={120}
        />
        <StatTile icon={Users} label="Membres de l'équipe" value={members?.length} color="violet" delayMs={180} />
      </StatTileRow>

      <div className="grid gap-6 lg:grid-cols-2 mb-10">
        <Card className="p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium text-slate-800">Derniers entraînements</h2>
            <Link to="/training/history" className="text-xs text-primary hover:text-primary/80">
              Voir tout
            </Link>
          </div>

          {jobs === null ? (
            <p className="text-sm text-slate-500">Chargement…</p>
          ) : recentJobs.length === 0 ? (
            <p className="text-sm text-slate-500">
              Aucun entraînement pour l'instant — lancez-en un depuis{" "}
              <Link to="/training" className="text-primary hover:text-primary/80">
                Entraînement
              </Link>
              .
            </p>
          ) : (
            <ul className="divide-y divide-slate-200">
              {recentJobs.map((job) => {
                const isCompleted = job.status === "completed";
                return (
                  <li
                    key={job.id}
                    onClick={() => isCompleted && setViewingJob(job)}
                    className={`group py-2.5 flex items-center justify-between gap-3 ${
                      isCompleted ? "cursor-pointer hover:bg-slate-50 -mx-1 px-1 rounded-lg" : ""
                    }`}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <ColorIconBadge icon={BrainCircuit} color={accentColorForId(job.id)} size="sm" />
                      <div className="min-w-0">
                        <p className="text-sm text-slate-800 truncate">
                          {job.dataset_name ?? "Dataset"} <span className="text-slate-400">→</span>{" "}
                          {job.target_column}
                        </p>
                        <p className="text-xs text-slate-500">{formatDateTime(job.created_at)}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {isCompleted && job.headline_metric && (
                        <span className="text-xs text-slate-500 tabular-nums">
                          {job.headline_metric.name} = {job.headline_metric.value?.toFixed(3) ?? "—"}
                        </span>
                      )}
                      {jobStatusBadge(job)}
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteJob(job.id);
                        }}
                        onMouseLeave={() => setConfirmingDeleteJobId((id) => (id === job.id ? null : id))}
                        aria-label={
                          confirmingDeleteJobId === job.id ? "Confirmer la suppression" : "Supprimer cet entraînement"
                        }
                        title={
                          confirmingDeleteJobId === job.id
                            ? "Cliquer à nouveau pour confirmer"
                            : "Supprimer cet entraînement"
                        }
                        className={`flex-shrink-0 h-7 w-7 flex items-center justify-center rounded-md transition-colors ${
                          confirmingDeleteJobId === job.id
                            ? "text-rose-700 bg-rose-100"
                            : "text-slate-300 hover:text-rose-600 hover:bg-rose-50"
                        }`}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>

        <Card className="p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium text-slate-800">Derniers datasets</h2>
            <Link to="/datasets" className="text-xs text-primary hover:text-primary/80">
              Voir tout
            </Link>
          </div>

          {datasets === null ? (
            <p className="text-sm text-slate-500">Chargement…</p>
          ) : recentDatasets.length === 0 ? (
            <p className="text-sm text-slate-500">
              Aucun dataset pour l'instant — importez-en un depuis{" "}
              <Link to="/datasets" className="text-primary hover:text-primary/80">
                Mes données
              </Link>
              .
            </p>
          ) : (
            <ul className="divide-y divide-slate-200">
              {recentDatasets.map((dataset) => (
                <li key={dataset.id} className="group py-2.5 flex items-center justify-between gap-3 hover:bg-slate-50 -mx-1 px-1 rounded-lg transition-colors">
                  <div className="flex items-center gap-3 min-w-0">
                    <ColorIconBadge icon={FileSpreadsheet} color={accentColorForId(dataset.id)} size="sm" />
                    <div className="min-w-0">
                      <p className="text-sm text-slate-800 truncate">{dataset.name}</p>
                      <p className="text-xs text-slate-500">
                        {dataset.row_count ?? "—"} lignes · {dataset.column_count ?? "—"} colonnes
                      </p>
                    </div>
                  </div>
                  {datasetStatusBadge(dataset.status)}
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="p-5 lg:col-span-1">
          <div className="flex items-center gap-3 mb-4">
            <Avatar name={user.nom} />
            <div className="min-w-0">
              <p className="font-medium text-slate-900 truncate">{user.nom}</p>
              <p className="text-xs text-slate-500 truncate">{user.email}</p>
            </div>
          </div>
          <RoleBadge role={user.role} />
        </Card>

        <Card className="p-5 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Users size={16} className="text-primary" />
              <h2 className="text-sm font-medium text-slate-800">
                Équipe — {user.organization_name}
              </h2>
            </div>
            {members && (
              <Badge variant="neutral">
                {members.length} membre{members.length > 1 ? "s" : ""}
              </Badge>
            )}
          </div>

          {membersError && <ErrorNote message={membersError} />}

          {members === null && !membersError ? (
            <p className="text-sm text-slate-500">Chargement…</p>
          ) : (
            <ul className="divide-y divide-slate-200">
              {members?.map((member) => (
                <li key={member.id} className="py-2.5 flex items-center justify-between">
                  <div className="flex items-center gap-3 min-w-0">
                    <Avatar name={member.nom} size="sm" />
                    <div className="min-w-0">
                      <p className="text-sm text-slate-800 truncate">{member.nom}</p>
                      <p className="text-xs text-slate-500 truncate">{member.email}</p>
                    </div>
                  </div>
                  <RoleBadge role={member.role} />
                </li>
              ))}
            </ul>
          )}
        </Card>

        {user.role === "owner" && (
          <Card className="p-5 lg:col-span-3">
            <AddMemberForm onMemberAdded={loadMembers} />
          </Card>
        )}

        {user.role === "owner" && (
          <Card className="p-5 lg:col-span-3">
            <AuditLogPanel />
          </Card>
        )}
      </div>

      {viewingJob && <ModelResultModal job={viewingJob} onClose={() => setViewingJob(null)} />}
    </AppShell>
  );
}

function RoleBadge({ role }: { role: "owner" | "member" }) {
  return (
    <Badge variant={role === "owner" ? "accent" : "neutral"}>
      {role === "owner" ? "Propriétaire" : "Membre"}
    </Badge>
  );
}

function ErrorNote({ message }: { message: string }) {
  return (
    <p className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2 mb-3">
      {message}
    </p>
  );
}

function AddMemberForm({ onMemberAdded }: { onMemberAdded: () => void }) {
  const [email, setEmail] = useState("");
  const [nom, setNom] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSuccess(false);
    setIsSubmitting(true);
    try {
      await api.team.addMember({ email, nom, password });
      setEmail("");
      setNom("");
      setPassword("");
      setSuccess(true);
      onMemberAdded();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible d'ajouter ce membre");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <>
      <h2 className="text-sm font-medium text-slate-800 mb-4">Ajouter un membre à l'équipe</h2>
      <form onSubmit={handleSubmit} className="grid sm:grid-cols-3 gap-3 items-start">
        <Input
          type="text"
          placeholder="Nom"
          required
          minLength={2}
          value={nom}
          onChange={(e) => setNom(e.target.value)}
        />
        <Input
          type="email"
          placeholder="Email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <Input
          type="password"
          placeholder="Mot de passe temporaire"
          required
          minLength={8}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <Button type="submit" disabled={isSubmitting} className="sm:col-span-3">
          {isSubmitting ? "Ajout…" : "Ajouter"}
        </Button>
      </form>
      {error && <p className="text-sm text-rose-600 mt-2">{error}</p>}
      {success && <p className="text-sm text-emerald-600 mt-2">Membre ajouté.</p>}
    </>
  );
}
