import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Activity, BrainCircuit, Database, FileSpreadsheet, LayoutDashboard, ScrollText, Trash2, Users } from "lucide-react";
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
import { PageHeader } from "../components/ui/PageHeader";
import { DatasetStatusBadge, JobStatusBadge } from "../components/ui/StatusBadge";
import { useConfirmAction } from "../hooks/useConfirmAction";
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
      {error && <p className="text-sm text-destructive">{error}</p>}
      {entries === null && !error && <p className="text-sm text-muted-foreground">Chargement…</p>}
      {entries && entries.length === 0 && (
        <p className="text-sm text-muted-foreground">Aucune action enregistrée pour l'instant.</p>
      )}
      {entries && entries.length > 0 && (
        <ul className="divide-y divide-border max-h-72 overflow-y-auto">
          {entries.slice(0, 20).map((entry) => (
            <li key={entry.id} className="py-2 flex items-center justify-between gap-3">
              <p className="text-sm text-foreground/90 truncate">{auditActionLabel(entry)}</p>
              <span className="text-xs text-muted-foreground flex-shrink-0">
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
 * figé. Trois tranches (retour utilisateur explicite : deux ne suffisaient
 * pas à distinguer le milieu de journée) : "Bonjour" le matin (5h–12h),
 * "Bon après-midi" l'après-midi (12h–18h), "Bonsoir" le soir/la nuit
 * (18h–5h) — convention FR usuelle. */
function greeting(hour: number): string {
  if (hour >= 5 && hour < 12) return "Bonjour";
  if (hour >= 12 && hour < 18) return "Bon après-midi";
  return "Bonsoir";
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
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<TrainingJobSummary[] | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [viewingJob, setViewingJob] = useState<TrainingJobSummary | null>(null);
  const confirmDeleteJob = useConfirmAction<number>();

  // Résultat "deep-linkable" (AUDIT_ROADMAP.md, H20/D12 — signalé après le
  // correctif de persistance de la page Entraînement) : avant ce correctif,
  // ouvrir un résultat d'entraînement ne changeait jamais l'URL — un
  // rafraîchissement fermait la modale sans recours, et un lien vers "ce
  // résultat précis" ne pouvait pas se partager. `?job=<id>` synchronise
  // l'URL avec la modale ouverte, dans les deux sens.
  const [searchParams, setSearchParams] = useSearchParams();

  useEffect(() => {
    const jobId = searchParams.get("job");
    if (!jobId) return;
    api.training
      .getJob(Number(jobId))
      .then(setViewingJob)
      .catch(() => setSearchParams({}, { replace: true }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function openJob(job: TrainingJobSummary) {
    setViewingJob(job);
    setSearchParams({ job: String(job.id) }, { replace: false });
  }

  function closeJob() {
    setViewingJob(null);
    setSearchParams({}, { replace: false });
  }

  const loadMembers = useCallback(async () => {
    try {
      setMembers(await api.team.members());
      setMembersError(null);
    } catch (err) {
      setMembersError(err instanceof ApiError ? err.message : "Impossible de charger l'équipe");
    }
  }, []);

  const loadJobs = useCallback(async () => {
    try {
      setJobs(await api.training.listJobs());
      setJobsError(null);
    } catch (err) {
      // AUDIT_ROADMAP.md, H4/D3 : distinguer "aucun entraînement" (liste
      // vide légitime) d'un échec réseau — avant ce correctif, les deux cas
      // affichaient exactement le même message "Aucun entraînement".
      setJobsError(err instanceof ApiError ? err.message : "Impossible de charger les entraînements");
    }
  }, []);

  const loadDatasets = useCallback(async () => {
    try {
      setDatasets(await api.datasets.list());
      setDatasetsError(null);
    } catch (err) {
      setDatasetsError(err instanceof ApiError ? err.message : "Impossible de charger les datasets");
    }
  }, []);

  useEffect(() => {
    loadMembers();
    loadDatasets();
    loadJobs();
  }, [loadMembers, loadDatasets, loadJobs]);

  async function handleDeleteJob(id: number) {
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
      <PageHeader
        eyebrow="Vue d'ensemble"
        title={`${greeting(new Date().getHours())}, ${user.nom.split(" ")[0]}`}
        description={`${user.organization_name} — voici l'activité récente de votre équipe.`}
        icon={LayoutDashboard}
        color="blue"
        action={
          <Link to="/training">
            <Button>
              <BrainCircuit size={15} />
              Nouvel entraînement
            </Button>
          </Link>
        }
      />

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
            <h2 className="text-sm font-medium text-foreground">Derniers entraînements</h2>
            <Link to="/training/history" className="text-xs text-primary hover:text-primary/80">
              Voir tout
            </Link>
          </div>

          {jobsError ? (
            <ErrorNote message={jobsError} />
          ) : jobs === null ? (
            <p className="text-sm text-muted-foreground">Chargement…</p>
          ) : recentJobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Aucun entraînement pour l'instant — lancez-en un depuis{" "}
              <Link to="/training" className="text-primary hover:text-primary/80">
                Entraînement
              </Link>
              .
            </p>
          ) : (
            <ul className="divide-y divide-border">
              {recentJobs.map((job) => {
                const isCompleted = job.status === "completed";
                const pendingDelete = confirmDeleteJob.isPending(job.id);
                return (
                  <li
                    key={job.id}
                    onClick={() => isCompleted && openJob(job)}
                    className={`group py-2.5 flex items-center justify-between gap-3 ${
                      isCompleted ? "cursor-pointer hover:bg-muted/50 -mx-1 px-1 rounded-lg transition-colors" : ""
                    }`}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <ColorIconBadge icon={BrainCircuit} color={accentColorForId(job.id)} size="sm" />
                      <div className="min-w-0">
                        <p className="text-sm text-foreground/90 truncate">
                          {job.dataset_name ?? "Dataset"} <span className="text-muted-foreground">→</span>{" "}
                          {job.target_column}
                        </p>
                        <p className="text-xs text-muted-foreground">{formatDateTime(job.created_at)}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {isCompleted && job.headline_metric && (
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {job.headline_metric.name} = {job.headline_metric.value?.toFixed(3) ?? "—"}
                        </span>
                      )}
                      <JobStatusBadge status={job.status} />
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          confirmDeleteJob.trigger(job.id, () => handleDeleteJob(job.id));
                        }}
                        onMouseLeave={confirmDeleteJob.reset}
                        aria-label={pendingDelete ? "Confirmer la suppression" : "Supprimer cet entraînement"}
                        title={pendingDelete ? "Cliquer à nouveau pour confirmer" : "Supprimer cet entraînement"}
                        className={`flex-shrink-0 h-7 w-7 flex items-center justify-center rounded-md transition-colors ${
                          pendingDelete
                            ? "text-destructive bg-destructive/15"
                            : "text-muted-foreground/50 hover:text-destructive hover:bg-destructive/10"
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
            <h2 className="text-sm font-medium text-foreground">Derniers datasets</h2>
            <Link to="/datasets" className="text-xs text-primary hover:text-primary/80">
              Voir tout
            </Link>
          </div>

          {datasetsError ? (
            <ErrorNote message={datasetsError} />
          ) : datasets === null ? (
            <p className="text-sm text-muted-foreground">Chargement…</p>
          ) : recentDatasets.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Aucun dataset pour l'instant — importez-en un depuis{" "}
              <Link to="/datasets" className="text-primary hover:text-primary/80">
                Mes données
              </Link>
              .
            </p>
          ) : (
            <ul className="divide-y divide-border">
              {recentDatasets.map((dataset) => (
                <li key={dataset.id} className="group py-2.5 flex items-center justify-between gap-3 hover:bg-muted/50 -mx-1 px-1 rounded-lg transition-colors">
                  <div className="flex items-center gap-3 min-w-0">
                    <ColorIconBadge icon={FileSpreadsheet} color={accentColorForId(dataset.id)} size="sm" />
                    <div className="min-w-0">
                      <p className="text-sm text-foreground/90 truncate">{dataset.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {dataset.row_count ?? "—"} lignes · {dataset.column_count ?? "—"} colonnes
                      </p>
                    </div>
                  </div>
                  <DatasetStatusBadge status={dataset.status} />
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
              <p className="font-medium text-foreground truncate">{user.nom}</p>
              <p className="text-xs text-muted-foreground truncate">{user.email}</p>
            </div>
          </div>
          <RoleBadge role={user.role} />
        </Card>

        <Card className="p-5 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Users size={16} className="text-primary" />
              <h2 className="text-sm font-medium text-foreground">
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
            <p className="text-sm text-muted-foreground">Chargement…</p>
          ) : (
            <ul className="divide-y divide-border">
              {members?.map((member) => (
                <li key={member.id} className="py-2.5 flex items-center justify-between">
                  <div className="flex items-center gap-3 min-w-0">
                    <Avatar name={member.nom} size="sm" />
                    <div className="min-w-0">
                      <p className="text-sm text-foreground/90 truncate">{member.nom}</p>
                      <p className="text-xs text-muted-foreground truncate">{member.email}</p>
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

      {viewingJob && <ModelResultModal job={viewingJob} onClose={closeJob} />}
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
    <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2 mb-3">
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
      <h2 className="text-sm font-medium text-foreground mb-4">Ajouter un membre à l'équipe</h2>
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
      {error && <p className="text-sm text-destructive mt-2">{error}</p>}
      {success && <p className="text-sm text-success mt-2">Membre ajouté.</p>}
    </>
  );
}
