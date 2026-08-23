import { useCallback, useEffect, useState, type FormEvent } from "react";
import { KeyRound, Palette, ScrollText, User, Users } from "lucide-react";
import { useNavigate } from "react-router-dom";
import {
  ApiError,
  api,
  type AuditLogEntry,
  type TeamMember,
} from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import AppShell from "../components/AppShell";
import { PasswordStrengthMeter } from "../components/auth/PasswordStrengthMeter";
import { Avatar } from "../components/ui/Avatar";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ErrorNote } from "../components/ui/ErrorNote";
import { Input } from "../components/ui/Input";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Tabs, type TabItem } from "../components/ui/Tabs";
import { ThemePickerGrid } from "../components/ui/ThemePicker";
import { formatDateTime } from "../utils/format";

const AUDIT_ACTION_LABELS: Record<string, string> = {
  "member.added": "Membre ajouté",
  "dataset.deleted": "Dataset supprimé",
  "training_job.deleted": "Entraînement supprimé",
  "model.promoted": "Modèle promu",
  "clustering_job.deleted": "Clustering supprimé",
  "dimensionality_job.deleted": "Réduction de dimension supprimée",
  "anomaly_job.deleted": "Détection d'anomalies supprimée",
};

const UNSUPERVISED_JOB_DELETED_ACTIONS = new Set([
  "clustering_job.deleted",
  "dimensionality_job.deleted",
  "anomaly_job.deleted",
]);

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
  if (UNSUPERVISED_JOB_DELETED_ACTIONS.has(entry.action) && entry.details?.dataset_id) {
    return `${base} — dataset #${entry.details.dataset_id}`;
  }
  return base;
}

function RoleBadge({ role }: { role: "owner" | "member" }) {
  return <Badge variant={role === "owner" ? "accent" : "neutral"}>{role === "owner" ? "Propriétaire" : "Membre"}</Badge>;
}

type TabId = "profile" | "organization" | "preferences";

const TABS: TabItem<TabId>[] = [
  { id: "profile", label: "Profil", icon: User },
  { id: "organization", label: "Organisation & équipe", icon: Users },
  { id: "preferences", label: "Préférences", icon: Palette },
];

/** Profil personnel + administration d'organisation — jusqu'ici mélangés au
 * Dashboard (retour utilisateur : "le dashboard doit montrer l'activité, pas
 * l'admin"). `PATCH /auth/me` et `PATCH /auth/me/password` existaient déjà
 * côté API (Lot 1) mais n'étaient exposés dans aucune page — construits ici
 * pour la première fois. Le reste (membres, ajout de membre, journal
 * d'audit) est déplacé tel quel depuis Dashboard.tsx, pas réécrit. */
export default function Profile() {
  const { user } = useAuth();
  const [tab, setTab] = useState<TabId>("profile");

  if (!user) return null;

  return (
    <AppShell>
      <PageHeader
        eyebrow="Compte"
        title="Profil & Organisation"
        description="Vos informations personnelles, et la gestion de votre bureau d'études."
        icon={User}
        color="blue"
      />

      <div className="mb-5">
        <Tabs items={TABS} active={tab} onChange={setTab} />
      </div>

      {tab === "profile" ? <ProfileTab /> : tab === "organization" ? <OrganizationTab /> : <PreferencesTab />}
    </AppShell>
  );
}

function ProfileTab() {
  const { user, refreshUser } = useAuth();
  if (!user) return null;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="p-5">
        <div className="flex items-center gap-3 mb-5">
          <Avatar name={user.nom} />
          <div className="min-w-0">
            <p className="font-medium text-foreground truncate">{user.nom}</p>
            <p className="text-xs text-muted-foreground truncate">{user.email}</p>
          </div>
          <div className="ml-auto flex-shrink-0">
            <RoleBadge role={user.role} />
          </div>
        </div>
        <EditNameForm currentName={user.nom} onSaved={refreshUser} />
      </Card>

      <Card className="p-5">
        <SectionHeader icon={KeyRound} color="violet" label="Mot de passe" />
        <ChangePasswordForm />
      </Card>
    </div>
  );
}

function EditNameForm({ currentName, onSaved }: { currentName: string; onSaved: () => Promise<void> }) {
  const [nom, setNom] = useState(currentName);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSuccess(false);
    setIsSubmitting(true);
    try {
      await api.auth.updateMe({ nom });
      await onSaved();
      setSuccess(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de mettre à jour le profil");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div>
        <label htmlFor="profile-nom" className="block text-sm text-muted-foreground mb-1">
          Nom affiché
        </label>
        <Input id="profile-nom" type="text" required minLength={2} value={nom} onChange={(e) => setNom(e.target.value)} />
      </div>
      {error && <ErrorNote message={error} />}
      {success && <p className="text-sm text-success">Profil mis à jour.</p>}
      <Button type="submit" disabled={isSubmitting || nom === currentName}>
        {isSubmitting ? "Enregistrement…" : "Enregistrer"}
      </Button>
    </form>
  );
}

function ChangePasswordForm() {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    if (newPassword !== confirmPassword) {
      setError("La confirmation ne correspond pas au nouveau mot de passe.");
      return;
    }
    setIsSubmitting(true);
    try {
      await api.auth.changePassword({
        current_password: currentPassword,
        new_password: newPassword,
        new_password_confirm: confirmPassword,
      });
      // Phase 1B (AUDIT_BACKEND_2026-08-23.md) — le backend révoque TOUTES
      // les sessions au changement de mot de passe, y compris celle-ci :
      // le jeton actuel est déjà invalide côté serveur. On ne laisse pas
      // l'utilisateur découvrir ça au hasard sur le prochain appel API —
      // déconnexion explicite et reconnexion demandée immédiatement.
      await logout();
      navigate("/login?password_changed=1");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de changer le mot de passe");
      setIsSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div>
        <label htmlFor="current-password" className="block text-sm text-muted-foreground mb-1">
          Mot de passe actuel
        </label>
        <Input
          id="current-password"
          type="password"
          required
          autoComplete="current-password"
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="new-password" className="block text-sm text-muted-foreground mb-1">
          Nouveau mot de passe
        </label>
        <Input
          id="new-password"
          type="password"
          required
          minLength={8}
          autoComplete="new-password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
        />
        <PasswordStrengthMeter password={newPassword} />
      </div>
      <div>
        <label htmlFor="confirm-password" className="block text-sm text-muted-foreground mb-1">
          Confirmer le nouveau mot de passe
        </label>
        <Input
          id="confirm-password"
          type="password"
          required
          minLength={8}
          autoComplete="new-password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
        />
      </div>
      {error && <ErrorNote message={error} />}
      <p className="text-caption text-muted-foreground">
        Toutes vos sessions ouvertes seront fermées, y compris celle-ci — vous devrez vous reconnecter.
      </p>
      <Button type="submit" disabled={isSubmitting}>
        {isSubmitting ? "Enregistrement…" : "Changer le mot de passe"}
      </Button>
    </form>
  );
}

function OrganizationTab() {
  const { user } = useAuth();
  const [members, setMembers] = useState<TeamMember[] | null>(null);
  const [membersError, setMembersError] = useState<string | null>(null);

  const loadMembers = useCallback(async () => {
    try {
      setMembers(await api.team.members());
      setMembersError(null);
    } catch (err) {
      setMembersError(err instanceof ApiError ? err.message : "Impossible de charger l'équipe");
    }
  }, []);

  useEffect(() => {
    loadMembers();
  }, [loadMembers]);

  if (!user) return null;

  return (
    <div className="grid gap-6">
      <Card className="p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Users size={16} className="text-primary" />
            <h2 className="text-h3 text-foreground">Équipe — {user.organization_name}</h2>
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
        <Card className="p-5">
          <AddMemberForm onMemberAdded={loadMembers} />
        </Card>
      )}

      {user.role === "owner" && (
        <Card className="p-5">
          <AuditLogPanel />
        </Card>
      )}
    </div>
  );
}

/** Sélecteur de thème (Lot UI — refonte visuelle) — 5 directions, chacune
 * avec son propre aperçu réel, sa phrase de positionnement et son contraste
 * minimum mesuré (voir ThemePicker.tsx, contenu sourcé depuis
 * _design/themes.css). S'applique immédiatement (aucun rechargement),
 * persisté sur le profil serveur ET en localStorage (voir ThemeContext). */
function PreferencesTab() {
  return (
    <Card className="p-5 max-w-4xl">
      <SectionHeader icon={Palette} color="violet" label="Apparence" help="S'applique immédiatement, sur tous vos appareils." />
      <ThemePickerGrid />
    </Card>
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
      <h2 className="text-h3 text-foreground mb-4">Ajouter un membre à l'équipe</h2>
      <form onSubmit={handleSubmit} className="grid sm:grid-cols-3 gap-3 items-start">
        <Input type="text" placeholder="Nom" required minLength={2} value={nom} onChange={(e) => setNom(e.target.value)} />
        <Input type="email" placeholder="Email" required value={email} onChange={(e) => setEmail(e.target.value)} />
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

/** Journal d'audit (Lot 10, owner uniquement) — déplacé tel quel depuis
 * Dashboard.tsx : actions sensibles de l'équipe, pas un log applicatif
 * complet (déjà couvert côté serveur). */
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
        help="Actions sensibles de l'équipe — ajout de membre, suppression de dataset/entraînement/analyse, promotion de modèle. Visible uniquement par le propriétaire de l'organisation."
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
