import { useCallback, useEffect, useState, type FormEvent } from "react";
import { KeyRound, Palette, ScrollText, User, Users } from "lucide-react";
import {
  ApiError,
  api,
  apiErrorReference,
  type AuditLogEntry,
  type TeamMember,
} from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import AppShell from "../components/AppShell";
import { ChangePasswordForm } from "../components/auth/ChangePasswordForm";
import { Avatar } from "../components/ui/Avatar";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ErrorNote } from "../components/ui/ErrorNote";
import { Input } from "../components/ui/Input";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Tabs, type TabItem } from "../components/ui/Tabs";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { ThemePickerGrid } from "../components/ui/ThemePicker";
import { formatDateTime } from "../utils/format";

const AUDIT_ACTION_LABELS: Record<string, string> = {
  "member.added": "Membre ajouté",
  "member.deactivated": "Accès d'un membre désactivé",
  "member.reactivated": "Accès d'un membre réactivé",
  "member.promoted": "Membre promu propriétaire",
  "member.demoted": "Propriétaire rétrogradé membre",
  "member.anonymized": "Données personnelles d'un membre effacées",
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
  const [errorRef, setErrorRef] = useState<string | undefined>(undefined);
  const [success, setSuccess] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setErrorRef(undefined);
    setSuccess(false);
    setIsSubmitting(true);
    try {
      await api.auth.updateMe({ nom });
      await onSaved();
      setSuccess(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de mettre à jour le profil");
      setErrorRef(apiErrorReference(err));
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
      {error && <ErrorNote message={error} reference={errorRef} />}
      {success && <p className="text-sm text-success">Profil mis à jour.</p>}
      <Button type="submit" disabled={isSubmitting || nom === currentName}>
        {isSubmitting ? "Enregistrement…" : "Enregistrer"}
      </Button>
    </form>
  );
}

function OrganizationTab() {
  const { user, refreshUser } = useAuth();
  const [members, setMembers] = useState<TeamMember[] | null>(null);
  const [membersError, setMembersError] = useState<string | null>(null);
  const [membersErrorRef, setMembersErrorRef] = useState<string | undefined>(undefined);
  // Désactivation/réactivation d'un membre (départ d'un collaborateur).
  // Confirmation à deux clics — même motif que les suppressions ailleurs
  // dans l'app, plutôt qu'une modale dédiée pour une action réversible.
  const confirmToggle = useConfirmAction<number>();
  const [togglingId, setTogglingId] = useState<number | null>(null);
  // Succession : promouvoir un membre propriétaire, ou se rétrograder une
  // fois son successeur en place. Confirmation à deux clics également —
  // céder ses droits n'est pas anodin.
  const confirmRole = useConfirmAction<number>();
  const [roleChangingId, setRoleChangingId] = useState<number | null>(null);
  // Anonymisation : effacement DÉFINITIF des données personnelles. Seule
  // action irréversible de cette page — d'où une confirmation dont le
  // libellé le dit explicitement, plutôt qu'un « Confirmer ? » anodin.
  const confirmAnonymize = useConfirmAction<number>();
  const [anonymizingId, setAnonymizingId] = useState<number | null>(null);

  const loadMembers = useCallback(async () => {
    try {
      setMembers(await api.team.members());
      setMembersError(null);
      setMembersErrorRef(undefined);
    } catch (err) {
      setMembersError(err instanceof ApiError ? err.message : "Impossible de charger l'équipe");
      setMembersErrorRef(apiErrorReference(err));
    }
  }, []);

  async function toggleMemberAccess(member: TeamMember) {
    setTogglingId(member.id);
    try {
      await api.team.setMemberActive(member.id, !member.actif);
      setMembersError(null);
      setMembersErrorRef(undefined);
      await loadMembers();
    } catch (err) {
      setMembersError(
        err instanceof ApiError ? err.message : "Impossible de modifier l'accès de ce membre",
      );
      setMembersErrorRef(apiErrorReference(err));
    } finally {
      setTogglingId(null);
    }
  }

  async function changeMemberRole(member: TeamMember) {
    const nextRole = member.role === "owner" ? "member" : "owner";
    setRoleChangingId(member.id);
    try {
      await api.team.setMemberRole(member.id, nextRole);
      setMembersError(null);
      setMembersErrorRef(undefined);
      await loadMembers();
      // Se rétrograder soi-même retire ses propres droits : le contexte
      // d'authentification porte encore l'ancien rôle, il faut le relire
      // sinon l'interface continuerait d'afficher des actions désormais
      // refusées par le serveur.
      if (member.id === user?.id) await refreshUser();
    } catch (err) {
      setMembersError(err instanceof ApiError ? err.message : "Impossible de modifier ce rôle");
      setMembersErrorRef(apiErrorReference(err));
    } finally {
      setRoleChangingId(null);
    }
  }

  async function anonymizeMember(member: TeamMember) {
    setAnonymizingId(member.id);
    try {
      await api.team.anonymizeMember(member.id);
      setMembersError(null);
      setMembersErrorRef(undefined);
      await loadMembers();
    } catch (err) {
      setMembersError(err instanceof ApiError ? err.message : "Impossible d'anonymiser ce compte");
      setMembersErrorRef(apiErrorReference(err));
    } finally {
      setAnonymizingId(null);
    }
  }

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

        {membersError && <ErrorNote message={membersError} reference={membersErrorRef} />}

        {members === null && !membersError ? (
          <p className="text-sm text-muted-foreground">Chargement…</p>
        ) : (
          <ul className="divide-y divide-border">
            {members?.map((member) => (
              <li key={member.id} className="py-2.5 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <Avatar name={member.nom} size="sm" />
                  <div className="min-w-0">
                    <p className={`text-sm truncate ${member.actif ? "text-foreground/90" : "text-muted-foreground line-through"}`}>
                      {member.nom}
                    </p>
                    <p className="text-xs text-muted-foreground truncate">{member.email}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {member.must_change_password && member.actif && (
                    <Badge variant="warning">Mot de passe provisoire</Badge>
                  )}
                  {member.anonymized_at ? (
                    <Badge variant="neutral">Identité effacée le {formatDateTime(member.anonymized_at)}</Badge>
                  ) : (
                    !member.actif && (
                      <Badge variant="danger">
                        {member.deactivated_at
                          ? `Accès révoqué le ${formatDateTime(member.deactivated_at)}`
                          : "Accès révoqué"}
                      </Badge>
                    )
                  )}
                  <RoleBadge role={member.role} />
                  {/* Le propriétaire ne peut pas se désactiver lui-même : son
                      organisation n'aurait plus personne pour gérer l'équipe
                      (l'API le refuse aussi, ceci n'est que le reflet UI). */}
                  {/* Changement de rôle : proposé AUSSI sur sa propre ligne —
                      se rétrograder après avoir promu son successeur est
                      précisément le scénario du départ. Le serveur refuse de
                      rétrograder le dernier propriétaire actif. */}
                  {user.role === "owner" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      loading={roleChangingId === member.id}
                      onClick={() => confirmRole.trigger(member.id, () => changeMemberRole(member))}
                      onMouseLeave={confirmRole.reset}
                    >
                      {confirmRole.isPending(member.id)
                        ? member.role === "owner"
                          ? member.id === user.id
                            ? "Confirmer — vous perdrez vos droits ?"
                            : "Confirmer la rétrogradation ?"
                          : "Confirmer la promotion ?"
                        : member.role === "owner"
                          ? "Rétrograder"
                          : "Promouvoir propriétaire"}
                    </Button>
                  )}
                  {user.role === "owner" && member.id !== user.id && !member.anonymized_at && (
                    <Button
                      variant={member.actif ? "destructive" : "secondary"}
                      size="sm"
                      loading={togglingId === member.id}
                      onClick={() => confirmToggle.trigger(member.id, () => toggleMemberAccess(member))}
                      onMouseLeave={confirmToggle.reset}
                    >
                      {confirmToggle.isPending(member.id)
                        ? member.actif
                          ? "Confirmer la révocation ?"
                          : "Confirmer la réactivation ?"
                        : member.actif
                          ? "Révoquer l'accès"
                          : "Réactiver"}
                    </Button>
                  )}
                  {/* Effacement définitif — proposé UNIQUEMENT sur un compte
                      déjà révoqué et pas encore anonymisé, ce que l'API
                      impose aussi de son côté. */}
                  {user.role === "owner" && member.id !== user.id && !member.actif && !member.anonymized_at && (
                    <Button
                      variant="destructive"
                      size="sm"
                      loading={anonymizingId === member.id}
                      onClick={() => confirmAnonymize.trigger(member.id, () => anonymizeMember(member))}
                      onMouseLeave={confirmAnonymize.reset}
                    >
                      {confirmAnonymize.isPending(member.id)
                        ? "Effacer définitivement — irréversible ?"
                        : "Effacer les données personnelles"}
                    </Button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}

        {user.role === "owner" && (
          <p className="mt-4 text-xs text-muted-foreground border-t border-border pt-3">
            Révoquer l'accès d'un membre coupe sa connexion <strong>immédiatement</strong>, y compris
            les sessions déjà ouvertes. Ce n'est pas une suppression : son compte, ses datasets, ses
            entraînements et sa trace dans le journal d'audit sont conservés, et l'accès peut être
            rétabli à tout moment.
            <br />
            Avant de quitter l'organisation, <strong>promouvez votre successeur propriétaire</strong>,
            puis rétrogradez-vous. Une organisation conserve toujours au moins un propriétaire actif :
            rétrograder le dernier est refusé, sans quoi plus personne ne pourrait gérer l'équipe.
            <br />
            « Effacer les données personnelles » (disponible une fois l'accès révoqué) supprime
            définitivement l'e-mail et le nom, y compris dans le journal d'audit.{" "}
            <strong>C'est irréversible.</strong> Ses datasets, entraînements et l'historique de ses
            actions restent : ils appartiennent à l'organisation, pas à la personne.
          </p>
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
  // Invitation par défaut : c'est le mode sûr — aucun mot de passe n'est
  // choisi ni connu par le propriétaire. Le repli « mot de passe
  // provisoire » n'est proposé qu'explicitement, pour les déploiements sans
  // service d'e-mail (l'API refuse alors l'invitation).
  const [useTemporaryPassword, setUseTemporaryPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorRef, setErrorRef] = useState<string | undefined>(undefined);
  const [success, setSuccess] = useState<"invited" | "created" | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setErrorRef(undefined);
    setSuccess(null);
    setIsSubmitting(true);
    try {
      await api.team.addMember(
        useTemporaryPassword ? { email, nom, password } : { email, nom },
      );
      setEmail("");
      setNom("");
      setPassword("");
      setSuccess(useTemporaryPassword ? "created" : "invited");
      onMemberAdded();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible d'ajouter ce membre");
      setErrorRef(apiErrorReference(err));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <>
      <h2 className="text-h3 text-foreground mb-4">Ajouter un membre à l'équipe</h2>
      <p className="text-sm text-muted-foreground mb-3">
        Le membre reçoit un e-mail avec un lien personnel et choisit lui-même son mot de passe.
        Vous ne le connaîtrez jamais — c'est voulu : personne ne doit pouvoir se connecter à sa place.
      </p>
      <form onSubmit={handleSubmit} className="grid sm:grid-cols-2 gap-3 items-start">
        <Input type="text" placeholder="Nom" required minLength={2} value={nom} onChange={(e) => setNom(e.target.value)} />
        <Input type="email" placeholder="Email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        {useTemporaryPassword && (
          <Input
            type="password"
            placeholder="Mot de passe temporaire"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="sm:col-span-2"
          />
        )}
        <Button type="submit" disabled={isSubmitting} className="sm:col-span-2">
          {isSubmitting ? "Ajout…" : useTemporaryPassword ? "Ajouter" : "Envoyer l'invitation"}
        </Button>
      </form>

      <button
        type="button"
        onClick={() => setUseTemporaryPassword((v) => !v)}
        className="mt-2 text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
      >
        {useTemporaryPassword
          ? "Revenir à l'invitation par e-mail"
          : "Pas de service d'e-mail ? Définir un mot de passe temporaire"}
      </button>
      {useTemporaryPassword && (
        <p className="mt-2 text-xs text-muted-foreground">
          Vous devrez transmettre ce mot de passe au membre, et vous le connaîtrez donc. Il sera
          contraint de le remplacer à sa première connexion.
        </p>
      )}

      {error && <ErrorNote message={error} reference={errorRef} />}
      {success === "invited" && (
        <p className="text-sm text-success mt-2">Invitation envoyée — le membre apparaîtra comme en attente jusqu'à ce qu'il choisisse son mot de passe.</p>
      )}
      {success === "created" && (
        <p className="text-sm text-success mt-2">Membre ajouté — transmettez-lui son mot de passe temporaire.</p>
      )}
    </>
  );
}

/** Journal d'audit (Lot 10, owner uniquement) — déplacé tel quel depuis
 * Dashboard.tsx : actions sensibles de l'équipe, pas un log applicatif
 * complet (déjà couvert côté serveur). */
function AuditLogPanel() {
  const [entries, setEntries] = useState<AuditLogEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorRef, setErrorRef] = useState<string | undefined>(undefined);

  useEffect(() => {
    api.team
      .auditLog()
      .then(setEntries)
      .catch((err) => {
        setError(err instanceof ApiError ? err.message : "Journal indisponible");
        setErrorRef(apiErrorReference(err));
      });
  }, []);

  return (
    <>
      <SectionHeader
        icon={ScrollText}
        color="amber"
        label="Journal d'audit"
        help="Actions sensibles de l'équipe — ajout de membre, suppression de dataset/entraînement/analyse, promotion de modèle. Visible uniquement par le propriétaire de l'organisation."
      />
      {error && <ErrorNote message={error} reference={errorRef} />}
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
