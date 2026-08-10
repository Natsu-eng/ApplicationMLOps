import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Users } from "lucide-react";
import { ApiError, api, type TeamMember } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import AppShell from "../components/AppShell";
import { Avatar } from "../components/ui/Avatar";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";

/** Page protégée du Lot 1 : profil, équipe de l'organisation, ajout de membre (owner). */
export default function Dashboard() {
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
    <AppShell pillarId="supervised">
      <div className="mb-8">
        <p className="text-xs uppercase tracking-widest text-teal-600 font-semibold mb-1">
          Vue d'ensemble
        </p>
        <h1 className="text-2xl font-serif text-slate-900">
          Bonjour, {user.nom.split(" ")[0]}
        </h1>
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
              <Users size={16} className="text-teal-600" />
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
      </div>
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
