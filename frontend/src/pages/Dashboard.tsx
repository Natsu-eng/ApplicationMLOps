import { useCallback, useEffect, useState, type FormEvent } from "react";
import { ApiError, api, type TeamMember } from "../api/client";
import { useAuth } from "../contexts/AuthContext";

/** Page protégée du Lot 1 : profil, équipe de l'organisation, ajout de membre (owner). */
export default function Dashboard() {
  const { user, logout } = useAuth();

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

  if (!user) return null; // ProtectedRoute garantit déjà un utilisateur connecté

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-widest text-teal-400 font-semibold">
            DataLab Pro
          </p>
          <h1 className="text-lg font-serif">{user.organization_name}</h1>
        </div>
        <button
          onClick={logout}
          className="text-sm text-slate-400 hover:text-slate-200 border border-slate-800 hover:border-slate-700 rounded-md px-3 py-1.5 transition-colors"
        >
          Se déconnecter
        </button>
      </header>

      <div className="max-w-3xl mx-auto px-6 py-10 space-y-8">
        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-sm uppercase tracking-wide text-slate-500 mb-3">Mon compte</h2>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">{user.nom}</p>
              <p className="text-sm text-slate-500">{user.email}</p>
            </div>
            <RoleBadge role={user.role} />
          </div>
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-sm uppercase tracking-wide text-slate-500 mb-3">
            Équipe — {user.organization_name}
          </h2>

          {membersError && (
            <p className="text-sm text-rose-400 bg-rose-950/40 border border-rose-900 rounded-md px-3 py-2 mb-3">
              {membersError}
            </p>
          )}

          {members === null && !membersError ? (
            <p className="text-sm text-slate-500">Chargement…</p>
          ) : (
            <ul className="divide-y divide-slate-800">
              {members?.map((member) => (
                <li key={member.id} className="py-2.5 flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{member.nom}</p>
                    <p className="text-xs text-slate-500">{member.email}</p>
                  </div>
                  <RoleBadge role={member.role} />
                </li>
              ))}
            </ul>
          )}
        </section>

        {user.role === "owner" && (
          <AddMemberForm onMemberAdded={loadMembers} />
        )}
      </div>
    </main>
  );
}

function RoleBadge({ role }: { role: "owner" | "member" }) {
  const isOwner = role === "owner";
  return (
    <span
      className={`text-xs font-medium px-2 py-0.5 rounded-full ${
        isOwner
          ? "bg-teal-500/15 text-teal-300 border border-teal-500/30"
          : "bg-slate-800 text-slate-400 border border-slate-700"
      }`}
    >
      {isOwner ? "Propriétaire" : "Membre"}
    </span>
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
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <h2 className="text-sm uppercase tracking-wide text-slate-500 mb-3">
        Ajouter un membre à l'équipe
      </h2>
      <form onSubmit={handleSubmit} className="grid sm:grid-cols-3 gap-3 items-start">
        <input
          type="text"
          placeholder="Nom"
          required
          minLength={2}
          value={nom}
          onChange={(e) => setNom(e.target.value)}
          className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
        />
        <input
          type="email"
          placeholder="Email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
        />
        <input
          type="password"
          placeholder="Mot de passe temporaire"
          required
          minLength={8}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
        />
        <button
          type="submit"
          disabled={isSubmitting}
          className="sm:col-span-3 rounded-md bg-teal-500 hover:bg-teal-400 disabled:opacity-50 disabled:cursor-not-allowed text-slate-950 font-medium py-2 text-sm transition-colors"
        >
          {isSubmitting ? "Ajout…" : "Ajouter"}
        </button>
      </form>
      {error && <p className="text-sm text-rose-400 mt-2">{error}</p>}
      {success && <p className="text-sm text-emerald-400 mt-2">Membre ajouté.</p>}
    </section>
  );
}
