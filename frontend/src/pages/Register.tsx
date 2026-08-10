import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError } from "../api/client";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { useAuth } from "../contexts/AuthContext";

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [organizationName, setOrganizationName] = useState("");
  const [nom, setNom] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await register({ organization_name: organizationName, nom, email, password });
      navigate("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Inscription impossible");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen text-slate-100 flex items-center justify-center px-6 py-12">
      <div className="max-w-sm w-full">
        <div className="text-center mb-8">
          <div className="h-10 w-10 mx-auto mb-4 rounded-xl bg-brand-gradient flex items-center justify-center text-white font-bold">
            D
          </div>
          <p className="text-xs uppercase tracking-widest text-teal-400 font-semibold mb-2">
            DataLab Pro
          </p>
          <h1 className="text-2xl font-serif">Créer votre bureau d'études</h1>
          <p className="text-sm text-slate-500 mt-2">
            Vous devenez propriétaire de l'organisation — vous pourrez ensuite y inviter votre équipe.
          </p>
        </div>

        <Card className="p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="organization_name" className="block text-sm text-slate-400 mb-1">
                Nom du bureau d'études
              </label>
              <Input
                id="organization_name"
                type="text"
                required
                minLength={2}
                value={organizationName}
                onChange={(e) => setOrganizationName(e.target.value)}
              />
            </div>

            <div>
              <label htmlFor="nom" className="block text-sm text-slate-400 mb-1">
                Votre nom
              </label>
              <Input
                id="nom"
                type="text"
                required
                minLength={2}
                value={nom}
                onChange={(e) => setNom(e.target.value)}
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm text-slate-400 mb-1">
                Email
              </label>
              <Input
                id="email"
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm text-slate-400 mb-1">
                Mot de passe
              </label>
              <Input
                id="password"
                type="password"
                required
                minLength={8}
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <p className="text-xs text-slate-600 mt-1">8 caractères minimum</p>
            </div>

            {error && (
              <p className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <Button type="submit" disabled={isSubmitting} className="w-full">
              {isSubmitting ? "Création…" : "Créer mon organisation"}
            </Button>
          </form>
        </Card>

        <p className="text-center text-sm text-slate-500 mt-4">
          Déjà un compte ?{" "}
          <Link to="/login" className="text-teal-400 hover:text-teal-300">
            Se connecter
          </Link>
        </p>
      </div>
    </main>
  );
}
