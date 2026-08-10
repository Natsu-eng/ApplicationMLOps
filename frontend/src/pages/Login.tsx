import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError } from "../api/client";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { useAuth } from "../contexts/AuthContext";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await login(email, password);
      navigate("/");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Connexion impossible");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen text-slate-100 flex items-center justify-center px-6">
      <div className="max-w-sm w-full">
        <div className="text-center mb-8">
          <div className="h-10 w-10 mx-auto mb-4 rounded-xl bg-brand-gradient flex items-center justify-center text-white font-bold">
            D
          </div>
          <p className="text-xs uppercase tracking-widest text-teal-400 font-semibold mb-2">
            DataLab Pro
          </p>
          <h1 className="text-2xl font-serif">Connexion</h1>
        </div>

        <Card className="p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
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
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            {error && (
              <p className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <Button type="submit" disabled={isSubmitting} className="w-full">
              {isSubmitting ? "Connexion…" : "Se connecter"}
            </Button>
          </form>
        </Card>

        <p className="text-center text-sm text-slate-500 mt-4">
          Pas encore de compte ?{" "}
          <Link to="/register" className="text-teal-400 hover:text-teal-300">
            Créer mon bureau d'études
          </Link>
        </p>
      </div>
    </main>
  );
}
