import { useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { AuthBrandPanel } from "../components/auth/AuthBrandPanel";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { useAuth } from "../contexts/AuthContext";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  // Lot 0.3 (correctif C5, AUDIT_DATALAB_2026-08-16.md) — posé par
  // handleUnauthorized() (api/client.ts) lors d'une redirection sur 401,
  // jamais par un lien direct construit par l'utilisateur.
  const sessionExpired = searchParams.get("expired") === "1";
  // Phase 1B — posé par Profile.tsx::ChangePasswordForm après un
  // changement de mot de passe réussi (le backend révoque alors toutes les
  // sessions, y compris la session courante) : message positif, pas un
  // avertissement — ce n'est pas une erreur.
  const passwordChanged = searchParams.get("password_changed") === "1";

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
    <div className="min-h-screen flex flex-col lg:flex-row">
      <AuthBrandPanel
        kicker="Explore · Analyse · Prédit"
        heading="Bon retour sur DataLab Pro"
        tagline="Reconnectez-vous pour reprendre l'exploration de vos données et l'entraînement de vos modèles."
      />

      <div className="flex-1 flex items-center justify-center p-6 sm:p-10 bg-muted">
        <div className="max-w-sm w-full">
          <div className="mb-6">
            <h1 className="text-h2 text-foreground">Connexion</h1>
            <p className="text-sm text-muted-foreground mt-1">Accédez à votre bureau d'études.</p>
          </div>

          <Card className="p-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              {sessionExpired && (
                <p className="text-sm text-warning bg-warning/10 border border-warning/20 rounded-lg px-3 py-2">
                  Votre session a expiré, reconnectez-vous.
                </p>
              )}

              {passwordChanged && (
                <p className="text-sm text-success bg-success/10 border border-success/20 rounded-lg px-3 py-2">
                  Mot de passe modifié — reconnectez-vous avec votre nouveau mot de passe.
                </p>
              )}

              <div>
                <label htmlFor="email" className="block text-sm text-muted-foreground mb-1">
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
                <div className="flex items-center justify-between mb-1">
                  <label htmlFor="password" className="block text-sm text-muted-foreground">
                    Mot de passe
                  </label>
                  <Link to="/forgot-password" className="text-sm text-primary hover:text-primary/80 font-medium">
                    Mot de passe oublié ?
                  </Link>
                </div>
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
                <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
                  {error}
                </p>
              )}

              <Button type="submit" disabled={isSubmitting} className="w-full">
                {isSubmitting ? "Connexion…" : "Se connecter"}
              </Button>
            </form>
          </Card>

          <p className="text-center text-sm text-muted-foreground mt-4">
            Pas encore de compte ?{" "}
            <Link to="/register" className="text-primary hover:text-primary/80 font-medium">
              Créer mon bureau d'études
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
