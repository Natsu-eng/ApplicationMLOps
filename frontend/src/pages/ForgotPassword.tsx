import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { AuthBrandPanel } from "../components/auth/AuthBrandPanel";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";

// Phase 1B (AUDIT_BACKEND_2026-08-23.md) — le backend répond 204 de façon
// STRICTEMENT identique que l'adresse corresponde à un compte ou non (y
// compris si la limite de débit est atteinte). Cet écran ne doit donc
// JAMAIS afficher un message différent selon un signal quelconque de la
// réponse — un seul état "envoyé", toujours le même texte, quoi qu'il
// arrive côté serveur (sauf une vraie erreur réseau/technique).
export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await api.auth.requestPasswordReset(email);
      setSubmitted(true);
    } catch {
      // Erreur réseau/technique uniquement — jamais "adresse inconnue",
      // qui annulerait toute la précaution côté serveur.
      setError("Impossible d'envoyer la demande pour le moment. Réessayez dans quelques instants.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      <AuthBrandPanel
        kicker="Explore · Analyse · Prédit"
        heading="Mot de passe oublié"
        tagline="Indiquez votre adresse professionnelle, nous vous envoyons un lien pour en choisir un nouveau."
      />

      <div className="flex-1 flex items-center justify-center p-6 sm:p-10 bg-muted">
        <div className="max-w-sm w-full">
          <div className="mb-6">
            <h1 className="text-h2 text-foreground">Mot de passe oublié ?</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Nous vous enverrons un lien de réinitialisation par e-mail.
            </p>
          </div>

          <Card className="p-6">
            {submitted ? (
              <p className="text-sm text-foreground" role="status">
                Si un compte existe pour cette adresse, un lien de réinitialisation vient d'être envoyé.
                Vérifiez votre boîte de réception (et vos indésirables) dans les prochaines minutes.
              </p>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
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

                {error && (
                  <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
                    {error}
                  </p>
                )}

                <Button type="submit" disabled={isSubmitting} className="w-full">
                  {isSubmitting ? "Envoi…" : "Envoyer le lien de réinitialisation"}
                </Button>
              </form>
            )}
          </Card>

          <p className="text-center text-sm text-muted-foreground mt-4">
            <Link to="/login" className="text-primary hover:text-primary/80 font-medium">
              Retour à la connexion
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
