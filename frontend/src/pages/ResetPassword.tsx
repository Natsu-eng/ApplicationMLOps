import { useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ApiError, api } from "../api/client";
import { AuthBrandPanel } from "../components/auth/AuthBrandPanel";
import { PasswordStrengthMeter } from "../components/auth/PasswordStrengthMeter";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";

export default function ResetPassword() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token") ?? "";

  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [done, setDone] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await api.auth.confirmPasswordReset({
        token,
        new_password: newPassword,
        new_password_confirm: confirmPassword,
      });
      setDone(true);
    } catch (err) {
      // AUTH_RESET_TOKEN_INVALIDE / AUTH_MDP_TROP_FAIBLE : le backend
      // renvoie déjà un message français actionnable — jamais reformulé
      // ici, sauf en cas d'erreur réseau/technique (pas d'ApiError).
      setError(err instanceof ApiError ? err.message : "Impossible de réinitialiser le mot de passe pour le moment.");
    } finally {
      setIsSubmitting(false);
    }
  }

  const linkLooksExpired = error?.toLowerCase().includes("invalide") || error?.toLowerCase().includes("expiré");

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      <AuthBrandPanel
        kicker="Explore · Analyse · Prédit"
        heading="Choisissez un nouveau mot de passe"
        tagline="Ce lien est personnel et à usage unique."
      />

      <div className="flex-1 flex items-center justify-center p-6 sm:p-10 bg-muted">
        <div className="max-w-sm w-full">
          <div className="mb-6">
            <h1 className="text-h2 text-foreground">Nouveau mot de passe</h1>
          </div>

          <Card className="p-6">
            {!token ? (
              <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
                Lien de réinitialisation incomplet — redemandez un lien depuis l'écran précédent.
              </p>
            ) : done ? (
              <div className="space-y-4">
                <p className="text-sm text-success" role="status">
                  Votre mot de passe a été modifié. Vous pouvez maintenant vous connecter avec.
                </p>
                <Link to="/login">
                  <Button className="w-full">Aller à la connexion</Button>
                </Link>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="new_password" className="block text-sm text-muted-foreground mb-1">
                    Nouveau mot de passe
                  </label>
                  <Input
                    id="new_password"
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
                  <label htmlFor="confirm_password" className="block text-sm text-muted-foreground mb-1">
                    Confirmer le mot de passe
                  </label>
                  <Input
                    id="confirm_password"
                    type="password"
                    required
                    minLength={8}
                    autoComplete="new-password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                  />
                </div>

                {error && (
                  <div className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
                    <p>{error}</p>
                    {linkLooksExpired && (
                      <Link to="/forgot-password" className="underline font-medium">
                        Redemander un lien
                      </Link>
                    )}
                  </div>
                )}

                <Button type="submit" disabled={isSubmitting} className="w-full">
                  {isSubmitting ? "Validation…" : "Réinitialiser le mot de passe"}
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
