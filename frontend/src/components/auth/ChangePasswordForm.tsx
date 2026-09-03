import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError, api, apiErrorReference } from "../../api/client";
import { useAuth } from "../../contexts/AuthContext";
import { Button } from "../ui/Button";
import { ErrorNote } from "../ui/ErrorNote";
import { Input } from "../ui/Input";
import { PasswordStrengthMeter } from "./PasswordStrengthMeter";

/** Formulaire de changement de mot de passe du compte connecté.
 *
 * Extrait de `pages/Profile.tsx` pour être réutilisé tel quel par l'écran
 * de changement OBLIGATOIRE (`ForcePasswordChange`) : un membre qui vient
 * d'être ajouté doit remplacer le mot de passe provisoire choisi par son
 * propriétaire, et il n'y avait aucune raison d'en dupliquer la logique —
 * mêmes règles de validation, même déconnexion forcée ensuite. */
export function ChangePasswordForm() {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [errorRef, setErrorRef] = useState<string | undefined>(undefined);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setErrorRef(undefined);
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
      setErrorRef(apiErrorReference(err));
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
      {error && <ErrorNote message={error} reference={errorRef} />}
      <p className="text-caption text-muted-foreground">
        Toutes vos sessions ouvertes seront fermées, y compris celle-ci — vous devrez vous reconnecter.
      </p>
      <Button type="submit" disabled={isSubmitting}>
        {isSubmitting ? "Enregistrement…" : "Changer le mot de passe"}
      </Button>
    </form>
  );
}

