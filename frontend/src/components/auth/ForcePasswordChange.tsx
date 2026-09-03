import { KeyRound } from "lucide-react";

import { useAuth } from "../../contexts/AuthContext";
import { Card } from "../ui/Card";
import { ChangePasswordForm } from "./ChangePasswordForm";

/** Écran imposé tant qu'un membre n'a pas remplacé le mot de passe
 * provisoire fixé par le propriétaire de son organisation.
 *
 * Ce n'est PAS la sécurité elle-même : le serveur refuse déjà tout appel
 * hors profil / changement de mot de passe / déconnexion (voir
 * `get_current_user`, code AUTH_MDP_PROVISOIRE). Cet écran évite que
 * l'utilisateur se heurte à une cascade de 403 sans comprendre — il
 * explique la situation et donne le seul geste qui débloque.
 *
 * Rendu depuis `ProtectedRoute`, point de passage unique de toutes les
 * pages authentifiées : c'est le pendant côté interface du point de
 * passage unique côté serveur, plutôt qu'une vérification à recopier dans
 * chaque page (qu'on finirait par oublier quelque part). */
export default function ForcePasswordChange() {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4 py-10">
      <Card className="w-full max-w-md p-6">
        <div className="flex items-center gap-2 mb-4">
          <KeyRound size={18} className="text-primary" aria-hidden="true" />
          <h1 className="text-h3 text-foreground">Choisissez votre mot de passe</h1>
        </div>

        <p className="text-sm text-muted-foreground mb-5">
          Votre compte a été créé avec un mot de passe provisoire, choisi par la personne qui vous a
          ajouté à l'organisation{user?.organization_name ? ` « ${user.organization_name} »` : ""} —
          elle le connaît donc. Choisissez le vôtre pour accéder à la plateforme ; personne d'autre
          ne le connaîtra.
        </p>

        <ChangePasswordForm />

        <p className="mt-4 text-xs text-muted-foreground border-t border-border pt-3">
          Le mot de passe actuel à saisir est celui qui vous a été communiqué. Vous serez ensuite
          invité à vous reconnecter : changer de mot de passe ferme toutes les sessions ouvertes, y
          compris celle-ci.
        </p>
      </Card>
    </div>
  );
}
