import { Component, type ErrorInfo, type ReactNode } from "react";
import { OctagonAlert } from "lucide-react";
import { Button } from "./ui/Button";
import { Card } from "./ui/Card";

/** Filet de sécurité final contre un plantage au rendu (Lot
 * robustesse/dashboard-dynamique) — avant ce composant, AUCUN
 * `ErrorBoundary` n'existait dans toute l'application (vérifié par
 * recherche : zéro occurrence de `componentDidCatch`/
 * `getDerivedStateFromError`) : une exception non gérée pendant le rendu
 * d'une page (ex. une page qui suppose une forme de réponse API que le
 * backend viole après une régression) démontait l'app entière vers un
 * écran blanc silencieux, sans aucune indication pour l'utilisateur.
 *
 * SEUL composant classe de tout le dépôt (confirmé : le reste est 100 %
 * fonctionnel/hooks) — pas un choix de style, une nécessité structurelle :
 * React n'expose `componentDidCatch`/`getDerivedStateFromError` que sur
 * les composants classe, aucun hook équivalent n'existe (React 19 compris).
 *
 * Recharge complète de la page en réponse (pas une tentative de
 * "réessayer" en mémoire) — l'état React qui a mené au plantage est
 * inconnu et potentiellement incohérent ; redémarrer proprement depuis
 * zéro est plus sûr qu'un retour à un état qu'on ne peut pas garantir
 * sain. */
interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Seul filet de diagnostic disponible ici (pas de `request_id` côté
    // rendu React, contrairement aux erreurs API — voir apiErrorReference).
    console.error("[ErrorBoundary] Rendu interrompu :", error, info.componentStack);
  }

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div className="flex items-center justify-center min-h-screen p-6">
        <Card className="max-w-md p-6 text-center">
          <OctagonAlert size={32} className="mx-auto mb-3 text-destructive" aria-hidden="true" />
          <h1 className="text-h3 text-foreground mb-2">Un problème est survenu</h1>
          <p className="text-sm text-muted-foreground mb-5">
            Cette page a rencontré une erreur inattendue. Rechargez la page — si le problème persiste, contactez le
            support.
          </p>
          <Button onClick={() => window.location.reload()}>Recharger la page</Button>
        </Card>
      </div>
    );
  }
}
