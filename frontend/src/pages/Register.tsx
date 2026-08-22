import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Check, ShieldCheck, Waypoints } from "lucide-react";
import { ApiError } from "../api/client";
import { AuthBrandPanel, type AuthBrandFeature } from "../components/auth/AuthBrandPanel";
import { PasswordStrengthMeter } from "../components/auth/PasswordStrengthMeter";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { useAuth } from "../contexts/AuthContext";

// Contenu repris tel quel de _design/apercu/Inscription.html — les 3
// promesses concrètes de la plateforme, pas du texte de remplissage.
const REGISTER_FEATURES: AuthBrandFeature[] = [
  {
    icon: Check,
    title: "Un verdict, pas un rapport",
    description:
      "« Utilisable en production, avec une réserve au-delà de 70 MPa » — et l'explication en dessous.",
  },
  {
    icon: ShieldCheck,
    title: "Les pièges détectés avant vous",
    description:
      "Fuites de données, découpes trompeuses, colonnes trop parfaites. C'est ce qui fait échouer neuf projets sur dix.",
  },
  {
    icon: Waypoints,
    title: "Chaque chiffre est traçable",
    description:
      "Empreinte des données, graine, versions de bibliothèques. Six mois plus tard, vous saurez d'où vient une prédiction.",
  },
];

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
      navigate("/onboarding");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Inscription impossible");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      <AuthBrandPanel
        kicker="Essai gratuit · 14 jours"
        heading="Vos données savent des choses. Faites-les parler — sans écrire une ligne de code."
        tagline="Chargez un fichier, dites ce que vous cherchez, laissez la plateforme entraîner et comparer les modèles. Vous recevez un verdict en français, pas un tableau de métriques."
        features={REGISTER_FEATURES}
      />

      <div className="flex-1 flex items-center justify-center p-6 sm:p-10 bg-muted">
        <div className="max-w-sm w-full">
          <div className="mb-6">
            <h1 className="text-h2 text-foreground">Créer votre bureau d'études</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Vous devenez propriétaire de l'organisation — vous pourrez ensuite y inviter votre équipe.
            </p>
          </div>

          <Card className="p-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label htmlFor="organization_name" className="block text-sm text-muted-foreground mb-1">
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
                <label htmlFor="nom" className="block text-sm text-muted-foreground mb-1">
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
                <label htmlFor="password" className="block text-sm text-muted-foreground mb-1">
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
                <PasswordStrengthMeter password={password} />
              </div>

              {error && (
                <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
                  {error}
                </p>
              )}

              <Button type="submit" disabled={isSubmitting} className="w-full">
                {isSubmitting ? "Création…" : "Créer mon organisation"}
              </Button>
            </form>
          </Card>

          <p className="text-center text-sm text-muted-foreground mt-4">
            Déjà un compte ?{" "}
            <Link to="/login" className="text-primary hover:text-primary/80 font-medium">
              Se connecter
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
