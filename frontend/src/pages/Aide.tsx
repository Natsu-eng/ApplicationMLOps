import { useMemo, useState, type FormEvent } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  AlertTriangle,
  BrainCircuit,
  ChartColumn,
  CheckCircle2,
  ChevronDown,
  Database,
  HelpCircle,
  History,
  Images,
  LineChart,
  ScatterChart,
  Search,
  Send,
  Shapes,
  ShieldCheck,
  UserCog,
} from "lucide-react";
import { ApiError, api } from "../api/client";
import AppShell from "../components/AppShell";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { PageHeader } from "../components/ui/PageHeader";
import { ColorIconBadge, type AccentColor } from "../components/ui/ColorIconBadge";
import { Tabs } from "../components/ui/Tabs";

interface GuideStep {
  icon: typeof Database;
  color: AccentColor;
  title: string;
  description: string;
}

// Contenu repris tel quel de l'ancien HelpModal (Lot 16, déjà exact et
// vérifié — jamais réécrit à la légère) : la modale disparaît au profit
// de cette page dédiée, seule source désormais (voir AppShell.tsx).
const SUPERVISED_STEPS: GuideStep[] = [
  {
    icon: Database,
    color: "blue",
    title: "1. Importez vos données",
    description:
      "Déposez un fichier CSV, Parquet, Excel ou JSON depuis « Mes données ». Explorez-le (statistiques, corrélations, valeurs manquantes) avant même de lancer un entraînement.",
  },
  {
    icon: BrainCircuit,
    color: "teal",
    title: "2. Entraînez un modèle",
    description:
      "Choisissez un dataset et la colonne à prédire. DataLab compare automatiquement plusieurs algorithmes et retient le meilleur — aucune expertise technique requise en mode guidé.",
  },
  {
    icon: LineChart,
    color: "violet",
    title: "3. Comprenez le résultat",
    description:
      "Chaque résultat s'accompagne d'un Verdict en langage clair : pourquoi ce modèle a gagné, quelles variables comptent le plus, à quel point on peut lui faire confiance.",
  },
  {
    icon: ChartColumn,
    color: "amber",
    title: "4. Testez une prédiction",
    description:
      "Saisissez un nouveau cas dans le formulaire généré automatiquement et obtenez une prédiction immédiate, avec sa fourchette de confiance ou ses probabilités par classe.",
  },
  {
    icon: History,
    color: "rose",
    title: "5. Suivez vos expériences",
    description:
      "L'historique garde chaque entraînement passé — comparez plusieurs modèles entre eux et promouvez celui sur lequel vous pouvez compter en « production ».",
  },
];

const UNSUPERVISED_STEPS: GuideStep[] = [
  {
    icon: Shapes,
    color: "rose",
    title: "1. Découvrez des groupes (Clustering)",
    description:
      "Choisissez un dataset et les variables à analyser. Plusieurs algorithmes et nombres de groupes sont comparés automatiquement — chaque groupe est décrit par ce qui le distingue le plus des autres.",
  },
  {
    icon: ScatterChart,
    color: "blue",
    title: "2. Visualisez en 2D (Réduction de dimension)",
    description:
      "PCA, t-SNE ou UMAP ramènent vos données à 2 dimensions pour repérer visuellement des groupes ou des points isolés. Attention : t-SNE/UMAP préservent les voisinages locaux, pas les distances globales.",
  },
  {
    icon: AlertTriangle,
    color: "amber",
    title: "3. Repérez les atypiques (Détection d'anomalies)",
    description:
      "Isolation Forest et LOF tournent toujours ensemble — le score de consensus recoupe les deux méthodes. Chaque observation classée porte une explication réelle (variables qui s'écartent le plus).",
  },
];

const VISION_STEPS: GuideStep[] = [
  {
    icon: Images,
    color: "teal",
    title: "1. Classifiez vos images",
    description:
      "Importez un ZIP organisé en dossiers de classes, puis laissez DataLab comparer les architectures pré-entraînées disponibles. Chaque prédiction peut être expliquée visuellement (Grad-CAM).",
  },
  {
    icon: AlertTriangle,
    color: "violet",
    title: "2. Détectez les défauts visuels",
    description:
      "Le modèle apprend à reconstruire une image normale à partir de pièces saines uniquement ; un écart important à la reconstruction signale un défaut. Le seuil est calibré automatiquement.",
  },
];

interface FaqEntry {
  category: "donnees" | "resultats" | "compte";
  question: string;
  answer: string;
}

// Réponses fondées sur les seuils et comportements RÉELLEMENT implémentés
// (backend/domains/shared/data_quality.py, services/verdict.py,
// ModelRegistryControls) — jamais un chiffre copié d'une maquette sans
// vérification. Voir _design/JOURNAL.md, Lot 10, pour le détail des
// vérifications faites avant d'écrire chaque réponse.
const FAQ: FaqEntry[] = [
  {
    category: "resultats",
    question: "Mon score est de 0,99. C'est excellent, non ?",
    answer:
      "Presque toujours non. Un score quasi parfait sur des données réelles signale d'abord une fuite : une colonne contient, directement ou indirectement, la réponse. La plateforme signale automatiquement toute colonne trop corrélée à la cible (au-delà de 0,95 en numérique, 0,70 pour une catégorie), mais elle ne peut pas tout attraper. Le réflexe utile : « cette colonne serait-elle disponible au moment où je veux faire la prédiction ? » Si la réponse est non, elle doit sortir.",
  },
  {
    category: "donnees",
    question: "Combien de lignes me faut-il au minimum ?",
    answer:
      "100 lignes est le plancher en dessous duquel la plateforme vous avertit explicitement (l'entraînement reste possible, mais peu fiable). Au-delà, la règle pratique est d'avoir au moins 10 lignes par variable utilisée — avec moins, le modèle a plus de paramètres à régler que d'exemples pour les contraindre.",
  },
  {
    category: "resultats",
    question: "Pourquoi le score baisse quand j'active la découpe par groupe ?",
    answer:
      "C'est un signe que le score sans cette option était gonflé. Sans découpe par groupe, un même client/chantier/lot peut apparaître à la fois à l'entraînement et au test — le modèle « reconnaît » ce groupe plutôt que d'apprendre une règle générale. Le score qui baisse avec la découpe activée est le score honnête : celui que vous obtiendrez vraiment sur un groupe qu'il n'a jamais vu.",
  },
  {
    category: "resultats",
    question: "Que veut dire « intervalle à 90 % » exactement ?",
    answer:
      "Plutôt qu'une seule valeur, le modèle peut donner une fourchette dans laquelle la vraie valeur tombe la plupart du temps. Un intervalle à 90 % calibré signifie que, sur 100 prédictions de ce type, environ 90 verraient la vraie valeur tomber dans la fourchette annoncée — la fiabilité de cette promesse est elle-même mesurée après coup (couverture observée) et affichée à côté de la couverture visée.",
  },
  {
    category: "compte",
    question: "Mes données sortent-elles de mon organisation ?",
    answer:
      "Non. Chaque compte appartient à une organisation, et toutes les requêtes sont filtrées par cette organisation au niveau du serveur — un jeu de données, un modèle ou un entraînement d'une organisation n'est jamais visible ni utilisé par une autre. Le journal d'accès de votre organisation (qui a ouvert quoi, et quand) est consultable depuis votre profil.",
  },
  {
    category: "compte",
    question: "Puis-je revenir à une version précédente d'un modèle ?",
    answer:
      "Oui. Promouvoir un nouveau modèle en production ne supprime jamais l'ancien : il repasse simplement en validation (staging), toujours disponible. Reprendre une version précédente revient à la repromouvoir en production depuis l'onglet Détails de son résultat d'entraînement.",
  },
  {
    category: "resultats",
    question: "Le modèle peut-il se tromper sans que je le sache ?",
    answer:
      "Le Verdict affiché en tête de chaque résultat est justement conçu pour ça : il répond explicitement à « ce modèle surapprend-il ? », « le résultat est-il fiable ? », « le gagnant est-il vraiment meilleur que le 2ᵉ ? » — chaque affirmation cite le chiffre qui la fonde. Aucun résultat n'est présenté comme fiable sans que ces vérifications aient été faites.",
  },
];

const FAQ_CATEGORIES: { id: "toutes" | FaqEntry["category"]; label: string }[] = [
  { id: "toutes", label: "Toutes" },
  { id: "donnees", label: "Données" },
  { id: "resultats", label: "Résultats" },
  { id: "compte", label: "Compte" },
];

interface GlossaryEntry {
  term: string;
  definition: string;
}

const GLOSSARY: GlossaryEntry[] = [
  {
    term: "Sur-apprentissage",
    definition:
      "Le modèle a retenu vos exemples par cœur au lieu d'apprendre la règle. Il excelle sur ce qu'il a vu et échoue sur le reste.",
  },
  {
    term: "Validation croisée",
    definition:
      "On découpe les données en plusieurs paquets et on entraîne autant de fois, en changeant à chaque fois le paquet mis de côté. Un score moyen vaut mieux qu'un coup de chance.",
  },
  {
    term: "Fuite de données",
    definition:
      "Une information qui n'existera pas au moment réel de la prédiction s'est glissée dans l'entraînement. C'est la cause n°1 des modèles qui déçoivent en production.",
  },
  {
    term: "R²",
    definition:
      "Part de la variation expliquée par le modèle. 1 = parfait, 0 = pas mieux que de toujours répondre la moyenne. Négatif = pire que la moyenne.",
  },
  {
    term: "SHAP",
    definition:
      "Méthode qui répartit l'écart entre une prédiction et la moyenne, variable par variable. Elle répond à « pourquoi ce chiffre-là pour ce cas-là ».",
  },
  {
    term: "Score de silhouette",
    definition:
      "Mesure, de -1 à 1, si les groupes trouvés par un clustering sont réellement distincts. Proche de 1 : groupes nets. Proche de 0 : découpage arbitraire d'un nuage continu.",
  },
];

// Un pilier affiché à la fois (onglets) plutôt que les 3 empilés — retour
// utilisateur direct : la page était trop longue à faire défiler pour
// atteindre le lexique/la FAQ. `urlParam` rend l'onglet actif partageable
// par lien, même motif que les autres onglets de l'app (Tabs.tsx).
const PILLAR_GUIDES = [
  { id: "supervise" as const, label: "ML supervisé", description: "En 5 étapes — de vos données à une prédiction exploitable.", steps: SUPERVISED_STEPS },
  { id: "non-supervise" as const, label: "Non supervisé", description: "En 3 modules — sans cible à prédire, pour explorer vos données autrement.", steps: UNSUPERVISED_STEPS },
  { id: "vision" as const, label: "Vision", description: "En 2 modules — pour analyser des images plutôt que des tableaux de données.", steps: VISION_STEPS },
];
type PillarGuideId = (typeof PILLAR_GUIDES)[number]["id"];

function GuideSection({ description, steps }: { description: string; steps: GuideStep[] }) {
  return (
    <div>
      <p className="text-sm text-muted-foreground mb-3">{description}</p>
      <div className="grid gap-3 sm:grid-cols-2">
        {steps.map((step) => (
          <Card key={step.title} className="p-4 flex items-start gap-3">
            <ColorIconBadge icon={step.icon} color={step.color} size="sm" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-foreground mb-1">{step.title}</p>
              <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

function GlossaryItem({ entry, forceOpen }: { entry: GlossaryEntry; forceOpen: boolean }) {
  const [open, setOpen] = useState(false);
  const isOpen = open || forceOpen;
  return (
    <div className="border-t border-border/50 py-2.5 first:border-t-0 first:pt-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={isOpen}
        className="w-full flex items-center gap-2 text-left text-xs font-medium text-foreground"
      >
        <ChevronDown
          size={13}
          className={`flex-shrink-0 text-muted-foreground transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
        {entry.term}
      </button>
      {isOpen && (
        <p className="text-xs text-muted-foreground leading-relaxed mt-1.5 pl-[21px]">{entry.definition}</p>
      )}
    </div>
  );
}

function FaqItem({ entry }: { entry: FaqEntry }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-border/60 py-3.5 first:border-t-0 first:pt-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full flex items-start gap-2.5 text-left text-sm font-medium text-foreground"
      >
        <ChevronDown
          size={15}
          className={`flex-shrink-0 mt-0.5 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
        />
        {entry.question}
      </button>
      {open && <p className="text-xs text-muted-foreground leading-relaxed mt-2 pl-[26px]">{entry.answer}</p>}
    </div>
  );
}

/** Formulaire de retour utilisateur (Lot 10, retour direct : « ajoute un
 * formulaire pour renseigner ce problème » plutôt qu'un lien mailto vers un
 * support qui n'existe pas). Envoie réellement au serveur (`POST
 * /api/feedback`, table `Feedback`) — jamais un bouton décoratif sans effet
 * derrière, consultable ensuite par les administrateurs de l'organisation. */
function FeedbackForm() {
  const location = useLocation();
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent">("idle");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!message.trim()) return;
    setStatus("sending");
    setError(null);
    try {
      await api.feedback.create({ page: location.pathname + location.search, message: message.trim() });
      setStatus("sent");
      setMessage("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible d'envoyer ce retour pour le moment.");
      setStatus("idle");
    }
  }

  if (status === "sent") {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-success/20 bg-success/10 px-3 py-2.5 text-sm text-success">
        <CheckCircle2 size={16} className="flex-shrink-0 mt-0.5" />
        <div>
          <p>Retour envoyé. Merci.</p>
          <button type="button" onClick={() => setStatus("idle")} className="text-xs underline underline-offset-2 mt-1">
            Envoyer un autre retour
          </button>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <label htmlFor="feedback-message" className="sr-only">
        Décrivez le problème
      </label>
      <textarea
        id="feedback-message"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Décrivez ce qui bloque ou ne fonctionne pas comme attendu…"
        rows={3}
        maxLength={4000}
        className="w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 resize-y"
      />
      {error && <p className="text-xs text-destructive">{error}</p>}
      <Button type="submit" size="sm" disabled={!message.trim()} loading={status === "sending"}>
        <Send size={13} />
        Envoyer
      </Button>
    </form>
  );
}

export default function Aide() {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<"toutes" | FaqEntry["category"]>("toutes");
  const [guideOpen, setGuideOpen] = useState(false);
  const [activePillar, setActivePillar] = useState<PillarGuideId>("supervise");
  const activeGuide = PILLAR_GUIDES.find((p) => p.id === activePillar) ?? PILLAR_GUIDES[0];

  const filteredFaq = useMemo(() => {
    const q = search.trim().toLowerCase();
    return FAQ.filter((entry) => {
      if (category !== "toutes" && entry.category !== category) return false;
      if (!q) return true;
      return entry.question.toLowerCase().includes(q) || entry.answer.toLowerCase().includes(q);
    });
  }, [search, category]);

  const filteredGlossary = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return GLOSSARY;
    return GLOSSARY.filter(
      (entry) => entry.term.toLowerCase().includes(q) || entry.definition.toLowerCase().includes(q),
    );
  }, [search]);

  return (
    <AppShell>
      <PageHeader
        eyebrow="Centre d'aide"
        title="Comprendre ce que fait la plateforme"
        description="Écrit pour quelqu'un qui connaît son métier, pas le vocabulaire de l'apprentissage automatique."
        icon={HelpCircle}
        color="blue"
      />

      <div className="mb-5 relative max-w-sm">
        <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
        <input
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Chercher dans l'aide…"
          aria-label="Chercher dans l'aide"
          className="w-full h-9 rounded-lg border border-input bg-card pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
        <div className="space-y-5">
          <Card className="p-5">
            <button
              type="button"
              onClick={() => setGuideOpen((v) => !v)}
              aria-expanded={guideOpen}
              className="w-full flex items-center gap-2 text-left"
            >
              <ChevronDown
                size={16}
                className={`flex-shrink-0 text-muted-foreground transition-transform ${guideOpen ? "rotate-180" : ""}`}
              />
              <h2 className="text-h3 text-foreground">Le parcours, pilier par pilier</h2>
            </button>
            {guideOpen && (
              <div className="mt-4 space-y-4">
                <Tabs
                  items={PILLAR_GUIDES.map((p) => ({ id: p.id, label: p.label }))}
                  active={activePillar}
                  onChange={setActivePillar}
                  urlParam="pilier"
                />
                <GuideSection description={activeGuide.description} steps={activeGuide.steps} />
              </div>
            )}
          </Card>

          <Card className="p-5">
            <div className="flex items-center justify-between mb-3 flex-wrap gap-3">
              <h2 className="text-h3 text-foreground">Les questions qui reviennent</h2>
              <div className="flex items-center gap-1 rounded-lg bg-muted p-1">
                {FAQ_CATEGORIES.map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    onClick={() => setCategory(c.id)}
                    aria-pressed={category === c.id}
                    className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                      category === c.id ? "bg-card text-foreground shadow-control" : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            </div>
            {filteredFaq.length === 0 ? (
              <p className="text-sm text-muted-foreground">Aucune question ne correspond à votre recherche.</p>
            ) : (
              <div>
                {filteredFaq.map((entry) => (
                  <FaqItem key={entry.question} entry={entry} />
                ))}
              </div>
            )}
          </Card>
        </div>

        <div className="space-y-5">
          <Card className="p-5">
            <h2 className="text-h3 text-foreground mb-1">Petit lexique</h2>
            <p className="text-xs text-muted-foreground mb-3">Le vocabulaire que vous croiserez, traduit en français courant.</p>
            {filteredGlossary.length === 0 ? (
              <p className="text-xs text-muted-foreground">Aucun terme ne correspond à votre recherche.</p>
            ) : (
              <div>
                {filteredGlossary.map((entry) => (
                  <GlossaryItem key={entry.term} entry={entry} forceOpen={search.trim().length > 0} />
                ))}
              </div>
            )}
          </Card>

          <Card className="p-5 border-success/30">
            <div className="flex items-center gap-2 mb-3">
              <ShieldCheck size={16} className="text-success" />
              <h2 className="text-h3 text-foreground">Vos données</h2>
            </div>
            <ul className="space-y-2.5 text-xs text-muted-foreground">
              <li>
                Chaque organisation est cloisonnée au niveau du serveur — un jeu de données, un modèle ou un
                entraînement d'une organisation n'est jamais visible ni utilisé par une autre.
              </li>
              <li>
                Un jeu de données peut être supprimé à tout moment depuis « Mes données ».
              </li>
              <li>
                Le journal des accès de votre organisation (qui a ouvert quoi, et quand) est consultable depuis{" "}
                <Link to="/profile" className="text-primary underline underline-offset-2 hover:text-primary/80">
                  votre profil
                </Link>
                .
              </li>
            </ul>
          </Card>

          <Card className="p-5">
            <div className="flex items-center gap-2 mb-3">
              <UserCog size={16} className="text-muted-foreground" />
              <h2 className="text-h3 text-foreground">Toujours bloqué ?</h2>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed mb-3">
              Pour un problème de compte ou d'accès, l'administrateur de votre organisation peut gérer les membres
              depuis son profil. Pour un doute sur un résultat précis, le Verdict de l'entraînement concerné répond
              généralement à la question la plus utile en premier. Sinon, décrivez le problème ci-dessous — les
              administrateurs de votre organisation pourront le consulter.
            </p>
            <FeedbackForm />
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
