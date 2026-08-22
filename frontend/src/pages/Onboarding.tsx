import { useRef, useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Compass, ShieldCheck, UploadCloud } from "lucide-react";
import { ApiError, api } from "../api/client";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";

const ACCEPTED_EXTENSIONS = ".csv,.parquet,.xlsx,.xls,.json";

/** Premier écran après l'inscription (_design/apercu/Onboarding.html) —
 * contrairement à la maquette de référence (assistant à 3 étapes complet :
 * upload, jeu de démonstration, aperçu du résultat), ce lot ne construit
 * QUE la première carte en pleine fidélité fonctionnelle : un vrai dépôt de
 * fichier branché sur `POST /datasets`. La carte "jeu de démonstration" de
 * la maquette suppose un jeu de données pré-chargé côté serveur — cette
 * capacité n'existe pas dans l'API actuelle, et ce projet a pour règle
 * explicite de ne jamais poser une UI qui a l'air fonctionnelle sans
 * l'être (AppShell.tsx, même principe déjà appliqué à la recherche/aux
 * notifications). Remplacée ici par un second choix réel : explorer le
 * produit d'abord, sans rien importer. Voir _design/JOURNAL.md, Lot 4. */
export default function Onboarding() {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setIsUploading(true);
    setError(null);
    try {
      const dataset = await api.datasets.upload(file);
      navigate(`/datasets?preview=${dataset.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Échec de l'upload");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <div className="min-h-screen relative overflow-hidden bg-background flex flex-col">
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none" aria-hidden="true">
        <div
          className="absolute left-1/2 -top-96 h-[860px] w-[860px] -translate-x-1/2 rounded-full opacity-60"
          style={{ background: "radial-gradient(circle, color-mix(in oklch, var(--accent) 17%, transparent), transparent 66%)" }}
        />
        <div
          className="absolute -right-52 -bottom-72 h-[680px] w-[680px] rounded-full opacity-60"
          style={{ background: "radial-gradient(circle, color-mix(in oklch, var(--info) 13%, transparent), transparent 68%)" }}
        />
      </div>

      <header className="relative z-10 flex items-center px-10 py-6">
        <div className="flex items-center gap-2.5">
          <img src="/icon.svg" alt="" className="h-8 w-8 rounded-lg" />
          <span className="text-h3 text-foreground">DataLab</span>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <div className="flex items-center gap-1.5" aria-hidden="true">
            <span className="h-1.5 w-5 rounded-full bg-primary" />
            <span className="h-1.5 w-1.5 rounded-full bg-muted" />
            <span className="h-1.5 w-1.5 rounded-full bg-muted" />
          </div>
          <button
            type="button"
            onClick={() => navigate("/")}
            className="text-caption text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] rounded"
          >
            Passer cette étape
          </button>
        </div>
      </header>

      <div className="relative z-10 flex-1 px-10 pb-10">
        <div className="text-center max-w-2xl mx-auto mb-8">
          <p className="ov mb-2">Bienvenue · étape 1 sur 1</p>
          <h1 className="text-h1 text-foreground mb-2.5">Commençons par un vrai résultat</h1>
          <p className="text-body text-muted-foreground">
            En quelques minutes vous aurez un modèle entraîné et un verdict à lire. Chargez un fichier, ou explorez
            d'abord le produit.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 gap-4 max-w-3xl mx-auto">
          <Card variant="interactive" className="p-6 flex flex-col" style={{ borderColor: "var(--accent)" }}>
            <div className="flex items-center gap-2.5 mb-3">
              <div className="h-9 w-9 rounded-xl bg-primary/15 flex items-center justify-center">
                <UploadCloud size={18} className="text-primary" />
              </div>
              <span className="ml-auto text-overline uppercase bg-primary text-primary-foreground px-2 py-0.5 rounded-full font-semibold">
                recommandé
              </span>
            </div>
            <h2 className="text-h3 text-foreground mb-1">J'ai mon propre fichier</h2>
            <p className="text-caption text-muted-foreground mb-4">
              La meilleure façon de juger la plateforme : sur vos données à vous.
            </p>

            <label
              onDragOver={(e: DragEvent) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e: DragEvent) => {
                e.preventDefault();
                setIsDragging(false);
                handleFiles(e.dataTransfer.files);
              }}
              className={`flex-1 flex flex-col items-center justify-center gap-2 rounded-card border-2 border-dashed px-4 py-8 text-center cursor-pointer transition-colors ${
                isDragging ? "border-primary bg-primary/6" : "border-border hover:border-primary/40"
              }`}
            >
              <UploadCloud size={26} className="text-primary" aria-hidden="true" />
              <p className="text-body font-medium text-foreground">
                {isUploading ? "Envoi en cours…" : "Déposez votre fichier ici"}
              </p>
              <p className="text-caption text-muted-foreground">ou parcourez votre ordinateur</p>
              <div className="flex flex-wrap justify-center gap-1.5 mt-1">
                {[".csv", ".xlsx", ".parquet"].map((ext) => (
                  <span key={ext} className="text-overline bg-muted text-muted-foreground px-2 py-0.5 rounded-full">
                    {ext}
                  </span>
                ))}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_EXTENSIONS}
                className="sr-only"
                disabled={isUploading}
                onChange={(e) => handleFiles(e.target.files)}
              />
            </label>

            {error && <p className="text-caption text-destructive mt-3">{error}</p>}

            <div className="flex items-start gap-2 mt-4 pt-3 border-t border-border/60">
              <ShieldCheck size={14} className="text-success flex-shrink-0 mt-0.5" aria-hidden="true" />
              <p className="text-caption text-muted-foreground">
                Stocké sur l'infrastructure de votre organisation. Jamais utilisé pour entraîner un modèle destiné à
                un autre client. Supprimable en un clic.
              </p>
            </div>
          </Card>

          <Card className="p-6 flex flex-col">
            <div className="h-9 w-9 rounded-xl bg-info/15 flex items-center justify-center mb-3">
              <Compass size={18} className="text-info" />
            </div>
            <h2 className="text-h3 text-foreground mb-1">Explorer d'abord</h2>
            <p className="text-caption text-muted-foreground mb-4 flex-1">
              Découvrez les six analyses possibles et comment lire un verdict, avant d'importer quoi que ce soit.
            </p>
            <Button variant="secondary" onClick={() => navigate("/")}>
              Découvrir le produit
            </Button>
          </Card>
        </div>
      </div>
    </div>
  );
}
