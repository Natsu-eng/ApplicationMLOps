import type { LucideIcon } from "lucide-react";

interface Term {
  label: string;
  size: string;
  top: string;
  left?: string;
  right?: string;
  duration: number;
  delay: number;
  drift?: "b" | "c";
}

/** Vocabulaire ML flottant en filigrane — purement décoratif, jamais
 * indispensable à la lecture (voir @media prefers-reduced-motion dans
 * index.css). Dispersé sur toute la hauteur du panneau, à basse opacité,
 * dérive lente et déphasée (delay négatif = démarre "en cours"). */
const TERMS: Term[] = [
  { label: "R²", size: "text-4xl", left: "8%", top: "4%", duration: 18, delay: 0 },
  { label: "ROC-AUC", size: "text-2xl", right: "10%", top: "6%", duration: 20, delay: -7, drift: "b" },
  { label: "SHAP", size: "text-3xl", left: "42%", top: "2%", duration: 17, delay: -12, drift: "c" },
  { label: "LightGBM", size: "text-3xl", left: "4%", top: "20%", duration: 21, delay: -4 },
  { label: "F1-score", size: "text-2xl", right: "6%", top: "22%", duration: 19, delay: -15, drift: "b" },
  { label: "RMSE", size: "text-4xl", left: "26%", top: "30%", duration: 16, delay: -9, drift: "c" },
  { label: "XGBoost", size: "text-2xl", right: "26%", top: "34%", duration: 22, delay: -2 },
  { label: "Validation croisée", size: "text-2xl", left: "50%", top: "42%", duration: 18, delay: -17, drift: "b" },
  { label: "CatBoost", size: "text-3xl", left: "6%", top: "50%", duration: 20, delay: -6, drift: "c" },
  { label: "Gradient boosting", size: "text-2xl", right: "4%", top: "52%", duration: 17, delay: -13 },
  { label: "Optuna", size: "text-3xl", left: "30%", top: "60%", duration: 23, delay: -3, drift: "b" },
  { label: "MAE", size: "text-4xl", right: "20%", top: "66%", duration: 16, delay: -19, drift: "c" },
  { label: "Précision", size: "text-2xl", left: "8%", top: "74%", duration: 19, delay: -8 },
  { label: "Rappel", size: "text-2xl", right: "10%", top: "80%", duration: 21, delay: -1, drift: "b" },
  { label: "Intervalle de confiance", size: "text-xl", left: "36%", top: "88%", duration: 18, delay: -14, drift: "c" },
];

export interface AuthBrandFeature {
  icon: LucideIcon;
  title: string;
  description: string;
}

interface Props {
  kicker: string;
  heading: string;
  tagline: string;
  /** 3 promesses concrètes (Inscription.html) — optionnel : Login.tsx garde
   * la version plus sobre (tagline seule), Register.tsx les affiche pour
   * convaincre un premier visiteur. */
  features?: AuthBrandFeature[];
}

/** Panneau de marque partagé par Login et Register — dégradé bleu→cyan
 * (.bg-brand-gradient, E1), monogramme blanc (/icon-mark.svg — glyphe seul,
 * pas la tuile dégradée : le panneau porte déjà la couleur), motifs ML
 * flottants. Un seul composant qui change de FORME selon la largeur d'écran
 * plutôt que deux variantes : bandeau réduit en haut sous lg, panneau plein
 * à gauche au-delà — jamais complètement masqué. */
export function AuthBrandPanel({ kicker, heading, tagline, features }: Props) {
  return (
    <section className="relative overflow-hidden bg-brand-gradient text-primary-foreground flex flex-col h-36 sm:h-44 lg:h-auto lg:min-h-screen lg:w-[42%] xl:w-[40%] shrink-0 px-6 py-5 lg:p-12">
      <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-28 translate-x-28 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-44 h-44 bg-white/5 rounded-full translate-y-20 -translate-x-20 pointer-events-none hidden lg:block" />

      {/* Motifs flottants — masqués sur le bandeau mobile réduit, pas la place */}
      <div className="absolute inset-0 overflow-hidden hidden lg:block" aria-hidden="true">
        {TERMS.map((term, i) => (
          <span
            key={i}
            className={`hero-term ${term.size}`}
            style={{
              left: term.left,
              right: term.right,
              top: term.top,
              animationDuration: `${term.duration}s`,
              animationDelay: `${term.delay}s`,
              animationName: term.drift ? `hero-term-drift-${term.drift}` : undefined,
            }}
          >
            {term.label}
          </span>
        ))}
      </div>

      <div className="relative z-10 flex items-center gap-3 lg:block">
        <img src="/icon-mark.svg" alt="DataLab Pro" className="h-10 w-10 lg:h-12 lg:w-12 shrink-0" />
        <div className="lg:hidden leading-tight">
          <p className="text-sm font-semibold">DataLab Pro</p>
          <p className="text-xs text-white/70">{kicker}</p>
        </div>
      </div>

      <div className="relative z-10 mt-8 hidden lg:block">
        <span className="inline-flex items-center bg-white/15 border border-white/15 px-2.5 py-1 rounded-full text-xs font-medium text-white/90 mb-4">
          {kicker}
        </span>
        <h1 className="text-3xl xl:text-4xl font-bold leading-[1.15] tracking-tight max-w-md">
          {heading}
        </h1>
        <p className="mt-5 text-white/80 text-base leading-relaxed max-w-md">{tagline}</p>

        {features && features.length > 0 && (
          <ul className="mt-7 space-y-4 max-w-md">
            {features.map((f) => (
              <li key={f.title} className="flex items-start gap-3">
                <span className="flex-shrink-0 h-7 w-7 rounded-lg bg-white/10 flex items-center justify-center mt-0.5">
                  <f.icon size={14} className="text-white" />
                </span>
                <div>
                  <p className="text-sm font-semibold text-white">{f.title}</p>
                  <p className="text-xs text-white/70 leading-relaxed mt-0.5">{f.description}</p>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      <p className="relative z-10 mt-auto text-xs text-white/50 tracking-wide hidden lg:block">
        LightGBM · XGBoost · CatBoost · Optuna · SHAP
      </p>
    </section>
  );
}
