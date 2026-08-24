import { Check } from "lucide-react";

/** Barre d'étapes horizontale, partagée par les 3 wizards de l'app
 * (Training.tsx, VisionClassification.tsx, VisionAnomalies.tsx) —
 * refonte (retour utilisateur direct : l'ancien motif "pastilles +
 * chevrons" en `flex-wrap` retombait à la ligne sur un conteneur étroit
 * SANS jamais s'aligner sur une grille, laissant une pastille isolée en
 * décalé sur la ligne suivante). Avant cette refonte, ce motif était de
 * toute façon DUPLIQUÉ deux fois à l'identique (`Training.tsx` et
 * `components/vision/VisionWizard.tsx`, malgré une note explicite dans ce
 * second fichier disant l'avoir déjà extrait pour éviter la duplication —
 * qui n'a en réalité couvert que 2 des 3 wizards).
 *
 * Ligne connectée continue (motif "stepper" standard, Stripe/GitHub) :
 * jamais de retour à la ligne — `overflow-x-auto` défile horizontalement
 * si l'espace manque, l'alignement reste toujours correct quelle que soit
 * la largeur, contrairement à `flex-wrap`. Le segment de ligne ENTRE deux
 * pastilles se colore progressivement (vert = franchi), pas seulement les
 * pastilles elles-mêmes — rend la progression lisible d'un coup d'œil
 * même si les libellés sont coupés en défilement. */
export function WizardStepper({
  steps,
  activeStep,
  maxReachedStep,
  onSelect,
  ariaLabel = "Étapes",
}: {
  steps: { number: number; label: string }[];
  activeStep: number;
  maxReachedStep: number;
  onSelect: (step: number) => void;
  ariaLabel?: string;
}) {
  return (
    <nav aria-label={ariaLabel} className="overflow-x-auto no-scrollbar -mx-1 px-1 py-1">
      <ol className="flex items-center min-w-max">
        {steps.map((step, i) => {
          const state = step.number < activeStep ? "done" : step.number === activeStep ? "current" : "pending";
          const isLast = i === steps.length - 1;
          return (
            <li key={step.number} className={`flex items-center ${isLast ? "" : "flex-1 min-w-[4.5rem]"}`}>
              <WizardStepButton
                number={step.number}
                label={step.label}
                state={state}
                current={step.number === activeStep}
                disabled={step.number > maxReachedStep}
                onClick={() => onSelect(step.number)}
              />
              {!isLast && (
                <div
                  aria-hidden="true"
                  className={`h-0.5 flex-1 min-w-6 mx-1.5 rounded-full transition-colors ${
                    step.number < activeStep ? "bg-success" : "bg-border"
                  }`}
                />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function WizardStepButton({
  number,
  label,
  state,
  current,
  disabled,
  onClick,
}: {
  number: number;
  label: string;
  state: "done" | "current" | "pending";
  current: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  // bg-card/border-input (jamais bg-white en dur) : jetons de thème, pas de
  // pastille blanche vive en mode sombre — même correctif déjà appliqué à
  // l'ancien StepPill, préservé ici.
  const circleStyle = {
    done: "bg-success text-primary-foreground",
    current: "bg-primary text-primary-foreground ring-4 ring-primary/15",
    pending: "bg-card border border-input text-muted-foreground",
  }[state];
  const labelStyle = {
    done: "text-foreground/70",
    current: "text-foreground font-semibold",
    pending: "text-muted-foreground",
  }[state];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-current={current ? "step" : undefined}
      aria-label={`Étape ${number} : ${label}`}
      className="flex flex-col items-center gap-1.5 flex-shrink-0 disabled:cursor-not-allowed group"
    >
      <span
        className={`h-8 w-8 rounded-full flex items-center justify-center text-xs font-semibold flex-shrink-0 transition-colors ${circleStyle} ${
          !disabled ? "group-hover:brightness-110" : ""
        }`}
      >
        {state === "done" ? <Check size={14} strokeWidth={3} aria-hidden="true" /> : number}
      </span>
      <span className={`text-[11px] leading-tight whitespace-nowrap ${labelStyle}`}>{label}</span>
    </button>
  );
}
