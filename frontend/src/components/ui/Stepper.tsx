import { Check, ChevronRight } from "lucide-react";

export interface StepperStep {
  number: number;
  label: string;
}

type StepState = "done" | "current" | "pending";

/** Fil d'étapes horizontal — assistant multi-écrans (entraînement, wizard
 * Vision...). Version partagée du système de design (Lot 2, bibliothèque de
 * composants) : `Training.tsx` et `VisionWizard.tsx` ont chacun leur propre
 * copie de ce motif, créée avant l'existence de ce composant — la
 * migration de ces deux appels vers celui-ci est laissée au lot qui touche
 * substantiellement chacun de ces écrans (Lot 6 pour Entraînement/
 * Progression, Lot 8 pour Vision), pas faite ici en aveugle sur des pages
 * qui fonctionnent déjà (voir _design/JOURNAL.md). */
export function Stepper({
  steps,
  activeStep,
  maxReachedStep,
  onSelect,
  ariaLabel = "Étapes",
}: {
  steps: StepperStep[];
  activeStep: number;
  maxReachedStep: number;
  onSelect: (step: number) => void;
  ariaLabel?: string;
}) {
  return (
    <nav aria-label={ariaLabel} className="flex flex-wrap items-center gap-2">
      {steps.map((step, i) => (
        <div key={step.number} className="flex items-center gap-2">
          <StepPill
            number={step.number}
            label={step.label}
            state={step.number < activeStep ? "done" : step.number === activeStep ? "current" : "pending"}
            disabled={step.number > maxReachedStep}
            onClick={() => onSelect(step.number)}
          />
          {i < steps.length - 1 && <ChevronRight size={14} className="text-muted-foreground/50 flex-shrink-0" aria-hidden="true" />}
        </div>
      ))}
    </nav>
  );
}

function StepPill({
  number,
  label,
  state,
  disabled,
  onClick,
}: {
  number: number;
  label: string;
  state: StepState;
  disabled: boolean;
  onClick: () => void;
}) {
  const pillStyle: Record<StepState, string> = {
    done: "border-success/30 bg-success/10 text-success",
    current: "border-primary/30 bg-primary/10 text-primary",
    pending: "border-border text-muted-foreground",
  };
  const circleStyle: Record<StepState, string> = {
    done: "bg-success text-primary-foreground",
    current: "bg-primary text-primary-foreground",
    pending: "bg-card border border-input text-muted-foreground",
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-current={state === "current" ? "step" : undefined}
      className={`flex items-center gap-2 rounded-full border pl-1.5 pr-3.5 py-1.5 text-caption font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:ring-[var(--accent)] disabled:opacity-[.42] disabled:cursor-not-allowed ${pillStyle[state]}`}
    >
      <span className={`flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full num text-[11px] ${circleStyle[state]}`}>
        {state === "done" ? <Check size={12} /> : number}
      </span>
      {label}
    </button>
  );
}
