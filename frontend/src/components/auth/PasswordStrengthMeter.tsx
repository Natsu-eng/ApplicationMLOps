type Strength = "weak" | "medium" | "strong" | "veryStrong";

/** 5 critères, calcul local, aucun appel réseau — pattern repris de CIAM
 * (PasswordStrengthMeter.tsx). Le minimum de 8 caractères reste la seule
 * règle bloquante (Input minLength) ; ce composant encourage au-delà,
 * jamais ne bloque. */
function passwordStrengthOf(password: string): { score: number; strength: Strength } {
  const checks = [
    password.length >= 12,
    /[A-Z]/.test(password),
    /[a-z]/.test(password),
    /\d/.test(password),
    /[^A-Za-z0-9]/.test(password),
  ];
  const score = checks.filter(Boolean).length;
  const strength: Strength = score <= 1 ? "weak" : score <= 3 ? "medium" : score < 5 ? "strong" : "veryStrong";
  return { score, strength };
}

const STRENGTH_LABEL: Record<Strength, string> = {
  weak: "Faible",
  medium: "Moyen",
  strong: "Fort",
  veryStrong: "Très fort",
};
const STRENGTH_BAR: Record<Strength, string> = {
  weak: "bg-rose-500",
  medium: "bg-amber-500",
  strong: "bg-emerald-500",
  veryStrong: "bg-teal-600",
};
const STRENGTH_TEXT: Record<Strength, string> = {
  weak: "text-rose-600",
  medium: "text-amber-600",
  strong: "text-emerald-600",
  veryStrong: "text-teal-600",
};

export function PasswordStrengthMeter({ password }: { password: string }) {
  const { score, strength } = passwordStrengthOf(password);

  return (
    <div className="mt-2" aria-live="polite">
      <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${password ? STRENGTH_BAR[strength] : ""}`}
          style={{ width: `${password ? Math.max(score, 1) * 20 : 0}%` }}
        />
      </div>
      <p className={`text-xs font-semibold mt-1 ${password ? STRENGTH_TEXT[strength] : "text-slate-400"}`}>
        {password ? STRENGTH_LABEL[strength] : "Choisissez un mot de passe"}
      </p>
      <p className="text-[11px] text-slate-400 mt-0.5">
        12+ caractères avec majuscule, minuscule, chiffre et caractère spécial pour un mot de passe robuste.
      </p>
    </div>
  );
}
