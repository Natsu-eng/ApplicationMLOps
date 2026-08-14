/** Interrupteur moderne (pilule + pastille glissante) — remplace les cases à
 * cocher pour un réglage ON/OFF unique (ex. "Mode expert", "Rééquilibrer les
 * classes"). Réservé aux vrais booléens : une liste à choix multiples (ex.
 * sélection de modèles) reste une case à cocher, dont la sémantique est
 * différente. Motif extrait de `ExpertModePanel.tsx` (Refonte UI) pour être
 * réutilisé partout ailleurs plutôt que dupliqué. */
export function Switch({
  checked,
  onChange,
  label,
  disabled = false,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 flex-shrink-0 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
        checked ? "bg-primary" : "bg-input"
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
          checked ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
}
