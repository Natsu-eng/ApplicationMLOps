import { cloneElement, useId, type ReactElement, type ReactNode } from "react";

/** Point d'entrée unique des formulaires (Lot 2A) — compose label, aide,
 * erreur et état requis autour d'un `Input`/`Select` (ses primitives
 * internes). Avant ce composant, chaque formulaire recomposait ces quatre
 * éléments à la main (AUDIT_DATALAB_2026-08-16.md, §J.4) — jamais
 * identique d'un endroit à l'autre (parfois pas d'aide, erreur en rouge
 * direct sans rôle ARIA...).
 *
 * `children` doit être un unique élément de champ (`Input`/`Select` ou
 * équivalent) — `Field` lui injecte `id`/`aria-describedby`/
 * `aria-invalid`, il ne les fixe jamais lui-même en dur pour rester
 * générique face à n'importe quelle primitive de champ future. */
export function Field({
  label,
  help,
  error,
  required = false,
  children,
}: {
  label: string;
  help?: string;
  error?: string;
  required?: boolean;
  children: ReactElement<{ id?: string; "aria-describedby"?: string; "aria-invalid"?: boolean }>;
}) {
  const id = useId();
  const helpId = help ? `${id}-help` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  const describedBy = [helpId, errorId].filter(Boolean).join(" ") || undefined;

  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-body font-medium text-foreground">
        {label}
        {required && (
          <span className="text-destructive ml-0.5" aria-hidden="true">
            *
          </span>
        )}
      </label>
      {cloneElement(children, {
        id,
        "aria-describedby": describedBy,
        "aria-invalid": error ? true : undefined,
      })}
      {help && !error && (
        <p id={helpId} className="text-caption text-muted-foreground">
          {help}
        </p>
      )}
      {error && (
        <p id={errorId} role="alert" className="text-caption text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}

/** Variante sans label visible (ex. champ de recherche déjà contextualisé
 * par une icône) — le label reste posé pour l'accessibilité, juste masqué
 * visuellement (`sr-only`), jamais absent. */
export function FieldInline({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="sr-only">
        {label}
      </label>
      {children}
    </div>
  );
}
