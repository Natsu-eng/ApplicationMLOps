import type { LucideIcon } from "lucide-react";
import { useSearchParams } from "react-router-dom";

export interface TabItem<T extends string> {
  id: T;
  label: string;
  icon?: LucideIcon;
}

/** Onglets modernes — contrôle segmenté (fond neutre, pastille active en
 * relief), pas un simple soulignement. Défile horizontalement sur petit
 * écran plutôt que de se compresser.
 *
 * `urlParam` (Lot 2A, AUDIT_DATALAB_2026-08-16.md §J.4) : synchronise
 * l'onglet actif avec un paramètre d'URL — un onglet de résultat devient
 * partageable par lien. Optionnel et rétrocompatible : sans lui, le
 * composant reste purement contrôlé par `active`/`onChange` comme avant. */
export function Tabs<T extends string>({
  items,
  active,
  onChange,
  urlParam,
}: {
  items: readonly TabItem<T>[];
  active: T;
  onChange: (id: T) => void;
  urlParam?: string;
}) {
  // Appelé inconditionnellement (règle des hooks) — sans effet tant que
  // `urlParam` n'est pas fourni. Exige un <Router> ambiant, toujours vrai
  // dans cette app (Tabs n'est jamais monté hors routeur).
  const [searchParams, setSearchParams] = useSearchParams();

  function select(id: T) {
    onChange(id);
    if (urlParam) {
      const next = new URLSearchParams(searchParams);
      next.set(urlParam, id);
      setSearchParams(next, { replace: true });
    }
  }

  return (
    <div className="flex items-center gap-1 overflow-x-auto rounded-control bg-muted p-1">
      {items.map((tab) => {
        const Icon = tab.icon;
        const isActive = active === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => select(tab.id)}
            className={`flex-shrink-0 flex items-center gap-1.5 rounded-control px-3 py-1.5 text-caption font-medium whitespace-nowrap transition-colors duration-150 ${
              isActive ? "bg-card text-primary shadow-control" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {Icon && <Icon size={14} />}
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
