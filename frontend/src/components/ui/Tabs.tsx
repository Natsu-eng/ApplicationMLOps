import type { LucideIcon } from "lucide-react";

export interface TabItem<T extends string> {
  id: T;
  label: string;
  icon?: LucideIcon;
}

/** Onglets modernes — contrôle segmenté (fond neutre, pastille active en
 * relief), pas un simple soulignement — remplace le motif "bordure basse"
 * dupliqué dans EdaModal.tsx/ModelResultModal.tsx (Lot Nettoyage guidé des
 * variables) par un composant partagé, au style plus proche des SaaS de
 * référence (Linear, Vercel). Défile horizontalement sur petit écran plutôt
 * que de se compresser. */
export function Tabs<T extends string>({
  items,
  active,
  onChange,
}: {
  items: readonly TabItem<T>[];
  active: T;
  onChange: (id: T) => void;
}) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto rounded-xl bg-muted p-1">
      {items.map((tab) => {
        const Icon = tab.icon;
        const isActive = active === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            className={`flex-shrink-0 flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-all ${
              isActive
                ? "bg-card text-primary shadow-sm"
                : "text-muted-foreground hover:text-foreground"
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
