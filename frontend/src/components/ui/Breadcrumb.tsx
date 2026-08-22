import { ChevronRight } from "lucide-react";
import { Link } from "react-router-dom";

export interface BreadcrumbItem {
  label: string;
  to?: string;
}

/** Fil d'Ariane — barre haute (SPEC-UI.md §5). Le dernier élément n'est
 * jamais un lien (page courante, `aria-current="page"`) même s'il porte un
 * `to` par commodité d'appel. */
export function Breadcrumb({ items, className = "" }: { items: BreadcrumbItem[]; className?: string }) {
  return (
    <nav aria-label="Fil d'Ariane" className={`flex items-center gap-1.5 text-caption text-muted-foreground min-w-0 ${className}`}>
      {items.map((item, i) => {
        const isLast = i === items.length - 1;
        return (
          <span key={`${item.label}-${i}`} className="flex items-center gap-1.5 min-w-0">
            {i > 0 && <ChevronRight size={12} className="flex-shrink-0 opacity-50" aria-hidden="true" />}
            {item.to && !isLast ? (
              <Link to={item.to} className="hover:text-foreground transition-colors truncate rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]">
                {item.label}
              </Link>
            ) : (
              <span aria-current={isLast ? "page" : undefined} className={`truncate ${isLast ? "text-foreground font-medium" : ""}`}>
                {item.label}
              </span>
            )}
          </span>
        );
      })}
    </nav>
  );
}
