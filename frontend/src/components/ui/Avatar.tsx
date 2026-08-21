const SIZE_CLASSES = {
  sm: "h-7 w-7 text-caption",
  md: "h-9 w-9 text-xs",
} as const;

/** Cercle d'initiales — évite de dépendre d'un service d'avatars externe. */
export function Avatar({ name, size = "md" }: { name: string; size?: keyof typeof SIZE_CLASSES }) {
  const initials = name
    .trim()
    .split(/\s+/)
    .map((part) => part[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

  return (
    <div
      className={`flex-shrink-0 flex items-center justify-center rounded-full bg-brand-gradient text-primary-foreground font-semibold ${SIZE_CLASSES[size]}`}
    >
      {initials || "?"}
    </div>
  );
}
