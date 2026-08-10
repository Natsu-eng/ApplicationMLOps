const SIZE_CLASSES = {
  sm: "h-7 w-7 text-[10px]",
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
      className={`flex-shrink-0 flex items-center justify-center rounded-full bg-gradient-to-br from-teal-500/15 to-teal-500/5 border border-teal-500/20 text-teal-700 font-semibold ${SIZE_CLASSES[size]}`}
    >
      {initials || "?"}
    </div>
  );
}
