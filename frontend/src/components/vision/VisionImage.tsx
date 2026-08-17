import { useEffect, useState } from "react";
import { apiUrl, getToken, handleUnauthorized } from "../../api/client";

/** Affiche une image individuelle d'un dataset vision — l'endpoint
 * `GET /vision/datasets/{id}/image` exige un Bearer token, qu'une balise
 * `<img src="...">` ne peut pas envoyer : on récupère l'image via `fetch`
 * (même mécanisme que `api.training.exportModel`) puis on la convertit en
 * URL de blob locale, révoquée au démontage pour ne pas fuiter de mémoire. */
export function VisionImage({
  datasetId,
  path,
  alt,
  className = "",
}: {
  datasetId: number;
  path: string;
  alt: string;
  className?: string;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    setSrc(null);
    setFailed(false);
    const token = getToken();
    fetch(apiUrl(`/vision/datasets/${datasetId}/image?path=${encodeURIComponent(path)}`), {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    })
      .then((res) => {
        // Lot 0.3 (correctif C5) — sans ce traitement, un token expiré
        // laissait la galerie afficher "Image indisponible" indéfiniment
        // sans jamais déconnecter l'utilisateur.
        if (res.status === 401) handleUnauthorized();
        return res.ok ? res.blob() : Promise.reject(new Error("image indisponible"));
      })
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [datasetId, path]);

  if (failed) {
    return <div className={`bg-muted flex items-center justify-center text-[10px] text-muted-foreground ${className}`}>Image indisponible</div>;
  }
  if (!src) {
    return <div className={`bg-muted animate-pulse ${className}`} />;
  }
  return <img src={src} alt={alt} className={className} />;
}
