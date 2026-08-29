import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, BellRing, Check } from "lucide-react";
import { api, type NotificationEntry } from "../api/client";
import { formatDateTime } from "../utils/format";

// Polling, pas SSE (retour utilisateur : "notifications de fin de job —
// email/navigateur") — une notification n'est jamais urgente à la seconde
// près comme une barre de progression activement regardée pendant un
// entraînement ; un polling léger reste plus simple à maintenir qu'un flux
// global "toute notification, tous types de job confondus" (voir
// domains/notifications/router.py pour le même raisonnement côté backend).
const POLL_INTERVAL_MS = 25_000;

/** Cloche de notifications (retour utilisateur direct) — badge de
 * comptage non lu, panneau déroulant avec les dernières notifications,
 * clic → marque lu + navigue vers le job concerné. Complète la
 * notification "navigateur" native (`window.Notification`) quand
 * l'utilisateur l'a autorisée : jamais activée sans consentement explicite
 * (bouton dédié dans le panneau), jamais redemandée si déjà refusée. */
export function NotificationBell() {
  const navigate = useNavigate();
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifications, setNotifications] = useState<NotificationEntry[] | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [browserPermission, setBrowserPermission] = useState<NotificationPermission | "unsupported">(
    typeof window !== "undefined" && "Notification" in window ? window.Notification.permission : "unsupported",
  );
  const panelRef = useRef<HTMLDivElement>(null);
  const lastSeenIdRef = useRef<number>(0);
  const firstPollRef = useRef(true);

  async function refreshUnreadCount() {
    try {
      const { count } = await api.notifications.unreadCount();
      setUnreadCount(count);
    } catch {
      // Confort, jamais bloquant — la cloche garde son dernier compte connu.
    }
  }

  // Notification navigateur (retour utilisateur : "email/navigateur") —
  // seulement les NOUVELLES notifications non lues apparues depuis le
  // dernier sondage, jamais un rattrapage de tout l'historique non lu au
  // premier chargement (bruyant, souvent périmé).
  async function checkForNewNotifications() {
    if (browserPermission !== "granted") return;
    try {
      const entries = await api.notifications.list(true);
      const fresh = entries.filter((n) => n.id > lastSeenIdRef.current);
      if (!firstPollRef.current) {
        for (const n of fresh) {
          new window.Notification(n.title, { body: n.message, icon: "/icon.svg" });
        }
      }
      if (entries.length > 0) lastSeenIdRef.current = Math.max(...entries.map((n) => n.id));
    } catch {
      // Confort, jamais bloquant.
    }
  }

  useEffect(() => {
    refreshUnreadCount();
    checkForNewNotifications();
    firstPollRef.current = false;
    const id = setInterval(() => {
      refreshUnreadCount();
      checkForNewNotifications();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [browserPermission]);

  useEffect(() => {
    if (!isOpen) return;
    api.notifications
      .list()
      .then(setNotifications)
      .catch(() => setNotifications([]));
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    function onPointerDown(e: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setIsOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setIsOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isOpen]);

  async function requestBrowserPermission() {
    if (!("Notification" in window)) return;
    const result = await window.Notification.requestPermission();
    setBrowserPermission(result);
  }

  async function handleItemClick(n: NotificationEntry) {
    setIsOpen(false);
    if (!n.read_at) {
      try {
        await api.notifications.markRead(n.id);
        setUnreadCount((c) => Math.max(0, c - 1));
      } catch {
        // Navigation quand même — un compteur pas rafraîchi n'empêche jamais d'ouvrir le job.
      }
    }
    navigate(n.link_path);
  }

  async function handleMarkAllRead() {
    try {
      await api.notifications.markAllRead();
      setUnreadCount(0);
      setNotifications((prev) => prev?.map((n) => ({ ...n, read_at: n.read_at ?? new Date().toISOString() })) ?? null);
    } catch {
      // Silencieux — pas de champ d'erreur dédié dans ce petit panneau.
    }
  }

  return (
    <div ref={panelRef} className="relative">
      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        aria-label={unreadCount > 0 ? `Notifications (${unreadCount} non lues)` : "Notifications"}
        className="relative flex items-center justify-center h-9 w-9 rounded-lg text-muted-foreground hover:bg-muted/60 hover:text-foreground transition-colors flex-shrink-0"
      >
        {unreadCount > 0 ? <BellRing size={17} /> : <Bell size={17} />}
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[10px] font-medium text-primary-foreground tabular-nums">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-80 max-h-[28rem] overflow-y-auto rounded-xl border border-border bg-card shadow-lg z-30">
          <div className="flex items-center justify-between gap-2 px-3.5 py-2.5 border-b border-border">
            <p className="text-sm font-medium text-foreground">Notifications</p>
            <div className="flex items-center gap-2">
              {browserPermission === "default" && (
                <button
                  type="button"
                  onClick={requestBrowserPermission}
                  className="text-xs text-primary hover:underline underline-offset-2"
                >
                  Activer navigateur
                </button>
              )}
              {unreadCount > 0 && (
                <button
                  type="button"
                  onClick={handleMarkAllRead}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  <Check size={12} />
                  Tout marquer lu
                </button>
              )}
            </div>
          </div>

          {notifications === null ? (
            <p className="px-3.5 py-6 text-sm text-muted-foreground text-center">Chargement…</p>
          ) : notifications.length === 0 ? (
            <p className="px-3.5 py-6 text-sm text-muted-foreground text-center">
              Aucune notification pour l'instant.
            </p>
          ) : (
            <ul className="divide-y divide-border">
              {notifications.map((n) => (
                <li key={n.id}>
                  <button
                    type="button"
                    onClick={() => handleItemClick(n)}
                    className={`block w-full text-left px-3.5 py-3 hover:bg-muted/60 transition-colors ${
                      !n.read_at ? "bg-primary/5" : ""
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      {!n.read_at && <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />}
                      <div className={`min-w-0 flex-1 ${n.read_at ? "pl-3.5" : ""}`}>
                        <p
                          className={`text-sm truncate ${n.status === "failed" ? "text-destructive" : "text-foreground"}`}
                        >
                          {n.title}
                        </p>
                        <p className="text-xs text-muted-foreground truncate">{n.message}</p>
                        <p className="text-xs text-muted-foreground/70 mt-0.5">{formatDateTime(n.created_at)}</p>
                      </div>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
