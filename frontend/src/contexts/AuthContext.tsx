import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  clearTokens,
  getToken,
  setTokens as persistTokens,
  type RegisterPayload,
  type UserProfile,
} from "../api/client";

interface AuthContextValue {
  user: UserProfile | null;
  isAuth: boolean;
  /** true tant que la vérification du token existant (au chargement) n'est pas terminée. */
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (data: RegisterPayload) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      setIsLoading(false);
      return;
    }
    try {
      const profile = await api.auth.me();
      setUser(profile);
    } catch {
      // Token présent mais invalide/expiré côté serveur — on efface l'état local.
      clearTokens();
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Vérifie une seule fois, au montage, si un token existant est encore valide.
  useEffect(() => {
    refreshUser();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const token = await api.auth.login(email, password);
    persistTokens(token.access_token, token.refresh_token);
    await refreshUser();
  }, [refreshUser]);

  const register = useCallback(async (data: RegisterPayload) => {
    const token = await api.auth.register(data);
    persistTokens(token.access_token, token.refresh_token);
    await refreshUser();
  }, [refreshUser]);

  const logout = useCallback(async () => {
    // Révocation réelle côté serveur (Phase 1, AUDIT_BACKEND_2026-08-23.md
    // §A.2) — best-effort : même si l'appel échoue (déjà expiré, réseau
    // coupé), on efface quand même l'état local, sinon l'utilisateur reste
    // bloqué sur un bouton "déconnexion" qui ne répond pas.
    try {
      await api.auth.logout();
    } catch {
      // ignoré volontairement — voir commentaire ci-dessus
    }
    clearTokens();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{ user, isAuth: user !== null, isLoading, login, register, logout, refreshUser }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth() doit être appelé à l'intérieur de <AuthProvider>");
  return ctx;
}
