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
  clearToken,
  getToken,
  setToken as persistToken,
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
  logout: () => void;
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
      clearToken();
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
    persistToken(token.access_token);
    await refreshUser();
  }, [refreshUser]);

  const register = useCallback(async (data: RegisterPayload) => {
    const token = await api.auth.register(data);
    persistToken(token.access_token);
    await refreshUser();
  }, [refreshUser]);

  const logout = useCallback(() => {
    clearToken();
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
