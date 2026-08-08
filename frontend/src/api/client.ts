// Client API centralisé — un seul point d'accès au backend FastAPI.
// Convention reprise de CIAM : pas d'axios, fetch natif ; en dev, VITE_API_URL
// reste vide et Vite proxy /api vers localhost:8000 (voir vite.config.ts).

export const BASE_URL = import.meta.env.VITE_API_URL ?? "";

export interface HealthStatus {
  status: string;
  app: string;
  version: string;
  environment: string;
  database: "up" | "down";
}

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Requête échouée (${res.status}) sur ${path}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthStatus>("/api/health"),
};
