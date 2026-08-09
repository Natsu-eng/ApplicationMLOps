// Client API centralisé — un seul point d'accès au backend FastAPI.
// Convention reprise de CIAM : pas d'axios, fetch natif ; en dev, VITE_API_URL
// reste vide et Vite proxy /api, /auth vers localhost:8000 (voir vite.config.ts).

export const BASE_URL = import.meta.env.VITE_API_URL ?? "";

const TOKEN_KEY = "datalab_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

/** Erreur API typée — porte le code métier ({code, message}) renvoyé par le backend
 * quand il existe, pour permettre un affichage précis côté UI. */
export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(status: number, message: string, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

interface ErrorDetail {
  code?: string;
  message?: string;
}

async function extractError(res: Response): Promise<ApiError> {
  let detail: ErrorDetail | string | undefined;
  try {
    const body = await res.json();
    detail = body?.detail;
  } catch {
    // pas de corps JSON exploitable
  }
  const message =
    typeof detail === "string"
      ? detail
      : (detail?.message ?? `Erreur ${res.status}`);
  const code = typeof detail === "object" ? detail?.code : undefined;
  return new ApiError(res.status, message, code);
}

/** Requêtes JSON classiques (GET/POST/PATCH avec corps JSON), token Bearer injecté
 * automatiquement s'il est présent. */
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!res.ok) throw await extractError(res);
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

/** Requête formulaire x-www-form-urlencoded — uniquement pour /auth/login, qui
 * attend un OAuth2PasswordRequestForm côté FastAPI (username/password). */
async function requestForm<T>(path: string, fields: Record<string, string>): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    body: new URLSearchParams(fields),
  });
  if (!res.ok) throw await extractError(res);
  return res.json() as Promise<T>;
}

/** Upload multipart — ne jamais fixer Content-Type soi-même : le navigateur
 * doit poser la boundary multipart lui-même. */
async function uploadFile<T>(path: string, file: File): Promise<T> {
  const token = getToken();
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    body: formData,
  });
  if (!res.ok) throw await extractError(res);
  return res.json() as Promise<T>;
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface HealthStatus {
  status: string;
  app: string;
  version: string;
  environment: string;
  database: "up" | "down";
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  role: "owner" | "member";
  nom: string;
  organization_name: string;
}

export interface UserProfile {
  id: number;
  email: string;
  nom: string;
  role: "owner" | "member";
  organization_id: number;
  organization_name: string;
  actif: boolean;
  created_at: string;
  last_login: string | null;
}

export interface TeamMember {
  id: number;
  email: string;
  nom: string;
  role: "owner" | "member";
  actif: boolean;
  created_at: string;
}

export interface RegisterPayload {
  email: string;
  nom: string;
  password: string;
  organization_name: string;
}

export interface AddMemberPayload {
  email: string;
  nom: string;
  password: string;
}

export interface ChangePasswordPayload {
  current_password: string;
  new_password: string;
  new_password_confirm: string;
}

export interface ColumnSchema {
  name: string;
  dtype: string;
}

export interface DatasetSummary {
  id: number;
  name: string;
  file_size_bytes: number;
  row_count: number | null;
  column_count: number | null;
  status: "processing" | "ready" | "error";
  error_message: string | null;
  uploaded_by: string | null;
  created_at: string;
}

export interface DatasetDetail extends DatasetSummary {
  columns: ColumnSchema[];
}

export interface PreviewResponse {
  columns: string[];
  rows: Record<string, unknown>[];
  sample_size: number;
  row_count: number | null;
}

export type TaskType = "classification" | "regression";
export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface TrainingJobCreatePayload {
  dataset_id: number;
  target_column: string;
  feature_columns?: string[];
  task_type?: TaskType;
  group_column?: string;
  test_size?: number;
  optuna_trials?: number;
  cv_folds?: number;
}

export interface HeadlineMetric {
  name: string;
  value: number | null;
}

export interface TrainingJobSummary {
  id: number;
  dataset_id: number;
  dataset_name: string | null;
  task_type: TaskType;
  target_column: string;
  status: JobStatus;
  progress_step: string | null;
  progress_percent: number;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  algorithm: string | null;
  headline_metric: HeadlineMetric | null;
}

export interface BootstrapCI {
  mean: number;
  ci_low: number;
  ci_high: number;
}

export interface CqrResult {
  alpha: number;
  target_coverage: number;
  empirical_coverage: number;
  mean_interval_width: number;
  n_strata: number;
  strata_bounds: (number | null)[];
  qhat_per_stratum: number[];
}

export interface ShapFeature {
  feature: string;
  importance: number;
}

export interface FeatureSchemaEntry {
  name: string;
  dtype: string;
}

export interface MLModelDetail {
  id: number;
  training_job_id: number;
  algorithm: string;
  task_type: TaskType;
  target_column: string;
  feature_columns: string[];
  feature_schema: FeatureSchemaEntry[];
  metrics: Record<string, number | BootstrapCI | null>;
  shap_summary: ShapFeature[];
  cqr: CqrResult | null;
  model_card: Record<string, unknown>;
  created_at: string;
}

export interface PredictionInterval {
  low: number;
  high: number;
  confidence: number;
}

export interface PredictionResult {
  prediction: number | string;
  probabilities?: Record<string, number>;
  interval?: PredictionInterval;
}

// ── API ────────────────────────────────────────────────────────────────────

export const api = {
  health: () => request<HealthStatus>("/api/health"),

  auth: {
    register: (data: RegisterPayload) =>
      request<TokenResponse>("/auth/register", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    login: (email: string, password: string) =>
      requestForm<TokenResponse>("/auth/login", { username: email, password }),
    me: () => request<UserProfile>("/auth/me"),
    updateMe: (data: { nom?: string }) =>
      request<UserProfile>("/auth/me", { method: "PATCH", body: JSON.stringify(data) }),
    changePassword: (data: ChangePasswordPayload) =>
      request<void>("/auth/me/password", { method: "PATCH", body: JSON.stringify(data) }),
    logout: () => request<{ message: string }>("/auth/logout", { method: "POST" }),
  },

  team: {
    members: () => request<TeamMember[]>("/auth/team/members"),
    addMember: (data: AddMemberPayload) =>
      request<TeamMember>("/auth/team/members", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  },

  datasets: {
    list: () => request<DatasetSummary[]>("/datasets"),
    upload: (file: File) => uploadFile<DatasetDetail>("/datasets", file),
    get: (id: number) => request<DatasetDetail>(`/datasets/${id}`),
    preview: (id: number, limit = 50) =>
      request<PreviewResponse>(`/datasets/${id}/preview?limit=${limit}`),
    remove: (id: number) => request<void>(`/datasets/${id}`, { method: "DELETE" }),
  },

  training: {
    createJob: (data: TrainingJobCreatePayload) =>
      request<TrainingJobSummary>("/training/jobs", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    listJobs: () => request<TrainingJobSummary[]>("/training/jobs"),
    getJob: (id: number) => request<TrainingJobSummary>(`/training/jobs/${id}`),
    getModel: (id: number) => request<MLModelDetail>(`/training/jobs/${id}/model`),
    predict: (jobId: number, data: Record<string, unknown>) =>
      request<PredictionResult>(`/training/jobs/${jobId}/predict`, {
        method: "POST",
        body: JSON.stringify({ data }),
      }),
    remove: (id: number) => request<void>(`/training/jobs/${id}`, { method: "DELETE" }),
  },
};
