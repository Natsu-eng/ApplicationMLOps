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

// Lot 10 — journal d'audit (owner uniquement).
export interface AuditLogEntry {
  id: number;
  action: string;
  target_type: string | null;
  target_id: number | null;
  details: Record<string, unknown> | null;
  actor_name: string | null;
  created_at: string;
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

export interface ColumnStat {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical";
  missing_count: number;
  missing_pct: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  median: number | null;
  n_unique: number | null;
  top_values: { value: string; count: number }[] | null;
}

export interface MissingSummaryEntry {
  column: string;
  missing_count: number;
  missing_pct: number;
}

export interface CorrelationMatrix {
  columns: string[];
  matrix: (number | null)[][];
}

export interface HistogramResponse {
  kind: "numeric" | "categorical";
  bin_edges?: number[];
  counts: number[];
  categories?: string[];
}

export interface BoxplotStat {
  column: string;
  min: number | null;
  q1: number | null;
  median: number | null;
  q3: number | null;
  max: number | null;
  outliers: number[];
  n: number;
}

export interface ScatterPoint {
  x: number | null;
  y: number | null;
}

export interface ScatterPair {
  x_column: string;
  y_column: string;
  correlation: number | null;
  points: ScatterPoint[];
}

export interface FeatureByTargetGroup {
  class_name: string;
  min: number | null;
  q1: number | null;
  median: number | null;
  q3: number | null;
  max: number | null;
  outliers: number[];
  n: number;
}

export interface FeatureByTargetResponse {
  feature: string;
  target: string;
  groups: FeatureByTargetGroup[];
}

export interface EdaResponse {
  row_count: number;
  column_stats: ColumnStat[];
  missing_summary: MissingSummaryEntry[];
  correlation_matrix: CorrelationMatrix;
  categorical_correlation_matrix: CorrelationMatrix;
  outlier_summary: BoxplotStat[];
  top_correlated_pairs: ScatterPair[];
  target_distribution: HistogramResponse | null;
}

export type WarningLevel = "info" | "attention" | "critique";

export interface DataWarning {
  level: WarningLevel;
  code: string;
  title: string;
  explanation: string;
  action: string;
  columns: string[];
  details: Record<string, unknown> | null;
}

export interface DataQualityResponse {
  warnings: DataWarning[];
}

/** Fragment de transformation (Lot 4c) — forme volontairement souple (types
 * différents selon `type`, ex. datetime_decompose vs ratio vs imputation) :
 * renvoyée telle quelle par une suggestion, échoée telle quelle si approuvée. */
export type FeatureEngineeringTransformation = Record<string, unknown> & { type: string };

export interface FeatureEngineeringChoice {
  type: string;
  options: string[];
  default: string;
}

export interface FeatureEngineeringSuggestion {
  code: string;
  title: string;
  explanation: string;
  action: string;
  columns: string[];
  based_on_warning: string | null;
  transformation: FeatureEngineeringTransformation;
  choice: FeatureEngineeringChoice | null;
}

export interface FeatureEngineeringSuggestionsResponse {
  suggestions: FeatureEngineeringSuggestion[];
}

export interface FeatureEngineeringSpec {
  version: number;
  upstream: FeatureEngineeringTransformation[];
  pipeline: {
    frequency_encoding?: string[];
    imputation?: Record<string, { strategy: string; fill_value?: unknown }>;
  };
}

export type TaskType = "classification" | "regression";
export type JobStatus = "queued" | "running" | "completed" | "failed";

// ── Clustering (Lot 11+ — ML non supervisé) ──────────────────────────────

export interface ClusteringJobCreatePayload {
  dataset_id: number;
  feature_columns: string[];
  seed?: number;
  algorithm_ids?: string[];
}

export interface ClusteringJobSummary {
  id: number;
  dataset_id: number;
  dataset_name: string | null;
  feature_columns: string[];
  status: JobStatus;
  progress_step: string | null;
  progress_percent: number;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  algorithm: string | null;
  n_clusters: number | null;
  silhouette: number | null;
}

export interface ClusterProfile {
  cluster_id: number;
  size: number;
  size_pct: number;
  numeric_summary: Record<string, { mean: number; median: number; z_score: number }>;
  categorical_summary: Record<string, { top_category: string; top_pct: number }>;
  differentiating_variables: string[];
}

export interface ClusteringResult {
  algorithm: string;
  n_clusters: number;
  metrics: {
    silhouette: number | null;
    davies_bouldin: number | null;
    calinski_harabasz: number | null;
    noise_ratio: number;
  };
  profiles: ClusterProfile[];
  model_card: Record<string, unknown>;
}

export interface ClusterCandidate {
  algorithm: string;
  family: string;
  params: Record<string, unknown>;
  n_clusters: number;
  silhouette: number | null;
  davies_bouldin: number | null;
  calinski_harabasz: number | null;
  noise_ratio: number;
  is_winner: boolean;
  rank: number;
}

export interface AlgorithmCatalogEntry {
  id: string;
  label: string;
  family: string;
  is_default: boolean;
}

// ── Réduction de dimension (Lot 13 — ML non supervisé) ───────────────────

export interface DimensionalityJobCreatePayload {
  dataset_id: number;
  feature_columns: string[];
  algorithm_id?: string;
  seed?: number;
}

export interface DimensionalityJobSummary {
  id: number;
  dataset_id: number;
  dataset_name: string | null;
  feature_columns: string[];
  algorithm_id: string;
  status: JobStatus;
  progress_step: string | null;
  progress_percent: number;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  algorithm: string | null;
  total_variance_explained: number | null;
}

export interface DimensionalityLoading {
  feature: string;
  pc1: number;
  pc2: number;
}

export interface DimensionalityResult {
  algorithm: string;
  algorithm_id: string;
  n_samples_total: number;
  n_samples_used: number;
  sampled: boolean;
  trustworthiness_primary: number;
  trustworthiness_pca: number;
  variance_explained: number[];
  total_variance_explained: number;
  loadings: DimensionalityLoading[];
  distance_fidelity_note: string;
  feature_columns: string[];
  model_card: Record<string, unknown>;
}

export interface DimensionalityPoint {
  row_index: number;
  x: number;
  y: number;
}

export interface DimensionalityColorByResponse {
  column: string;
  kind: "numeric" | "categorical";
  values: Record<string, number | string | null>;
}

// ── Détection d'anomalies (Lot 14 — ML non supervisé) ─────────────────────

export interface AnomalyJobCreatePayload {
  dataset_id: number;
  feature_columns: string[];
  top_n?: number;
  seed?: number;
}

export interface AnomalyJobSummary {
  id: number;
  dataset_id: number;
  dataset_name: string | null;
  feature_columns: string[];
  status: JobStatus;
  progress_step: string | null;
  progress_percent: number;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  n_anomalies_consensus: number | null;
  anomaly_rate_consensus: number | null;
}

export interface AnomalyResult {
  n_samples_total: number;
  n_samples_used: number;
  sampled: boolean;
  n_anomalies_isolation_forest: number;
  n_anomalies_lof: number;
  n_anomalies_consensus: number;
  anomaly_rate_isolation_forest: number;
  anomaly_rate_lof: number;
  anomaly_rate_consensus: number;
  score_histogram: { bin_edges: number[]; counts: number[] };
  model_card: Record<string, unknown>;
}

export type AnomalyAgreement = "both" | "isolation_forest_only" | "lof_only" | "none";

export interface AnomalyObservation {
  row_index: number;
  rank: number;
  consensus_score: number;
  score_isolation_forest: number;
  score_lof: number;
  is_anomaly_isolation_forest: boolean;
  is_anomaly_lof: boolean;
  agreement: AnomalyAgreement;
  numeric_deviations: Record<string, { value: number; z_score: number }>;
  categorical_flags: Record<string, { value: string; population_pct: number }>;
}

export interface TrainingJobCreatePayload {
  dataset_id: number;
  target_column: string;
  feature_columns?: string[];
  task_type?: TaskType;
  group_column?: string;
  feature_engineering?: {
    upstream: FeatureEngineeringTransformation[];
    pipeline: FeatureEngineeringSpec["pipeline"];
  };
  test_size?: number;
  optuna_trials?: number;
  cv_folds?: number;
  // Rééquilibrage des classes (lot déséquilibre) — sibling de
  // feature_engineering, jamais rejoué à l'inférence (voir client.ts).
  class_rebalancing?: boolean;
  // Mode expert (Lot E2) — reproductibilité, niveau de confiance des
  // intervalles CQR (régression uniquement) et sous-ensemble du catalogue de
  // modèles à comparer. Absents : mêmes défauts que le mode guidé aujourd'hui.
  seed?: number;
  cqr_alpha?: number;
  model_ids?: string[];
}

/** Une entrée du catalogue de modèles (Lot 5), exposée au mode expert
 * (Lot E2) — voir `GET /training/models-catalog`. */
export interface ModelCatalogEntry {
  id: string;
  label: string;
  family: string;
  is_default: boolean;
  supports_rebalancing: boolean;
  supported_tasks: TaskType[];
  slow: boolean;
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

export interface RocCurve {
  fpr: number[];
  tpr: number[];
}

export interface PrCurve {
  precision: number[];
  recall: number[];
}

export interface ClassificationEvaluation {
  confusion_matrix: number[][];
  class_names: string[];
  roc_curves: Record<string, RocCurve>;
  pr_curves: Record<string, PrCurve>;
}

export interface RegressionEvaluation {
  actual: number[];
  predicted: number[];
  residuals: number[];
}

export type ModelEvaluation = Partial<ClassificationEvaluation & RegressionEvaluation>;

// Lot Explicabilité globale — un point du beeswarm SHAP : signe + ampleur
// (shap_value) ET valeur de la variable (feature_value, sert à colorer le
// point côté frontend) pour une même observation du test.
export interface ShapBeeswarmPoint {
  feature: string;
  feature_value: number;
  shap_value: number;
}

// clé = nom de classe ("global" en régression/binaire, une clé par classe en multiclasse)
export type ShapBeeswarm = Record<string, ShapBeeswarmPoint[]>;

export interface PermutationImportanceFeature {
  feature: string;
  importance_mean: number;
  importance_std: number;
}

export interface CalibrationCurveData {
  mean_predicted: number[];
  fraction_positive: number[];
}

// clé = nom de classe (une seule clé en binaire, une par classe en multiclasse)
export type Calibration = Record<string, CalibrationCurveData>;

export interface LearningCurveData {
  train_sizes: number[];
  train_scores_mean: number[];
  train_scores_std: number[];
  val_scores_mean: number[];
  val_scores_std: number[];
  metric_label: string;
}

// Lot 9 — registre de modèles versionné.
export type ModelStage = "staging" | "production" | null;

export interface ModelRegistryEntry {
  job_id: number;
  model_id: number;
  dataset_id: number;
  dataset_name: string | null;
  task_type: TaskType;
  target_column: string;
  algorithm: string;
  stage: ModelStage;
  promoted_at: string | null;
  headline_metric: HeadlineMetric | null;
}

export interface ModelRegistryResponse {
  entries: ModelRegistryEntry[];
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
  evaluation: ModelEvaluation;
  feature_engineering: FeatureEngineeringSpec | null;
  // Lot Explicabilité globale — {}/[]/null sur les modèles entraînés avant
  // ce lot (rétrocompat), jamais absent grâce aux défauts Pydantic côté API.
  shap_beeswarm: ShapBeeswarm;
  permutation_importance: PermutationImportanceFeature[];
  calibration: Calibration | null;
  learning_curve: LearningCurveData | null;
  // Lot 9 — registre de modèles versionné. `null` = jamais promu.
  stage: ModelStage;
  promoted_at: string | null;
  created_at: string;
}

// Lot D — leaderboard : tous les modèles comparés par un job, pas
// seulement le gagnant (déjà porté par MLModelDetail.algorithm/metrics).
export interface ModelCandidate {
  algorithm: string;
  family: string;
  selection_score: number;
  is_winner: boolean;
  rank: number;
  fold_scores: number[] | null;
  secondary_metric: number | null;
  secondary_metric_label: string | null;
}

export interface LeaderboardResponse {
  selection_metric_label: string;
  candidates: ModelCandidate[];
}

// Lot D-bis — comparaison inter-jobs (pas seulement les modèles d'un même
// job, déjà couvert par LeaderboardResponse ci-dessus).
export interface JobComparisonEntry {
  job_id: number;
  dataset_id: number;
  dataset_name: string | null;
  task_type: TaskType;
  target_column: string;
  status: JobStatus;
  algorithm: string | null;
  created_at: string;
  headline_metric: HeadlineMetric | null;
  metrics: Record<string, unknown>;
  config: Record<string, unknown>;
  feature_engineering_active: boolean;
}

export interface JobComparisonResponse {
  entries: JobComparisonEntry[];
  differing_config_fields: string[];
}

export interface PredictionInterval {
  low: number;
  high: number;
  confidence: number;
}

// Lot Explicabilité locale — pourquoi CETTE prédiction précise (waterfall),
// complémentaire à l'explicabilité globale (importance moyenne) déjà
// affichée sur la page Résultats.
export interface LocalContribution {
  feature: string;
  value: number;
  contribution: number;
}

export interface LocalExplanation {
  status: "ok" | "degraded";
  message: string | null;
  base_value: number | null;
  contributions: LocalContribution[];
  other_contribution: number | null;
}

export interface PredictionResult {
  prediction: number | string;
  probabilities?: Record<string, number>;
  interval?: PredictionInterval;
  explanation?: LocalExplanation;
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
    // Lot 10 — journal d'audit (owner uniquement, 403 sinon).
    auditLog: () => request<AuditLogEntry[]>("/auth/team/audit-log"),
  },

  datasets: {
    list: () => request<DatasetSummary[]>("/datasets"),
    upload: (file: File) => uploadFile<DatasetDetail>("/datasets", file),
    get: (id: number) => request<DatasetDetail>(`/datasets/${id}`),
    preview: (id: number, limit = 50) =>
      request<PreviewResponse>(`/datasets/${id}/preview?limit=${limit}`),
    remove: (id: number) => request<void>(`/datasets/${id}`, { method: "DELETE" }),
    eda: (id: number, targetColumn?: string) =>
      request<EdaResponse>(
        `/datasets/${id}/eda${targetColumn ? `?target_column=${encodeURIComponent(targetColumn)}` : ""}`,
      ),
    histogram: (id: number, column: string, bins = 20) =>
      request<HistogramResponse>(
        `/datasets/${id}/histogram?column=${encodeURIComponent(column)}&bins=${bins}`,
      ),
    // `targetColumn` optionnel (Lot Nettoyage guidé des variables) : absent,
    // seules les détections structurelles reviennent (voir services/data_quality.py) —
    // permet d'appeler ce endpoint dès l'exploration d'un dataset, avant même
    // de choisir une cible.
    qualityCheck: (id: number, targetColumn?: string, groupColumn?: string) => {
      const params = new URLSearchParams();
      if (targetColumn) params.set("target_column", targetColumn);
      if (groupColumn) params.set("group_column", groupColumn);
      const qs = params.toString();
      return request<DataQualityResponse>(`/datasets/${id}/quality-check${qs ? `?${qs}` : ""}`);
    },
    featureByTarget: (id: number, feature: string, target: string) =>
      request<FeatureByTargetResponse>(
        `/datasets/${id}/feature-by-target?feature=${encodeURIComponent(feature)}&target=${encodeURIComponent(target)}`,
      ),
    featureEngineeringSuggestions: (id: number, targetColumn: string, groupColumn?: string) =>
      request<FeatureEngineeringSuggestionsResponse>(
        `/datasets/${id}/feature-engineering-suggestions?target_column=${encodeURIComponent(targetColumn)}` +
          (groupColumn ? `&group_column=${encodeURIComponent(groupColumn)}` : ""),
      ),
  },

  training: {
    createJob: (data: TrainingJobCreatePayload) =>
      request<TrainingJobSummary>("/training/jobs", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    modelsCatalog: () => request<{ models: ModelCatalogEntry[] }>("/training/models-catalog"),
    listJobs: () => request<TrainingJobSummary[]>("/training/jobs"),
    getJob: (id: number) => request<TrainingJobSummary>(`/training/jobs/${id}`),
    getModel: (id: number) => request<MLModelDetail>(`/training/jobs/${id}/model`),
    getCandidates: (id: number) => request<LeaderboardResponse>(`/training/jobs/${id}/candidates`),
    compareJobs: (jobIds: number[]) => {
      const qs = jobIds.map((id) => `job_ids=${id}`).join("&");
      return request<JobComparisonResponse>(`/training/jobs/compare?${qs}`);
    },
    predict: (jobId: number, data: Record<string, unknown>) =>
      request<PredictionResult>(`/training/jobs/${jobId}/predict`, {
        method: "POST",
        body: JSON.stringify({ data }),
      }),
    remove: (id: number) => request<void>(`/training/jobs/${id}`, { method: "DELETE" }),
    // Lot 9 — registre de modèles versionné.
    promoteModel: (jobId: number, stageValue: Exclude<ModelStage, null> | "none") =>
      request<MLModelDetail>(`/training/jobs/${jobId}/model/promote`, {
        method: "POST",
        body: JSON.stringify({ stage: stageValue }),
      }),
    registry: () => request<ModelRegistryResponse>("/training/models/registry"),
    /** Export de l'artefact — pas de JSON, un fichier binaire : fetch direct
     * (pas `request()`, qui suppose toujours une réponse JSON), déclenche le
     * téléchargement navigateur via un lien éphémère. */
    exportModel: async (jobId: number, suggestedFilename?: string): Promise<void> => {
      const token = getToken();
      const res = await fetch(`${BASE_URL}/training/jobs/${jobId}/model/export`, {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (!res.ok) throw await extractError(res);
      const blob = await res.blob();
      const disposition = res.headers.get("content-disposition") ?? "";
      const match = /filename="?([^"]+)"?/.exec(disposition);
      const filename = match?.[1] ?? suggestedFilename ?? `modele_job${jobId}.joblib`;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    },
  },

  clustering: {
    algorithmsCatalog: () => request<{ algorithms: AlgorithmCatalogEntry[] }>("/clustering/algorithms-catalog"),
    createJob: (data: ClusteringJobCreatePayload) =>
      request<ClusteringJobSummary>("/clustering/jobs", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    listJobs: () => request<ClusteringJobSummary[]>("/clustering/jobs"),
    getJob: (id: number) => request<ClusteringJobSummary>(`/clustering/jobs/${id}`),
    getResult: (id: number) => request<ClusteringResult>(`/clustering/jobs/${id}/result`),
    getCandidates: (id: number) => request<ClusterCandidate[]>(`/clustering/jobs/${id}/candidates`),
    remove: (id: number) => request<void>(`/clustering/jobs/${id}`, { method: "DELETE" }),
  },

  dimensionality: {
    algorithmsCatalog: () => request<{ algorithms: AlgorithmCatalogEntry[] }>("/dimensionality/algorithms-catalog"),
    createJob: (data: DimensionalityJobCreatePayload) =>
      request<DimensionalityJobSummary>("/dimensionality/jobs", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    listJobs: () => request<DimensionalityJobSummary[]>("/dimensionality/jobs"),
    getJob: (id: number) => request<DimensionalityJobSummary>(`/dimensionality/jobs/${id}`),
    getResult: (id: number) => request<DimensionalityResult>(`/dimensionality/jobs/${id}/result`),
    getPoints: (id: number) => request<DimensionalityPoint[]>(`/dimensionality/jobs/${id}/points`),
    getColorBy: (id: number, column: string) =>
      request<DimensionalityColorByResponse>(`/dimensionality/jobs/${id}/color-by?column=${encodeURIComponent(column)}`),
    remove: (id: number) => request<void>(`/dimensionality/jobs/${id}`, { method: "DELETE" }),
  },

  anomalies: {
    createJob: (data: AnomalyJobCreatePayload) =>
      request<AnomalyJobSummary>("/anomalies/jobs", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    listJobs: () => request<AnomalyJobSummary[]>("/anomalies/jobs"),
    getJob: (id: number) => request<AnomalyJobSummary>(`/anomalies/jobs/${id}`),
    getResult: (id: number) => request<AnomalyResult>(`/anomalies/jobs/${id}/result`),
    getObservations: (id: number) => request<AnomalyObservation[]>(`/anomalies/jobs/${id}/observations`),
    remove: (id: number) => request<void>(`/anomalies/jobs/${id}`, { method: "DELETE" }),
  },
};
