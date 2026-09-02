// Client API centralisé — un seul point d'accès au backend FastAPI.
// Convention reprise de CIAM : pas d'axios, fetch natif ; en dev, VITE_API_URL
// reste vide et Vite proxy /api, /auth vers localhost:8000 (voir vite.config.ts).

const BASE_URL = import.meta.env.VITE_API_URL ?? "";

// Préfixe /api (correctif — incident réel de déploiement) : nginx ne
// proxifie que `location /api/` vers le backend (nginx/nginx.conf), tout le
// reste tombe dans `try_files ... /index.html` (200, HTML de la SPA) — un
// appel sans préfixe recevait donc du HTML, jamais le backend. Appliqué une
// seule fois ici (`apiUrl()`), jamais en dur dans chaque chemin d'endpoint
// ci-dessous : ces chemins restent "/auth/login", "/datasets", etc.
const API_PREFIX = "/api";

/** URL absolue (BASE_URL + préfixe /api) pour un chemin donné — unique
 * point de composition, utilisé par les fetch internes (request/
 * uploadFile(WithFields)/exportModel) ET par les fetch directs hors de ce
 * client (ex. VisionImage.tsx), pour ne jamais dupliquer le préfixe en dur
 * à deux endroits. */
export function apiUrl(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`;
}

/** Idempotence de création de job (Phase 2, AUDIT_BACKEND_2026-08-23.md
 * §F4) — un double-clic ou une requête retentée après un timeout réseau ne
 * doit jamais créer deux jobs identiques. Le composant appelant génère la
 * clé UNE fois par tentative de soumission (voir
 * `hooks/useIdempotencyKey.ts`) et la réutilise tant que la tentative n'a
 * pas abouti — jamais générée ici, qui serait rappelé à chaque fetch et
 * annulerait la protection. `undefined` = pas de déduplication demandée
 * (comportement historique, toujours valide pour les appels qui n'en ont
 * pas besoin). */
function idempotencyKeyHeader(key?: string): Record<string, string> | undefined {
  return key ? { "Idempotency-Key": key } : undefined;
}

const TOKEN_KEY = "datalab_token";
// Jeton de rafraîchissement (Phase 1, AUDIT_BACKEND_2026-08-23.md §A.1) —
// le jeton d'accès est volontairement court (20 min côté backend,
// api/core/security.py) : sans ce second jeton et le renouvellement
// transparent ci-dessous, l'utilisateur serait déconnecté toutes les 20
// minutes.
const REFRESH_TOKEN_KEY = "datalab_refresh_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

function getRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}

/** Stocke le COUPLE de jetons — le backend est rotatif (chaque appel à
 * `/auth/refresh` invalide l'ancien refresh token et en émet un nouveau) :
 * ne jamais persister l'un sans l'autre, sous peine de garder un refresh
 * token déjà révoqué en mémoire locale. */
export function setTokens(accessToken: string, refreshToken: string): void {
  localStorage.setItem(TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
}

export function clearTokens(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

/** Erreur API typée — porte le code métier ({code, message}) renvoyé par le backend
 * quand il existe, pour permettre un affichage précis côté UI.
 *
 * `requestId` (Phase 6, AUDIT_BACKEND_2026-08-23.md, Axe I) — le backend
 * inclut désormais `request_id` dans CHAQUE réponse d'erreur (Phase 1,
 * `api/main.py`, les 3 gestionnaires globaux) ; jamais capturé côté
 * frontend avant ce correctif. Voir `apiErrorReference()` ci-dessous pour
 * l'afficher de façon cohérente. */
export class ApiError extends Error {
  status: number;
  code?: string;
  requestId?: string;

  constructor(status: number, message: string, code?: string, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

interface ErrorDetail {
  code?: string;
  message?: string;
  request_id?: string;
}

/** Référence courte à afficher sous un message d'erreur serveur (5xx) —
 * jamais pour un 4xx (dataset introuvable, quota atteint...) : ces
 * erreurs sont déjà explicites, une référence de support n'y ajoute rien
 * et alourdirait l'UI. Cohérent avec le message généré côté backend pour
 * un 500 non prévu (`ERREUR_INTERNE`, `api/main.py`), qui inclut déjà la
 * même référence dans son propre texte. Établi ici comme utilitaire
 * partagé (Phase 6) — pas encore adopté site par site dans chaque page
 * (des dizaines de sites `catch (err) { setError(err.message) }`, un
 * remplacement systématique sortirait du périmètre de cette phase, voir
 * _backend/JOURNAL.md) : nouvelle infrastructure disponible, migration
 * progressive documentée comme dette explicite. */
export function apiErrorReference(err: unknown): string | undefined {
  if (!(err instanceof ApiError)) return undefined;
  if (err.status < 500 || !err.requestId) return undefined;
  return `réf. ${err.requestId}`;
}

/** Décision pure — testable sans `window` (Lot 0.3, correctif C5,
 * AUDIT_DATALAB_2026-08-16.md) : jamais rediriger si on est déjà sur
 * l'écran de connexion, sinon boucle de redirection. Extraite de l'effet de
 * bord `handleUnauthorized` ci-dessous car ce dépôt n'a pas d'environnement
 * de test `jsdom` (`window` indisponible dans les tests Vitest) — seule la
 * logique de décision est testée, jamais la navigation réelle. */
export function shouldRedirectToLogin(pathname: string): boolean {
  return !pathname.startsWith("/login");
}

/** Traitement centralisé du 401 (Lot 0.3) — appelé par TOUS les chemins qui
 * atteignent l'API, pas seulement `request()` : `uploadFile`/
 * `uploadFileWithFields`, `exportModel` (fetch direct pour un blob) et
 * `components/vision/VisionImage.tsx` (fetch direct hors du client API)
 * font chacun leur propre `fetch`, donc chacun doit appeler ce helper.
 * Jamais déclenché sur l'échec de connexion lui-même : `/auth/login` passe
 * par `requestForm()`, qui n'appelle pas ce helper (et ne renvoie de toute
 * façon jamais 401 — identifiants incorrects = 400, voir routers/auth.py). */
export function handleUnauthorized(): void {
  clearTokens();
  if (typeof window !== "undefined" && shouldRedirectToLogin(window.location.pathname)) {
    window.location.href = "/login?expired=1";
  }
}

/** Renouvellement transparent (Phase 1, AUDIT_BACKEND_2026-08-23.md §A.1) —
 * appelé UNE fois par le premier 401 rencontré ; toute requête qui échoue en
 * 401 pendant qu'un rafraîchissement est déjà en cours attend la MÊME
 * promesse (`refreshingPromise`) plutôt que de déclencher son propre appel à
 * `/auth/refresh` — sans ce partage, un écran qui déclenche 5 requêtes en
 * parallèle à l'expiration du jeton consommerait le refresh token rotatif
 * une fois puis échouerait 4 fois (rotation à usage unique côté backend).
 * Retourne le nouveau jeton d'accès, ou `null` si le refresh token est lui
 * aussi absent/expiré/révoqué (session réellement terminée). */
let refreshingPromise: Promise<string | null> | null = null;

async function tryRefreshAccessToken(): Promise<string | null> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return null;
  if (refreshingPromise) return refreshingPromise;

  refreshingPromise = (async () => {
    try {
      const res = await fetch(apiUrl("/auth/refresh"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!res.ok) return null;
      const data = (await res.json()) as TokenResponse;
      setTokens(data.access_token, data.refresh_token);
      return data.access_token;
    } catch {
      return null;
    } finally {
      refreshingPromise = null;
    }
  })();
  return refreshingPromise;
}

/** Exécute `perform` (un fetch déjà construit avec le jeton d'accès courant),
 * et si la réponse est 401, tente UN renouvellement puis rejoue `perform`
 * avec le nouveau jeton. `perform` reçoit le jeton à utiliser pour que
 * l'appelant reconstruise ses en-têtes/`Authorization` avec la valeur
 * fraîche — jamais la même closure figée sur l'ancien jeton. Centralisé ici
 * plutôt que dupliqué dans `request`/`uploadFiles`/`uploadFileWithFields`/
 * `downloadModelExport`, qui faisaient chacun leur propre `fetch` direct. */
async function withAuthRetry(perform: (token: string | null) => Promise<Response>): Promise<Response> {
  const res = await perform(getToken());
  if (res.status !== 401) return res;
  const newToken = await tryRefreshAccessToken();
  if (!newToken) {
    handleUnauthorized();
    return res;
  }
  return perform(newToken);
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
  const requestId = typeof detail === "object" ? detail?.request_id : undefined;
  return new ApiError(res.status, message, code, requestId);
}

/** Requêtes JSON classiques (GET/POST/PATCH avec corps JSON), token Bearer
 * injecté automatiquement s'il est présent. Un 401 déclenche UN
 * renouvellement transparent (voir `withAuthRetry`) avant d'abandonner. */
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await withAuthRetry((token) =>
    fetch(apiUrl(path), {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...options.headers,
      },
    }),
  );
  if (!res.ok) throw await extractError(res);
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

/** Export d'artefact — pas de JSON, un fichier binaire : fetch direct (pas
 * `request()`, qui suppose toujours une réponse JSON), déclenche le
 * téléchargement navigateur via un lien éphémère. Partagé par tous les
 * domaines qui persistent un artefact de modèle (Lot 9 supervisé, Lot 10
 * clustering/réduction de dimension/anomalies/vision) — même logique,
 * jamais dupliquée à chaque domaine. */
async function downloadModelExport(path: string, suggestedFilename: string): Promise<void> {
  const res = await withAuthRetry((token) =>
    fetch(apiUrl(path), { headers: token ? { Authorization: `Bearer ${token}` } : undefined }),
  );
  if (!res.ok) throw await extractError(res);
  const blob = await res.blob();
  const disposition = res.headers.get("content-disposition") ?? "";
  const match = /filename="?([^"]+)"?/.exec(disposition);
  const filename = match?.[1] ?? suggestedFilename;
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** Requête formulaire x-www-form-urlencoded — uniquement pour /auth/login, qui
 * attend un OAuth2PasswordRequestForm côté FastAPI (username/password). */
async function requestForm<T>(path: string, fields: Record<string, string>): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    body: new URLSearchParams(fields),
  });
  if (!res.ok) throw await extractError(res);
  return res.json() as Promise<T>;
}

/** Upload multipart — ne jamais fixer Content-Type soi-même : le navigateur
 * doit poser la boundary multipart lui-même. */
async function uploadFile<T>(path: string, file: File): Promise<T> {
  return uploadFileWithFields<T>(path, file, {});
}

/** Lot 6A — plusieurs fichiers sous le MÊME champ "files" (FastAPI
 * `List[UploadFile] = File(...)` les agrège automatiquement) : un seul
 * fichier archive (`api.visionDatasets.upload`) ou tous les fichiers d'un
 * dossier (`api.visionDatasets.uploadFolder`, chemins relatifs portés par
 * `webkitRelativePath`, voir VisionDatasets.tsx). */
async function uploadFiles<T>(path: string, files: File[]): Promise<T> {
  const formData = new FormData();
  for (const file of files) {
    // 3ᵉ argument (filename) explicite : pour un import de dossier,
    // `file.webkitRelativePath` (ex. "mon_dataset/chat/1.png") porte le
    // chemin complet — `file.name` seul ("1.png") perdrait la hiérarchie
    // de classes que le backend a besoin de voir.
    formData.append("files", file, file.webkitRelativePath || file.name);
  }
  const res = await withAuthRetry((token) =>
    fetch(apiUrl(path), {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      body: formData,
    }),
  );
  if (!res.ok) throw await extractError(res);
  return res.json() as Promise<T>;
}

/** Variante avec champs de formulaire additionnels (ex. `target_label` pour
 * Grad-CAM, `POST /vision/classification/jobs/{id}/explain`) — même
 * contrainte Content-Type que `uploadFile`. */
async function uploadFileWithFields<T>(path: string, file: File, fields: Record<string, string>): Promise<T> {
  const formData = new FormData();
  formData.append("file", file);
  for (const [key, value] of Object.entries(fields)) formData.append(key, value);
  const res = await withAuthRetry((token) =>
    fetch(apiUrl(path), {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      body: formData,
    }),
  );
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
  refresh_token: string;
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

// Lot UI (Fondations) — thème d'interface persisté côté serveur, sur le
// profil utilisateur (voir ThemeContext.tsx pour l'ordre de résolution).
export type UiTheme = "graphite" | "ivoire" | "minuit" | "ardoise" | "porcelaine";

export interface UserPreferences {
  ui_theme: UiTheme;
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

// Lot 10 — retour utilisateur libre depuis le centre d'aide.
export interface FeedbackEntry {
  id: number;
  page: string;
  message: string;
  author_name: string;
  created_at: string;
}

export interface FeedbackCreatePayload {
  page: string;
  message: string;
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

export interface ResetPasswordPayload {
  token: string;
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

export interface DatasetUsage {
  training_jobs: number;
  clustering_jobs: number;
  dimensionality_jobs: number;
  anomaly_jobs: number;
  total: number;
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
  /** Question métier à se poser avant de décider — distincte de `explanation`
   * (le pourquoi) et `action` (la recommandation). */
  question: string;
  action: string;
  columns: string[];
  details: Record<string, unknown> | null;
}

export interface DataQualityResponse {
  warnings: DataWarning[];
}

export interface TargetSuggestion {
  column: string;
  score: number;
  reasons: string[];
}

export interface TargetSuggestionsResponse {
  suggestions: TargetSuggestion[];
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
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

// ── Notifications (retour utilisateur : "notifications de fin de job —
// email/navigateur") — nommé `NotificationEntry`, jamais `Notification` :
// collision directe avec l'API navigateur native (`window.Notification`,
// utilisée telle quelle pour la notification navigateur, voir
// NotificationBell.tsx) sinon. ─────────────────────────────────────────────

export type NotificationJobType =
  | "training"
  | "clustering"
  | "dimensionality"
  | "anomaly"
  | "vision_classification"
  | "vision_anomaly"
  | "batch_prediction";

export interface NotificationEntry {
  id: number;
  job_type: NotificationJobType;
  job_id: number;
  status: "completed" | "failed";
  title: string;
  message: string;
  link_path: string;
  read_at: string | null;
  created_at: string;
}

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
  categorical_summary: Record<
    string,
    { top_category: string; top_pct: number; population_pct: number; lift: number | null }
  >;
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
  // Transparence de la sélection (rang composite sur les 3 métriques,
  // jamais la silhouette seule depuis ce correctif) — voir
  // engine.py::_attach_composite_rank. `null` pour un candidat hors budget
  // de bruit (jamais classé).
  rank_silhouette: number | null;
  rank_davies_bouldin: number | null;
  rank_calinski_harabasz: number | null;
  composite_rank: number | null;
  // Résultats complets — uniquement pour le top 3 (retour utilisateur :
  // comparer plusieurs modèles au lieu d'un seul), `null` au-delà.
  cluster_profiles: ClusterProfile[] | null;
  noise_count: number | null;
}

export interface AlgorithmCatalogEntry {
  id: string;
  label: string;
  family: string;
  is_default: boolean;
}

export type ClusterAssignmentMethod =
  | "exact"
  | "approximate_centroid"
  | "approximate_nearest_core"
  | "unsupported";

export interface ClusterPrediction {
  cluster_id: number | null;
  is_noise: boolean;
  assignment_method: ClusterAssignmentMethod;
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

// Lot 6B, §F.2 — projeter une NOUVELLE observation (contrairement à
// DimensionalityPoint, qui décrit une ligne du jeu de données
// d'entraînement déjà projetée). "unsupported" pour t-SNE (transductif,
// aucun `.transform()` natif — jamais d'approximation inventée).
export type DimensionalityProjectionMethod = "exact" | "unsupported";

export interface DimensionalityProjection {
  x: number | null;
  y: number | null;
  projection_method: DimensionalityProjectionMethod;
}

// ── Détection d'anomalies (Lot 14 — ML non supervisé) ─────────────────────

export interface AnomalyJobCreatePayload {
  dataset_id: number;
  feature_columns: string[];
  top_n?: number;
  seed?: number;
  contamination?: number;
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

// Lot 6B, §F.2 — noter une NOUVELLE observation (contrairement à
// AnomalyObservation, qui décrit une ligne du jeu de données d'entraînement
// déjà classée) : mêmes champs de score, sans rang/déviations détaillées.
export interface AnomalyScore {
  consensus_score: number;
  score_isolation_forest: number;
  score_lof: number;
  is_anomaly_isolation_forest: boolean;
  is_anomaly_lof: boolean;
  is_anomaly_consensus: boolean;
  agreement: AnomalyAgreement;
}

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

// ── Pilier Vision (Lot 15) ───────────────────────────────────────────────

export type VisionDatasetStructureType = "classification" | "mvtec_ad";
export type VisionDatasetStatus = "processing" | "ready" | "error";

export interface VisionDatasetSummary {
  id: number;
  name: string;
  structure_type: VisionDatasetStructureType;
  n_images: number;
  n_classes: number | null;
  status: VisionDatasetStatus;
  error_message: string | null;
  uploaded_by: string | null;
  created_at: string;
}

export interface VisionLabelConflictGroup {
  categories: string[];
  paths: string[];
}

export interface VisionValidationReport {
  n_corrupted: number;
  corrupted_files: string[];
  // Lot 0.1 (correctif C1, AUDIT_DATALAB_2026-08-16.md) — absents sur les
  // datasets uploadés avant ce correctif (rétrocompatibilité par absence,
  // l'ancienne forme du rapport avait n_duplicates/duplicate_groups à la
  // place, jamais réaffichée : elle décrivait des doublons "conservés",
  // plus d'actualité depuis la déduplication à l'ingestion).
  n_duplicates_removed?: number;
  duplicate_removed_files?: string[];
  label_conflicts?: VisionLabelConflictGroup[];
  duplicate_detection_note?: string;
  n_undersized: number;
  undersized_files: string[];
  warnings: string[];
  // Lot 6A (§G.3/§G.4/§G.5) — absent sur les datasets uploadés avant ce
  // correctif (rétrocompatibilité par absence).
  image_eda?: VisionImageEda;
}

export interface VisionImageEda {
  resolution_buckets: Record<string, number>;
  width: { min: number | null; max: number | null; mean: number | null };
  height: { min: number | null; max: number | null; mean: number | null };
  format_distribution: Record<string, number>;
  color_mode_distribution: Record<string, number>;
}

export interface VisionDatasetDetail extends VisionDatasetSummary {
  class_distribution: Record<string, number>;
  validation_report: VisionValidationReport;
  // Lot 6A (correctif I9) — absent (null) pour un dataset "mvtec_ad"
  // (l'augmentation d'images ne concerne que la classification).
  recommended_augmentation_preset: AugmentationPreset | null;
}

export interface VisionDatasetImageList {
  class_name: string;
  total: number;
  paths: string[];
}

export interface VisionBackbone {
  id: string;
  label: string;
  // Lot 16F — voir domains/vision/classification/services/registry.py::
  // speed_tier (jamais recalculé côté client).
  params_millions: number;
  speed_tier: "rapide" | "modere" | "lent";
}

// Lot 6A (correctif I9) — "standard" reproduit l'augmentation historique
// (seule option avant ce lot), voir services/vision_classification_training.py.
export type AugmentationPreset = "aucune" | "legere" | "standard" | "forte";

export interface VisionClassificationJobCreatePayload {
  vision_dataset_id: number;
  backbone_id?: string;
  // Mode expert (retour utilisateur direct : parité avec le comparatif
  // multi-modèles du ML tabulaire) — quand fourni (≥ 2 entrées), remplace
  // backbone_id : chaque backbone listé est entraîné, le meilleur sur la
  // validation est automatiquement retenu.
  backbone_ids?: string[];
  num_epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  weight_decay?: number;
  dropout_rate?: number;
  freeze_backbone?: boolean;
  unfreeze_after_epoch?: number | null;
  seed?: number;
  // Mode expert — résolution d'entrée (voir ALLOWED_IMAGE_SIZES,
  // ImageSizePicker). Toujours envoyée par le wizard, jamais optionnelle
  // côté backend (défaut IMAGE_SIZE) — déclarée optionnelle ici seulement
  // par cohérence avec le reste de ce payload, jamais réellement omise.
  image_size?: number;
  // Lot 6A (correctif I8) — voir ClassificationConfig côté backend pour la
  // justification de chaque défaut.
  class_weighting?: boolean;
  early_stopping_patience?: number | null;
  use_lr_scheduler?: boolean;
  // Lot 6A (correctif I9).
  augmentation_preset?: AugmentationPreset;
  // Répartition personnalisée (Lot 6A, étape "Répartition").
  val_ratio?: number;
  test_ratio?: number;
}

// Lot 6A — aperçu avant/après d'un preset d'augmentation, généré côté
// backend (voir GET /vision/datasets/{id}/augmentation-preview) : jamais
// une approximation CSS côté client, la même transformation que
// l'entraînement réel.
export interface AugmentationPreviewPair {
  original_png: string;
  augmented_png: string;
}

export interface AugmentationPreviewResult {
  preset: AugmentationPreset;
  pairs: AugmentationPreviewPair[];
}

export interface VisionClassificationJobSummary {
  id: number;
  vision_dataset_id: number;
  vision_dataset_name: string | null;
  backbone_id: string;
  status: JobStatus;
  progress_step: string | null;
  progress_percent: number;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  test_accuracy: number | null;
}

export interface VisionEpochMetrics {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

export interface VisionPredictionExample {
  relative_path: string;
  true_label: string;
  predicted_label: string;
  confidence: number;
  correct: boolean;
}

export interface VisionClassificationResult {
  backbone_id: string;
  class_names: string[];
  n_train: number;
  n_val: number;
  n_test: number;
  history: VisionEpochMetrics[];
  test_accuracy: number;
  test_precision_macro: number;
  test_recall_macro: number;
  test_f1_macro: number;
  confusion_matrix: number[][];
  examples: VisionPredictionExample[];
  model_card: Record<string, unknown>;
  // Lot 6A (correctif 16G) — même forme que ClassificationEvaluation
  // (tabulaire) pour réutiliser EvaluationCharts.tsx tel quel. Absents
  // (undefined) sur les modèles entraînés avant ce correctif.
  roc_curves?: Record<string, RocCurve>;
  pr_curves?: Record<string, PrCurve>;
  test_roc_auc?: number | null;
  // Onglet "Fiabilité" (retour utilisateur : "d'autres fonctionnalités
  // modernes que les autres plateformes n'offrent pas") — même forme que
  // `Calibration` (tabulaire) pour réutiliser CalibrationChart.tsx tel
  // quel. Absent (undefined) sur les modèles entraînés avant ce correctif.
  calibration?: Calibration | null;
}

// Mode expert : comparatif de backbones (retour utilisateur direct, parité
// avec le comparatif multi-modèles du ML tabulaire) — une entrée par
// backbone testé, vit dans `result.model_card.candidates` (voir backend
// services/engine.py::train_and_compare_backbones). N'existe que sur un job
// lancé avec `backbone_ids` (≥ 2 entrées) — absent sinon.
export interface BackboneComparisonCandidate {
  backbone_id: string;
  backbone_label: string;
  best_val_loss: number;
  best_val_accuracy: number;
  test_accuracy: number;
  num_epochs_run: number;
  time_capped: boolean;
  training_seconds: number;
  selected: boolean;
}

export interface GradCamExplanation {
  predicted_label: string;
  probabilities: Record<string, number>;
  target_label: string;
  heatmap_png: string;
}

// Grad-CAM en lot (retour utilisateur direct : "Grad-CAM devrait supporter
// le batch, pas une image à la fois") — sur des images déjà présentes dans
// le dataset (`relative_path`), typiquement les exemples mal classés déjà
// affichés dans l'onglet "Exemples". `error` non nul = cette image précise
// n'a pas pu être expliquée (jamais un lot tout-ou-rien).
export interface GradCamBatchItem {
  relative_path: string;
  predicted_label: string | null;
  probabilities: Record<string, number> | null;
  target_label: string | null;
  heatmap_png: string | null;
  error: string | null;
}

// Constat transversal sur le lot entier (pas une heatmap de plus) — retour
// d'évaluation d'une maquette externe : "une observation sur ce qui cloche"
// plutôt qu'une image à la fois. `null` si moins de 4 images expliquées
// avec succès (voir backend MIN_IMAGES_FOR_SYNTHESIS) — jamais un constat
// forcé sur un échantillon trop petit.
export interface GradCamAttentionSynthesis {
  n_images: number;
  n_border_biased: number;
  border_biased_fraction: number;
  area_fraction_border: number;
  has_notable_pattern: boolean;
  observation: string;
}

export interface GradCamBatchResponse {
  results: GradCamBatchItem[];
  synthesis: GradCamAttentionSynthesis | null;
}

export interface VisionAnomalyModelOption {
  id: string;
  label: string;
}

export interface VisionAnomalyJobCreatePayload {
  vision_dataset_id: number;
  model_id?: string;
  // Mode expert (retour utilisateur direct — parité avec `backbone_ids` de
  // la classification) — quand fourni (≥ 2 entrées), remplace model_id.
  model_ids?: string[];
  // Mode expert — résolution d'entrée (voir ALLOWED_IMAGE_SIZES,
  // ImageSizePicker), même parité que le payload de classification
  // ci-dessus.
  image_size?: number;
  num_epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  weight_decay?: number;
  mask_percentile?: number;
  seed?: number;
  // Lot 6A — parité avec la classification (même presets, voir
  // AugmentationPreset), appliqué uniquement à train/good/.
  augmentation_preset?: AugmentationPreset;
  val_ratio?: number;
}

export interface VisionAnomalyJobSummary {
  id: number;
  vision_dataset_id: number;
  vision_dataset_name: string | null;
  model_id: string;
  status: JobStatus;
  progress_step: string | null;
  progress_percent: number;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  roc_auc: number | null;
}

export interface VisionAnomalyEpochMetrics {
  epoch: number;
  train_loss: number;
  val_loss: number;
}

export interface VisionAnomalyResult {
  model_id: string;
  n_train: number;
  n_val: number;
  n_test: number;
  // Lot 0.2 (correctif C2, AUDIT_DATALAB_2026-08-16.md) — absents sur les
  // modèles entraînés avant ce correctif (rétrocompatibilité par absence).
  // model_card.threshold_calibration_status ("ok"/"degraded") +
  // .threshold_calibration_message portent l'explication à afficher.
  n_calibration?: number | null;
  n_evaluation?: number | null;
  history: VisionAnomalyEpochMetrics[];
  threshold: number;
  roc_auc: number;
  test_accuracy: number;
  test_precision: number;
  test_recall: number;
  test_f1: number;
  confusion_matrix: number[][];
  model_card: Record<string, unknown>;
  // Retour utilisateur : "rendre l'onglet anomalies aussi riche/transparent
  // que la classification" + "d'autres fonctionnalités modernes que les
  // autres plateformes n'offrent pas" — absents sur les modèles entraînés
  // avant ce correctif (rétrocompatibilité par absence).
  // Même forme EXACTE que ClassificationEvaluation.roc_curves/pr_curves
  // (une seule clé "Défaut", classe positive binaire) pour réutiliser
  // EvaluationCharts.tsx tel quel.
  roc_curves?: Record<string, RocCurve>;
  pr_curves?: Record<string, PrCurve>;
  score_histogram?: { bin_edges: number[]; normal_counts: number[]; defect_counts: number[] } | null;
  category_breakdown?: { category: string; n: number; detection_rate: number }[] | null;
  // Retour utilisateur (maquette de refonte) : "un défaut manqué coûte plus
  // cher qu'un contrôle inutile ? descendez le seuil — voici le chiffrage"
  // — absent sur les modèles entraînés avant ce correctif.
  threshold_candidates?: VisionAnomalyThresholdCandidate[] | null;
}

export interface VisionAnomalyThresholdCandidate {
  threshold: number;
  is_current: boolean;
  defects_missed: number;
  defects_missed_pct: number;
  false_alarms: number;
  false_alarms_pct: number;
}

// Mode expert : comparatif d'architectures (retour utilisateur direct,
// parité avec BackboneComparisonCandidate côté classification) — une entrée
// par architecture testée, vit dans `result.model_card.candidates`. N'existe
// que sur un job lancé avec `model_ids` (≥ 2 entrées) — absent sinon.
export interface AnomalyModelComparisonCandidate {
  model_id: string;
  model_label: string;
  best_val_loss: number;
  roc_auc: number;
  test_accuracy: number;
  num_epochs_run: number;
  time_capped: boolean;
  training_seconds: number;
  selected: boolean;
}

export interface VisionAnomalyExample {
  relative_path: string;
  defect_category: string;
  true_label: number;
  predicted_label: number;
  anomaly_score: number;
  heatmap_png: string;
  mask_png: string;
}

// Lot 6B, §F.2 — noter une NOUVELLE image (contrairement à
// VisionAnomalyExample, qui décrit une image déjà présente dans le jeu de
// test d'entraînement).
export interface VisionAnomalyScore {
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  heatmap_png: string;
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
  // Mode expert hyperparamètres (retour utilisateur direct : "laisser le
  // choix sur les hyperparamètres, profondeur des arbres etc.") — absent :
  // recherche entièrement automatique, comportement inchangé.
  hyperparameter_overrides?: Record<string, Record<string, number | string>>;
}

/** Un hyperparamètre réglable en mode expert (retour utilisateur direct :
 * "laisser le choix sur les hyperparamètres, profondeur des arbres etc.")
 * — décrit le type de contrôle à rendre, jamais un catalogue codé en dur
 * côté frontend. */
export interface HyperparamMeta {
  name: string;
  label: string;
  kind: "int" | "float" | "categorical";
  low: number | null;
  high: number | null;
  log: boolean;
  choices: string[] | null;
  help: string;
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
  tunable_hyperparameters: HyperparamMeta[];
}

/** Valeurs fixées par l'utilisateur, clé = id du modèle (registre), valeur
 * = {nom_hyperparamètre: valeur fixée} — un hyperparamètre absent reste
 * recherché automatiquement par Optuna. */
export type HyperparameterOverrides = Record<string, Record<string, number | string>>;

export interface DurationEstimate {
  status: "estimated" | "degraded";
  estimated_seconds: number | null;
  based_on_n_jobs: number;
  message: string | null;
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

/** Une version passée d'un "problème" (même dataset + même cible) — Lot 5
 * backend (correctif P1), `GET /training/jobs/{id}/model/versions` : sans
 * appelant frontend jusqu'ici malgré un rollback déjà entièrement supporté
 * côté backend (repromouvoir `job_id` en "production" démet automatiquement
 * la version courante — même mécanisme que `promoteModel`, aucun endpoint
 * séparé nécessaire pour "revenir en arrière"). */
export interface ModelVersionEntry {
  job_id: number;
  model_id: number;
  version: number;
  algorithm: string;
  stage: ModelStage;
  promoted_at: string | null;
  created_at: string;
  headline_metric: HeadlineMetric | null;
}

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

// Lot 3 (correctif I1, AUDIT_DATALAB_2026-08-16.md §E.3) — verdict en
// langage clair calculé côté serveur (services/model_verdict.py), jamais
// recalculé côté frontend : mêmes règles partout (page Résultats, futures
// extensions clustering/vision), un seul endroit à faire évoluer.
export type VerdictLevel = "critique" | "attention" | "info";

export interface VerdictClaim {
  level: VerdictLevel;
  code: string;
  title: string;
  explanation: string;
  details: Record<string, unknown>;
}

export interface ModelVerdictData {
  claims: VerdictClaim[];
  next_action: string;
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
  // Lot 3 — voir ModelVerdictData ci-dessus. Toujours présent (calculé côté
  // serveur à la volée, jamais persisté).
  verdict: ModelVerdictData;
  // Lot 9 — registre de modèles versionné. `null` = jamais promu.
  stage: ModelStage;
  promoted_at: string | null;
  created_at: string;
  // Lot 5 (correctif P1) — numéro de version au sein du problème (dataset +
  // cible), voir `api/core/models.py::MLModel.version` côté backend.
  version: number;
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

/** Une prédiction passée, avec son lignage (Phase 3 backend,
 * AUDIT_BACKEND_2026-08-23.md, Axe I) — `GET /training/jobs/{id}/predictions`
 * n'avait jusqu'ici AUCUN appelant côté frontend malgré une persistance
 * complète côté backend (entrée, sortie, dataset/job/version d'origine) :
 * chaque prédiction disparaissait de l'écran sitôt la réponse affichée. */
export interface PredictionHistoryEntry {
  id: number;
  input: Record<string, unknown>;
  prediction: number | string;
  probabilities?: Record<string, number>;
  interval?: PredictionInterval;
  requested_by?: string;
  dataset_id: number;
  training_job_id: number;
  model_version: number;
  created_at: string;
}

// Dérive des données — voir backend/domains/shared/drift.py (PSI). severity
// : "stable" / "modere" / "significatif".
export interface DriftFeature {
  feature: string;
  psi: number;
  severity: "stable" | "modere" | "significatif";
}

export interface DriftReport {
  n_predictions_analyzed: number;
  insufficient_data: boolean;
  features: DriftFeature[];
  n_significant: number;
  n_moderate: number;
  min_predictions_required: number;
}

// Prédiction en lot (retour utilisateur : "batch prediction" — upload d'un
// fichier, prédictions pour toutes les lignes) — même forme que
// TrainingJobSummary (statut/progression/erreur), traitée par le frontend
// comme n'importe quel autre job de fond.
export interface BatchPredictionJobSummary {
  id: number;
  training_job_id: number;
  input_filename: string;
  status: JobStatus;
  progress_step: string | null;
  progress_percent: number;
  error_message: string | null;
  n_rows: number | null;
  created_by: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

// Lot 4 (correctif I3, AUDIT_DATALAB_2026-08-16.md §C.2.4) — remplace les 8
// appels de liste complets faits par Dashboard.tsx au montage par un seul
// aller-retour agrégé (GET /dashboard/summary). Mêmes formes de résumé de
// job que les endpoints de liste dédiés (TrainingJobSummary, etc.) —
// jamais une forme dupliquée.
export interface DashboardSummary {
  members_count: number;
  datasets_count: number;
  recent_datasets: DatasetSummary[];
  supervised_count: number;
  unsupervised_count: number;
  vision_count: number;
  active_count: number;
  // Part des modèles ML tabulaire en staging/production dont le verdict ne
  // comporte aucune affirmation "critique" — `null` si aucun modèle n'est
  // en staging/production (jamais un faux 0 %).
  active_models_reliability_pct: number | null;
  recent_supervised: TrainingJobSummary[];
  recent_clustering: ClusteringJobSummary[];
  recent_dimensionality: DimensionalityJobSummary[];
  recent_anomalies: AnomalyJobSummary[];
  recent_vision_classification: VisionClassificationJobSummary[];
  recent_vision_anomalies: VisionAnomalyJobSummary[];
}

// ── API ────────────────────────────────────────────────────────────────────

export const api = {
  health: () => request<HealthStatus>("/health"),

  dashboard: {
    summary: () => request<DashboardSummary>("/dashboard/summary"),
  },

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
    // Envoie le refresh token pour qu'il soit révoqué lui aussi (Phase 1,
    // AUDIT_BACKEND_2026-08-23.md §A.2) — sans lui, seul le jeton access de
    // CETTE requête serait révoqué, le refresh resté valide permettrait de
    // regénérer un accès juste après.
    logout: () =>
      request<void>("/auth/logout", {
        method: "POST",
        body: JSON.stringify({ refresh_token: getRefreshToken() }),
      }),
    // Phase 1B — réponse 204 identique que l'adresse corresponde à un
    // compte ou non (voir domains/auth/router.py) : jamais de branchement
    // différent ici selon le résultat, l'écran appelant affiche toujours
    // le même message de confirmation.
    requestPasswordReset: (email: string) =>
      request<void>("/auth/password-reset/request", { method: "POST", body: JSON.stringify({ email }) }),
    confirmPasswordReset: (data: ResetPasswordPayload) =>
      request<void>("/auth/password-reset/confirm", { method: "POST", body: JSON.stringify(data) }),
  },

  users: {
    preferences: () => request<UserPreferences>("/users/me/preferences"),
    updatePreferences: (data: UserPreferences) =>
      request<UserPreferences>("/users/me/preferences", { method: "PATCH", body: JSON.stringify(data) }),
  },

  // Lot 10 — retour utilisateur libre (centre d'aide).
  feedback: {
    create: (data: FeedbackCreatePayload) =>
      request<FeedbackEntry>("/feedback", { method: "POST", body: JSON.stringify(data) }),
    list: () => request<FeedbackEntry[]>("/feedback"),
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
    usage: (id: number) => request<DatasetUsage>(`/datasets/${id}/usage`),
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
    // `featureColumns` (retour utilisateur direct — diagnostic de cohérence
    // du wizard : "une colonne exclue à l'étape 1 déclenche encore une
    // alerte à l'étape 2") : liste blanche des variables encore retenues,
    // absente = toutes les colonnes du dataset (comportement historique,
    // ex. EdaModal.tsx qui n'a pas de notion de sélection de variables).
    qualityCheck: (id: number, targetColumn?: string, groupColumn?: string, featureColumns?: string[]) => {
      const params = new URLSearchParams();
      if (targetColumn) params.set("target_column", targetColumn);
      if (groupColumn) params.set("group_column", groupColumn);
      if (featureColumns) params.set("feature_columns", featureColumns.join(","));
      const qs = params.toString();
      return request<DataQualityResponse>(`/datasets/${id}/quality-check${qs ? `?${qs}` : ""}`);
    },
    featureByTarget: (id: number, feature: string, target: string) =>
      request<FeatureByTargetResponse>(
        `/datasets/${id}/feature-by-target?feature=${encodeURIComponent(feature)}&target=${encodeURIComponent(target)}`,
      ),
    featureEngineeringSuggestions: (id: number, targetColumn: string, groupColumn?: string, featureColumns?: string[]) =>
      request<FeatureEngineeringSuggestionsResponse>(
        `/datasets/${id}/feature-engineering-suggestions?target_column=${encodeURIComponent(targetColumn)}` +
          (groupColumn ? `&group_column=${encodeURIComponent(groupColumn)}` : "") +
          (featureColumns ? `&feature_columns=${encodeURIComponent(featureColumns.join(","))}` : ""),
      ),
    targetSuggestions: (id: number) => request<TargetSuggestionsResponse>(`/datasets/${id}/target-suggestions`),
  },

  training: {
    createJob: (data: TrainingJobCreatePayload, idempotencyKey?: string) =>
      request<TrainingJobSummary>("/training/jobs", {
        method: "POST",
        body: JSON.stringify(data),
        headers: idempotencyKeyHeader(idempotencyKey),
      }),
    modelsCatalog: () => request<{ models: ModelCatalogEntry[] }>("/training/models-catalog"),
    estimateDuration: (datasetId: number, nModels: number, optunaTrials: number, cvFolds: number) =>
      request<DurationEstimate>(
        `/training/estimate-duration?dataset_id=${datasetId}&n_models=${nModels}&optuna_trials=${optunaTrials}&cv_folds=${cvFolds}`,
      ),
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
    predictions: (jobId: number, limit = 50) =>
      request<{ entries: PredictionHistoryEntry[] }>(`/training/jobs/${jobId}/predictions?limit=${limit}`),
    getDrift: (jobId: number) => request<DriftReport>(`/training/jobs/${jobId}/model/drift`),
    cancel: (id: number) => request<TrainingJobSummary>(`/training/jobs/${id}/cancel`, { method: "POST" }),
    rerun: (id: number) => request<TrainingJobSummary>(`/training/jobs/${id}/rerun`, { method: "POST" }),
    remove: (id: number) => request<void>(`/training/jobs/${id}`, { method: "DELETE" }),
    // Lot 9 — registre de modèles versionné.
    promoteModel: (jobId: number, stageValue: Exclude<ModelStage, null> | "none") =>
      request<MLModelDetail>(`/training/jobs/${jobId}/model/promote`, {
        method: "POST",
        body: JSON.stringify({ stage: stageValue }),
      }),
    registry: () => request<ModelRegistryResponse>("/training/models/registry"),
    modelVersions: (jobId: number) => request<{ entries: ModelVersionEntry[] }>(`/training/jobs/${jobId}/model/versions`),
    exportModel: (jobId: number, suggestedFilename?: string) =>
      downloadModelExport(`/training/jobs/${jobId}/model/export`, suggestedFilename ?? `modele_job${jobId}.joblib`),
    // Script de déploiement autonome (retour utilisateur direct : "tous les
    // modèles doivent pouvoir être déployés dans d'autres plateformes") —
    // aucune dépendance à ce projet, voir services/deployment_export.py.
    exportDeploymentScript: (jobId: number, suggestedFilename?: string) =>
      downloadModelExport(
        `/training/jobs/${jobId}/model/export-script`,
        suggestedFilename ?? `modele_job${jobId}_deploiement.py`,
      ),
    // Prédiction en lot (retour utilisateur : "batch prediction" — upload
    // d'un fichier, prédictions pour toutes les lignes) — tâche de fond
    // comme un entraînement, jamais un calcul synchrone (fichier non borné
    // en taille, contrairement à `predict` ci-dessus).
    createBatchPrediction: (jobId: number, file: File) =>
      uploadFile<BatchPredictionJobSummary>(`/training/jobs/${jobId}/predict-batch`, file),
    listBatchPredictions: () => request<BatchPredictionJobSummary[]>("/training/batch-predictions"),
    getBatchPrediction: (id: number) => request<BatchPredictionJobSummary>(`/training/batch-predictions/${id}`),
    cancelBatchPrediction: (id: number) =>
      request<BatchPredictionJobSummary>(`/training/batch-predictions/${id}/cancel`, { method: "POST" }),
    removeBatchPrediction: (id: number) => request<void>(`/training/batch-predictions/${id}`, { method: "DELETE" }),
    downloadBatchPredictionResult: (id: number, filename: string) =>
      downloadModelExport(`/training/batch-predictions/${id}/download`, filename),
    // Même résultat, au format Excel (retour utilisateur direct : "on doit
    // télécharger aussi les prédictions en format excel pour voir
    // directement") — généré à la volée côté serveur depuis le CSV déjà
    // stocké, jamais un second fichier persisté.
    downloadBatchPredictionResultExcel: (id: number, filename: string) =>
      downloadModelExport(`/training/batch-predictions/${id}/download-excel`, filename),
  },

  clustering: {
    algorithmsCatalog: () => request<{ algorithms: AlgorithmCatalogEntry[] }>("/clustering/algorithms-catalog"),
    createJob: (data: ClusteringJobCreatePayload, idempotencyKey?: string) =>
      request<ClusteringJobSummary>("/clustering/jobs", {
        method: "POST",
        body: JSON.stringify(data),
        headers: idempotencyKeyHeader(idempotencyKey),
      }),
    listJobs: () => request<ClusteringJobSummary[]>("/clustering/jobs"),
    getJob: (id: number) => request<ClusteringJobSummary>(`/clustering/jobs/${id}`),
    getResult: (id: number) => request<ClusteringResult>(`/clustering/jobs/${id}/result`),
    getCandidates: (id: number) => request<ClusterCandidate[]>(`/clustering/jobs/${id}/candidates`),
    predict: (id: number, data: Record<string, unknown>) =>
      request<ClusterPrediction>(`/clustering/jobs/${id}/predict`, {
        method: "POST",
        body: JSON.stringify({ data }),
      }),
    getDrift: (id: number) => request<DriftReport>(`/clustering/jobs/${id}/drift`),
    cancel: (id: number) => request<ClusteringJobSummary>(`/clustering/jobs/${id}/cancel`, { method: "POST" }),
    rerun: (id: number) => request<ClusteringJobSummary>(`/clustering/jobs/${id}/rerun`, { method: "POST" }),
    remove: (id: number) => request<void>(`/clustering/jobs/${id}`, { method: "DELETE" }),
    exportModel: (id: number) => downloadModelExport(`/clustering/jobs/${id}/model/export`, `clustering_job${id}.joblib`),
    exportDeploymentScript: (id: number) =>
      downloadModelExport(`/clustering/jobs/${id}/model/export-script`, `clustering_job${id}_deploiement.py`),
    // Assigne un cluster à CHAQUE ligne du dataset d'origine (retour
    // utilisateur direct : "l'entreprise a 50 000 lignes, la plateforme
    // n'en clusterise que 5 000, comment couvrir le reste ?") — jamais
    // seulement l'échantillon effectivement utilisé à l'entraînement.
    exportAssignments: (id: number) =>
      downloadModelExport(`/clustering/jobs/${id}/assignments/export`, `clustering_assignations_job${id}.csv`),
  },

  dimensionality: {
    algorithmsCatalog: () => request<{ algorithms: AlgorithmCatalogEntry[] }>("/dimensionality/algorithms-catalog"),
    createJob: (data: DimensionalityJobCreatePayload, idempotencyKey?: string) =>
      request<DimensionalityJobSummary>("/dimensionality/jobs", {
        method: "POST",
        body: JSON.stringify(data),
        headers: idempotencyKeyHeader(idempotencyKey),
      }),
    listJobs: () => request<DimensionalityJobSummary[]>("/dimensionality/jobs"),
    getJob: (id: number) => request<DimensionalityJobSummary>(`/dimensionality/jobs/${id}`),
    getResult: (id: number) => request<DimensionalityResult>(`/dimensionality/jobs/${id}/result`),
    getPoints: (id: number) => request<DimensionalityPoint[]>(`/dimensionality/jobs/${id}/points`),
    getColorBy: (id: number, column: string) =>
      request<DimensionalityColorByResponse>(`/dimensionality/jobs/${id}/color-by?column=${encodeURIComponent(column)}`),
    project: (id: number, data: Record<string, unknown>) =>
      request<DimensionalityProjection>(`/dimensionality/jobs/${id}/project`, {
        method: "POST",
        body: JSON.stringify({ data }),
      }),
    getDrift: (id: number) => request<DriftReport>(`/dimensionality/jobs/${id}/drift`),
    cancel: (id: number) => request<DimensionalityJobSummary>(`/dimensionality/jobs/${id}/cancel`, { method: "POST" }),
    rerun: (id: number) => request<DimensionalityJobSummary>(`/dimensionality/jobs/${id}/rerun`, { method: "POST" }),
    remove: (id: number) => request<void>(`/dimensionality/jobs/${id}`, { method: "DELETE" }),
    exportModel: (id: number) => downloadModelExport(`/dimensionality/jobs/${id}/model/export`, `projection_job${id}.joblib`),
    exportDeploymentScript: (id: number) =>
      downloadModelExport(`/dimensionality/jobs/${id}/model/export-script`, `projection_job${id}_deploiement.py`),
    exportPoints: (id: number) =>
      downloadModelExport(`/dimensionality/jobs/${id}/points/export`, `projection_points_job${id}.csv`),
  },

  anomalies: {
    createJob: (data: AnomalyJobCreatePayload, idempotencyKey?: string) =>
      request<AnomalyJobSummary>("/anomalies/jobs", {
        method: "POST",
        body: JSON.stringify(data),
        headers: idempotencyKeyHeader(idempotencyKey),
      }),
    listJobs: () => request<AnomalyJobSummary[]>("/anomalies/jobs"),
    getJob: (id: number) => request<AnomalyJobSummary>(`/anomalies/jobs/${id}`),
    getResult: (id: number) => request<AnomalyResult>(`/anomalies/jobs/${id}/result`),
    getObservations: (id: number) => request<AnomalyObservation[]>(`/anomalies/jobs/${id}/observations`),
    predict: (id: number, data: Record<string, unknown>) =>
      request<AnomalyScore>(`/anomalies/jobs/${id}/predict`, {
        method: "POST",
        body: JSON.stringify({ data }),
      }),
    getDrift: (id: number) => request<DriftReport>(`/anomalies/jobs/${id}/drift`),
    cancel: (id: number) => request<AnomalyJobSummary>(`/anomalies/jobs/${id}/cancel`, { method: "POST" }),
    rerun: (id: number) => request<AnomalyJobSummary>(`/anomalies/jobs/${id}/rerun`, { method: "POST" }),
    remove: (id: number) => request<void>(`/anomalies/jobs/${id}`, { method: "DELETE" }),
    exportModel: (id: number) => downloadModelExport(`/anomalies/jobs/${id}/model/export`, `anomalies_job${id}.joblib`),
    exportDeploymentScript: (id: number) =>
      downloadModelExport(`/anomalies/jobs/${id}/model/export-script`, `anomalies_job${id}_deploiement.py`),
    // Note CHAQUE ligne du dataset d'origine (retour utilisateur direct,
    // même esprit que clustering.exportAssignments) — pas seulement
    // l'échantillon effectivement utilisé à l'entraînement.
    exportScores: (id: number) =>
      downloadModelExport(`/anomalies/jobs/${id}/observations/export`, `anomalies_scores_job${id}.csv`),
  },

  visionDatasets: {
    list: () => request<VisionDatasetSummary[]>("/vision/datasets"),
    // Lot 6A — une seule archive (.zip/.tar/.tar.gz/.tgz) OU tous les
    // fichiers d'un dossier sélectionné (voir uploadFiles/uploadFolder) :
    // même endpoint côté backend, distingué par le nombre de fichiers reçus.
    upload: (file: File) => uploadFiles<VisionDatasetDetail>("/vision/datasets", [file]),
    uploadFolder: (files: File[]) => uploadFiles<VisionDatasetDetail>("/vision/datasets", files),
    get: (id: number) => request<VisionDatasetDetail>(`/vision/datasets/${id}`),
    listImages: (id: number, className: string) =>
      request<VisionDatasetImageList>(`/vision/datasets/${id}/images?class_name=${encodeURIComponent(className)}`),
    // Lot 6A — aperçu réel avant/après (même transformation que l'entraînement,
    // voir services/vision_classification_training.py::augmentation_transforms),
    // partagé par les deux wizards Vision (classification ET anomalies).
    // `imageSize` optionnel (mode expert, retour utilisateur direct : "vision
    // n'offre pas de réduire/augmenter la taille des images") — reflète la
    // résolution réellement choisie par l'utilisateur, jamais la valeur par
    // défaut du pilier si l'utilisateur en a choisi une autre.
    augmentationPreview: (id: number, preset: AugmentationPreset, imageSize?: number) =>
      request<AugmentationPreviewResult>(
        `/vision/datasets/${id}/augmentation-preview?preset=${preset}${imageSize ? `&image_size=${imageSize}` : ""}`,
      ),
    remove: (id: number) => request<void>(`/vision/datasets/${id}`, { method: "DELETE" }),
  },

  visionClassification: {
    backbones: () => request<VisionBackbone[]>("/vision/classification/backbones"),
    createJob: (data: VisionClassificationJobCreatePayload, idempotencyKey?: string) =>
      request<VisionClassificationJobSummary>("/vision/classification/jobs", {
        method: "POST",
        body: JSON.stringify(data),
        headers: idempotencyKeyHeader(idempotencyKey),
      }),
    listJobs: () => request<VisionClassificationJobSummary[]>("/vision/classification/jobs"),
    getJob: (id: number) => request<VisionClassificationJobSummary>(`/vision/classification/jobs/${id}`),
    getResult: (id: number) => request<VisionClassificationResult>(`/vision/classification/jobs/${id}/result`),
    explain: (id: number, file: File, targetLabel?: string) =>
      uploadFileWithFields<GradCamExplanation>(
        `/vision/classification/jobs/${id}/explain`,
        file,
        targetLabel ? { target_label: targetLabel } : {},
      ),
    // Grad-CAM en lot sur des images déjà présentes dans le dataset (retour
    // utilisateur direct : "Grad-CAM devrait supporter le batch") — zéro
    // upload, un ou plusieurs `relative_path` (voir PredictionExampleOut).
    explainDatasetExamples: (id: number, relativePaths: string[]) =>
      request<GradCamBatchResponse>(`/vision/classification/jobs/${id}/explain-dataset-examples`, {
        method: "POST",
        body: JSON.stringify({ relative_paths: relativePaths }),
      }),
    // Grad-CAM en lot sur des images EXTERNES uploadées (retour utilisateur
    // direct : "possible d'expliquer plusieurs images en même temps ?") —
    // complémentaire d'`explain` (une seule image) et
    // d'`explainDatasetExamples` (images déjà dans le dataset).
    explainBatch: (id: number, files: File[]) =>
      uploadFiles<GradCamBatchResponse>(`/vision/classification/jobs/${id}/explain-batch`, files),
    cancel: (id: number) =>
      request<VisionClassificationJobSummary>(`/vision/classification/jobs/${id}/cancel`, { method: "POST" }),
    rerun: (id: number) =>
      request<VisionClassificationJobSummary>(`/vision/classification/jobs/${id}/rerun`, { method: "POST" }),
    remove: (id: number) => request<void>(`/vision/classification/jobs/${id}`, { method: "DELETE" }),
    exportModel: (id: number) =>
      downloadModelExport(`/vision/classification/jobs/${id}/model/export`, `vision_classification_job${id}.pt`),
    exportDeploymentScript: (id: number) =>
      downloadModelExport(
        `/vision/classification/jobs/${id}/model/export-script`,
        `vision_classification_job${id}_deploiement.py`,
      ),
  },

  visionAnomalies: {
    models: () => request<VisionAnomalyModelOption[]>("/vision/anomalies/models"),
    createJob: (data: VisionAnomalyJobCreatePayload, idempotencyKey?: string) =>
      request<VisionAnomalyJobSummary>("/vision/anomalies/jobs", {
        method: "POST",
        body: JSON.stringify(data),
        headers: idempotencyKeyHeader(idempotencyKey),
      }),
    listJobs: () => request<VisionAnomalyJobSummary[]>("/vision/anomalies/jobs"),
    getJob: (id: number) => request<VisionAnomalyJobSummary>(`/vision/anomalies/jobs/${id}`),
    getResult: (id: number) => request<VisionAnomalyResult>(`/vision/anomalies/jobs/${id}/result`),
    getExamples: (id: number) => request<VisionAnomalyExample[]>(`/vision/anomalies/jobs/${id}/examples`),
    predict: (id: number, file: File) =>
      uploadFileWithFields<VisionAnomalyScore>(`/vision/anomalies/jobs/${id}/predict`, file, {}),
    cancel: (id: number) => request<VisionAnomalyJobSummary>(`/vision/anomalies/jobs/${id}/cancel`, { method: "POST" }),
    rerun: (id: number) => request<VisionAnomalyJobSummary>(`/vision/anomalies/jobs/${id}/rerun`, { method: "POST" }),
    remove: (id: number) => request<void>(`/vision/anomalies/jobs/${id}`, { method: "DELETE" }),
    exportModel: (id: number) => downloadModelExport(`/vision/anomalies/jobs/${id}/model/export`, `vision_anomalies_job${id}.pt`),
    exportDeploymentScript: (id: number) =>
      downloadModelExport(`/vision/anomalies/jobs/${id}/model/export-script`, `vision_anomalies_job${id}_deploiement.py`),
    // Change le seuil de décision opérationnel (retour utilisateur, maquette
    // de refonte) — le seuil DOIT venir de `threshold_candidates` déjà
    // affiché, s'applique réellement ensuite à /predict et au script de
    // déploiement exporté.
    chooseThreshold: (id: number, threshold: number) =>
      request<VisionAnomalyResult>(`/vision/anomalies/jobs/${id}/model/threshold`, {
        method: "PATCH",
        body: JSON.stringify({ threshold }),
      }),
  },

  notifications: {
    list: (unreadOnly = false) =>
      request<NotificationEntry[]>(`/notifications${unreadOnly ? "?unread_only=true" : ""}`),
    unreadCount: () => request<{ count: number }>("/notifications/unread-count"),
    markRead: (id: number) => request<NotificationEntry>(`/notifications/${id}/read`, { method: "POST" }),
    markAllRead: () => request<void>("/notifications/read-all", { method: "POST" }),
  },
};
