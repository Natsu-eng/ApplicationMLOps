import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  Boxes,
  Brain,
  Building2,
  Database,
  Eye,
  Gauge,
  KeyRound,
  Layers,
  ListChecks,
  Radar,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  Users,
  type LucideIcon,
} from "lucide-react";

import {
  ApiError,
  api,
  apiErrorReference,
  type JobsByPillar,
  type OrganizationRow,
  type PlatformAuditRow,
  type PlatformOverview,
  type PlatformUserRow,
  type TimeseriesPoint,
} from "../api/client";
import AppShell from "../components/AppShell";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, type AccentColor } from "../components/ui/ColorIconBadge";
import { ErrorNote } from "../components/ui/ErrorNote";
import { PageHeader } from "../components/ui/PageHeader";
import { Segmented } from "../components/ui/Segmented";
import { Table, type TableColumn } from "../components/ui/Table";
import { Tabs, type TabItem } from "../components/ui/Tabs";
import { pillarColor } from "../config/pillars";
import { useAuth } from "../contexts/AuthContext";
import { formatDateTime } from "../utils/format";

/** Espace d'administration de la PLATEFORME — vue globale de l'éditeur.
 *
 * Miroir du router `domains/admin` : lecture seule, et seul endroit de
 * l'application qui affiche des données de plusieurs organisations. Le reste
 * de l'interface ne montre jamais que l'organisation de l'utilisateur
 * connecté, y compris pour un administrateur plateforme — la supervision
 * n'est pas un passe-partout.
 *
 * La page ne se protège pas elle-même : le serveur renvoie 403 sur chaque
 * route `/admin` à un compte ordinaire. Ce qui suit sert à ne pas afficher
 * une page vide et incompréhensible à qui n'y a pas droit.
 *
 * COULEUR : suit la règle posée dans `config/pillars.ts` — une donnée qui
 * n'appartient à AUCUN pilier reste `neutral`, jamais une teinte de pilier
 * empruntée pour faire joli. Les seules couleurs vives ici portent donc un
 * sens : l'identité d'un pilier réel (`pillarColor`), ou un statut
 * (`rose` = échec, `amber` = attente), et uniquement quand la valeur
 * concernée est non nulle — un compteur à zéro n'est pas une alerte. */

type TabId = "overview" | "organizations" | "users" | "activity";

const TABS: TabItem<TabId>[] = [
  { id: "overview", label: "Vue d'ensemble", icon: Activity },
  { id: "organizations", label: "Organisations", icon: Building2 },
  { id: "users", label: "Comptes", icon: Users },
  { id: "activity", label: "Activité", icon: Database },
];

/** Les 7 types de job du backend se répartissent sur les 3 piliers PRODUIT.
 * La couleur vient donc de `pillarColor()` — source unique — plutôt que
 * d'une teinte choisie ici, et l'icône distingue les types au sein d'un
 * même pilier. */
const JOB_PILLARS: Record<string, { icon: LucideIcon; color: AccentColor }> = {
  TrainingJob: { icon: Brain, color: pillarColor("supervised") },
  BatchPredictionJob: { icon: Layers, color: pillarColor("supervised") },
  ClusteringJob: { icon: Boxes, color: pillarColor("unsupervised") },
  DimensionalityJob: { icon: Radar, color: pillarColor("unsupervised") },
  AnomalyJob: { icon: ScanSearch, color: pillarColor("unsupervised") },
  VisionClassificationJob: { icon: Eye, color: pillarColor("vision") },
  VisionAnomalyJob: { icon: TriangleAlert, color: pillarColor("vision") },
};

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 o";
  const units = ["o", "Ko", "Mo", "Go", "To"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value >= 100 || exponent === 0 ? Math.round(value) : value.toFixed(1)} ${units[exponent]}`;
}

export default function PlatformAdmin() {
  const { user } = useAuth();
  const [tab, setTab] = useState<TabId>("overview");

  if (!user) return null;

  if (!user.is_platform_admin) {
    return (
      <AppShell>
        <Card className="p-6 max-w-xl">
          <h1 className="text-h3 text-foreground mb-2">Espace réservé</h1>
          <p className="text-sm text-muted-foreground">
            Cet espace est réservé à l'administration de la plateforme. Votre compte gère votre
            organisation, ce qui est un rôle différent — rendez-vous dans Profil &amp; Organisation.
          </p>
        </Card>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <PageHeader
        eyebrow="Plateforme"
        title="Administration"
        description="Vue globale de toutes les organisations. Lecture seule : superviser n'est pas agir sur les données d'un client."
        icon={ShieldCheck}
        color="violet"
      />
      <div className="mb-5">
        <Tabs items={TABS} active={tab} onChange={setTab} />
      </div>

      {tab === "overview" && <OverviewTab />}
      {tab === "organizations" && <OrganizationsTab />}
      {tab === "users" && <UsersTab />}
      {tab === "activity" && <ActivityTab />}
    </AppShell>
  );
}

/** Chargement + erreur mutualisés : chaque onglet fait un appel et rien
 * d'autre, inutile de recopier quatre fois le même échafaudage. */
function useAdminResource<T>(load: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorRef, setErrorRef] = useState<string | undefined>(undefined);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(load, deps);

  useEffect(() => {
    let cancelled = false;
    run()
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setError(null);
        setErrorRef(undefined);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Chargement impossible");
        setErrorRef(apiErrorReference(err));
      });
    return () => {
      cancelled = true;
    };
  }, [run]);

  return { data, error, errorRef };
}

function StatCard({
  label,
  value,
  hint,
  icon,
  color = "neutral",
}: {
  label: string;
  value: string;
  hint?: string;
  icon: LucideIcon;
  color?: AccentColor;
}) {
  return (
    <Card className="group p-4 flex items-start gap-3">
      <ColorIconBadge icon={icon} color={color} />
      <div className="min-w-0">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-h2 text-foreground leading-tight tabular-nums">{value}</p>
        {hint && <p className="text-xs text-muted-foreground mt-0.5">{hint}</p>}
      </div>
    </Card>
  );
}

/** Barres empilées en CSS pur plutôt qu'une dépendance de graphique : la
 * donnée est une série de comptages sur une échelle unique, un histogramme
 * complet n'apporterait rien qu'il faudrait ensuite rendre accessible et
 * thématisable. Les valeurs restent lisibles en texte. */
function Sparkbars({ points, label }: { points: TimeseriesPoint[]; label: string }) {
  const max = Math.max(1, ...points.map((p) => p.count));
  const total = points.reduce((sum, p) => sum + p.count, 0);

  return (
    <Card className="p-4">
      <div className="flex items-baseline justify-between mb-3">
        <p className="text-sm font-medium text-foreground">{label}</p>
        <p className="text-xs text-muted-foreground tabular-nums">{total} au total</p>
      </div>
      <div className="flex items-end gap-[2px] h-24" role="img" aria-label={`${label} : ${total} au total`}>
        {points.map((point) => (
          <div
            key={point.date}
            className={`flex-1 min-w-[2px] rounded-t transition-colors ${
              point.count > 0 ? "bg-primary/70 hover:bg-primary" : "bg-muted"
            }`}
            style={{ height: `${Math.max(2, (point.count / max) * 100)}%` }}
            title={`${point.date} — ${point.count}`}
          />
        ))}
      </div>
      <div className="flex justify-between mt-1.5 text-[10px] text-muted-foreground">
        <span>{points[0]?.date ?? ""}</span>
        <span>{points[points.length - 1]?.date ?? ""}</span>
      </div>
    </Card>
  );
}

function PillarRow({ pillar }: { pillar: JobsByPillar }) {
  const identity = JOB_PILLARS[pillar.pillar] ?? { icon: Gauge, color: "neutral" as AccentColor };
  const segments = [
    { key: "completed", count: pillar.completed, className: "bg-success", label: "terminés" },
    { key: "running", count: pillar.running, className: "bg-primary", label: "en cours" },
    { key: "queued", count: pillar.queued, className: "bg-muted-foreground/50", label: "en file" },
    { key: "failed", count: pillar.failed, className: "bg-destructive", label: "en échec" },
  ];
  const total = Math.max(1, pillar.total);

  return (
    <div className="group py-3 flex items-center gap-3">
      <ColorIconBadge icon={identity.icon} color={identity.color} size="sm" />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between mb-1.5 gap-3">
          <span className="text-sm text-foreground/90 truncate">{pillar.label}</span>
          <span className="text-xs text-muted-foreground tabular-nums flex-shrink-0">
            {pillar.total} job{pillar.total > 1 ? "s" : ""}
            {pillar.failed > 0 && <span className="text-destructive"> · {pillar.failed} en échec</span>}
          </span>
        </div>
        <div
          className="flex h-2 rounded-full overflow-hidden bg-muted"
          role="img"
          aria-label={segments.map((s) => `${s.count} ${s.label}`).join(", ")}
        >
          {segments.map((segment) =>
            segment.count > 0 ? (
              <div
                key={segment.key}
                className={segment.className}
                style={{ width: `${(segment.count / total) * 100}%` }}
                title={`${segment.count} ${segment.label}`}
              />
            ) : null,
          )}
        </div>
      </div>
    </div>
  );
}

function OverviewTab() {
  const [windowDays, setWindowDays] = useState(30);
  const { data, error, errorRef } = useAdminResource<PlatformOverview>(
    () => api.admin.overview(windowDays),
    [windowDays],
  );

  if (error) return <ErrorNote message={error} reference={errorRef} />;
  if (!data) return <p className="text-sm text-muted-foreground">Chargement…</p>;

  const c = data.counters;
  return (
    <div className="grid gap-5">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard icon={Building2} label="Organisations" value={String(c.organizations)} />
        <StatCard
          icon={Users}
          label="Comptes"
          value={String(c.users_total)}
          hint={`${c.users_active} actifs · ${c.users_revoked} révoqués · ${c.users_anonymized} anonymisés`}
        />
        <StatCard
          icon={Database}
          label="Datasets"
          value={String(c.datasets + c.vision_datasets)}
          hint={`${c.datasets} tabulaires (${formatBytes(c.datasets_bytes)}) · ${c.vision_datasets} vision`}
        />
        <StatCard
          icon={ListChecks}
          label="Jobs lancés"
          value={String(data.jobs_total)}
          hint={
            data.failure_rate === null
              ? "aucun job terminé — rien à mesurer"
              : `${(data.failure_rate * 100).toFixed(1)} % d'échec sur les jobs terminés`
          }
        />
        <StatCard icon={Sparkles} label="Modèles produits" value={String(c.models)} />
        <StatCard icon={Gauge} label="Prédictions servies" value={String(c.predictions)} />
        {/* Couleur portée par la VALEUR, pas par la carte : un compteur à
            zéro n'est pas une alerte et ne doit pas s'afficher comme telle. */}
        <StatCard
          icon={KeyRound}
          label="Comptes en attente"
          value={String(c.users_pending_password)}
          hint="mot de passe provisoire non encore remplacé"
          color={c.users_pending_password > 0 ? "amber" : "neutral"}
        />
        <StatCard
          icon={TriangleAlert}
          label="Jobs en échec"
          value={String(data.jobs_failed)}
          color={data.jobs_failed > 0 ? "rose" : "neutral"}
        />
      </div>

      <Card className="p-5">
        <div className="flex items-center justify-between mb-2 gap-3 flex-wrap">
          <h2 className="text-h3 text-foreground">Activité par pilier</h2>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-success" />terminés</span>
            <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-primary" />en cours</span>
            <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-muted-foreground/50" />en file</span>
            <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-destructive" />en échec</span>
          </div>
        </div>
        <div className="divide-y divide-border">
          {data.jobs_by_pillar.map((pillar) => (
            <PillarRow key={pillar.pillar} pillar={pillar} />
          ))}
        </div>
      </Card>

      <div className="flex items-center justify-end">
        <Segmented
          aria-label="Fenêtre d'observation"
          size="sm"
          value={String(windowDays)}
          onChange={(id) => setWindowDays(Number(id))}
          options={[
            { id: "7", label: "7 j" },
            { id: "30", label: "30 j" },
            { id: "90", label: "90 j" },
          ]}
        />
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        <Sparkbars points={data.jobs_per_day} label="Actions enregistrées par jour" />
        <Sparkbars points={data.signups_per_day} label="Comptes créés par jour" />
      </div>
    </div>
  );
}

function OrganizationsTab() {
  const { data, error, errorRef } = useAdminResource<OrganizationRow[]>(() => api.admin.organizations());

  const columns: TableColumn<OrganizationRow>[] = [
    { key: "name", header: "Organisation", sticky: true, sortable: true },
    {
      key: "members",
      header: "Membres",
      align: "right",
      sortable: true,
      sortValue: (r) => r.active_members,
      render: (r) => (
        <span className="tabular-nums">
          {r.active_members}
          {r.members !== r.active_members && <span className="text-muted-foreground"> / {r.members}</span>}
        </span>
      ),
    },
    { key: "datasets", header: "Datasets", align: "right", sortable: true },
    { key: "jobs", header: "Jobs", align: "right", sortable: true },
    {
      key: "created_at",
      header: "Créée le",
      sortable: true,
      render: (r) => <span className="text-muted-foreground">{formatDateTime(r.created_at)}</span>,
    },
    {
      key: "last_activity_at",
      header: "Dernière activité",
      sortable: true,
      sortValue: (r) => r.last_activity_at ?? "",
      render: (r) => (
        <span className="text-muted-foreground">
          {r.last_activity_at ? formatDateTime(r.last_activity_at) : "—"}
        </span>
      ),
    },
  ];

  if (error) return <ErrorNote message={error} reference={errorRef} />;
  return (
    <Table
      columns={columns}
      rows={data ?? []}
      rowKey={(r) => r.id}
      loading={data === null}
      caption="Organisations de la plateforme, avec leur volumétrie et leur dernière activité"
      emptyMessage="Aucune organisation."
      pageSize={15}
    />
  );
}

/** État d'un compte en un seul badge : les quatre situations s'excluent, et
 * l'ordre de test importe — un compte anonymisé est aussi révoqué, mais
 * c'est l'effacement qui le décrit. */
function AccountState({ user }: { user: PlatformUserRow }) {
  if (user.anonymized_at) return <Badge variant="neutral">Identité effacée</Badge>;
  if (!user.actif) return <Badge variant="danger">Accès révoqué</Badge>;
  if (user.must_change_password) return <Badge variant="warning">En attente</Badge>;
  return <Badge variant="success">Actif</Badge>;
}

function UsersTab() {
  const { data, error, errorRef } = useAdminResource<PlatformUserRow[]>(() => api.admin.users());

  const columns: TableColumn<PlatformUserRow>[] = [
    {
      key: "nom",
      header: "Compte",
      sticky: true,
      sortable: true,
      render: (r) => (
        <div className="min-w-0">
          <div className="text-foreground/90 truncate">{r.nom}</div>
          <div className="text-xs text-muted-foreground truncate">{r.email}</div>
        </div>
      ),
    },
    {
      key: "organization_name",
      header: "Organisation",
      sortable: true,
      render: (r) => <span className="text-muted-foreground">{r.organization_name}</span>,
    },
    {
      key: "role",
      header: "Rôle",
      sortable: true,
      render: (r) => (
        <div className="flex items-center gap-1.5">
          <Badge variant={r.role === "owner" ? "accent" : "neutral"}>
            {r.role === "owner" ? "Propriétaire" : "Membre"}
          </Badge>
          {r.is_platform_admin && <Badge variant="primary">Plateforme</Badge>}
        </div>
      ),
    },
    {
      key: "etat",
      header: "État",
      sortable: true,
      sortValue: (r) => (r.anonymized_at ? 3 : !r.actif ? 2 : r.must_change_password ? 1 : 0),
      render: (r) => <AccountState user={r} />,
    },
    {
      key: "last_login",
      header: "Dernière connexion",
      sortable: true,
      sortValue: (r) => r.last_login ?? "",
      render: (r) => (
        <span className="text-muted-foreground">{r.last_login ? formatDateTime(r.last_login) : "jamais"}</span>
      ),
    },
  ];

  if (error) return <ErrorNote message={error} reference={errorRef} />;
  return (
    <Table
      columns={columns}
      rows={data ?? []}
      rowKey={(r) => r.id}
      loading={data === null}
      caption="Tous les comptes de la plateforme, toutes organisations confondues"
      emptyMessage="Aucun compte."
      pageSize={20}
    />
  );
}

function ActivityTab() {
  const { data, error, errorRef } = useAdminResource<PlatformAuditRow[]>(() => api.admin.activity(150));

  const columns: TableColumn<PlatformAuditRow>[] = [
    {
      key: "created_at",
      header: "Quand",
      sticky: true,
      sortable: true,
      render: (r) => (
        <span className="text-muted-foreground whitespace-nowrap">{formatDateTime(r.created_at)}</span>
      ),
    },
    {
      key: "organization_name",
      header: "Organisation",
      sortable: true,
      render: (r) => <span className="text-muted-foreground">{r.organization_name}</span>,
    },
    {
      key: "actor_name",
      header: "Auteur",
      sortable: true,
      sortValue: (r) => r.actor_name ?? "",
      render: (r) => <span className="text-foreground/90">{r.actor_name ?? "—"}</span>,
    },
    {
      key: "action",
      header: "Action",
      sortable: true,
      render: (r) => <span className="font-mono text-xs text-foreground/90">{r.action}</span>,
    },
  ];

  if (error) return <ErrorNote message={error} reference={errorRef} />;
  return (
    <Table
      columns={columns}
      rows={data ?? []}
      rowKey={(r) => r.id}
      loading={data === null}
      caption="Journal d'audit de toutes les organisations, du plus récent au plus ancien"
      emptyMessage="Aucune activité enregistrée."
      pageSize={25}
    />
  );
}
