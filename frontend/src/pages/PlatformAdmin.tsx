import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  Building2,
  Database,
  ShieldCheck,
  Users,
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
import { ErrorNote } from "../components/ui/ErrorNote";
import { PageHeader } from "../components/ui/PageHeader";
import { Segmented } from "../components/ui/Segmented";
import { Tabs, type TabItem } from "../components/ui/Tabs";
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
 * une page vide et incompréhensible à qui n'y a pas droit. */

type TabId = "overview" | "organizations" | "users" | "activity";

const TABS: TabItem<TabId>[] = [
  { id: "overview", label: "Vue d'ensemble", icon: Activity },
  { id: "organizations", label: "Organisations", icon: Building2 },
  { id: "users", label: "Comptes", icon: Users },
  { id: "activity", label: "Activité", icon: Database },
];

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

function StatCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card className="p-4">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-h2 text-foreground mt-1 tabular-nums">{value}</p>
      {hint && <p className="text-xs text-muted-foreground mt-1">{hint}</p>}
    </Card>
  );
}

/** Barres empilées en CSS pur plutôt qu'une dépendance de graphique : la
 * donnée est une série de comptages sur une échelle unique, un histogramme
 * complet n'apporterait rien de plus qu'il faudrait ensuite rendre
 * accessible et thématisable. Les valeurs restent lisibles en texte. */
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
            className="flex-1 min-w-[2px] rounded-t bg-primary/70"
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
  const segments: { key: string; count: number; className: string; label: string }[] = [
    { key: "completed", count: pillar.completed, className: "bg-success", label: "terminés" },
    { key: "running", count: pillar.running, className: "bg-primary", label: "en cours" },
    { key: "queued", count: pillar.queued, className: "bg-muted-foreground/50", label: "en file" },
    { key: "failed", count: pillar.failed, className: "bg-destructive", label: "en échec" },
  ];
  const total = Math.max(1, pillar.total);

  return (
    <div className="py-2.5">
      <div className="flex items-baseline justify-between mb-1.5 gap-3">
        <span className="text-sm text-foreground/90 truncate">{pillar.label}</span>
        <span className="text-xs text-muted-foreground tabular-nums flex-shrink-0">
          {pillar.total} job{pillar.total > 1 ? "s" : ""}
          {pillar.failed > 0 && <span className="text-destructive"> · {pillar.failed} en échec</span>}
        </span>
      </div>
      <div className="flex h-2 rounded-full overflow-hidden bg-muted" role="img"
        aria-label={segments.map((s) => `${s.count} ${s.label}`).join(", ")}>
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
        <StatCard label="Organisations" value={String(c.organizations)} />
        <StatCard
          label="Comptes"
          value={String(c.users_total)}
          hint={`${c.users_active} actifs · ${c.users_revoked} révoqués · ${c.users_anonymized} anonymisés`}
        />
        <StatCard
          label="Datasets"
          value={String(c.datasets + c.vision_datasets)}
          hint={`${c.datasets} tabulaires (${formatBytes(c.datasets_bytes)}) · ${c.vision_datasets} vision`}
        />
        <StatCard
          label="Jobs lancés"
          value={String(data.jobs_total)}
          hint={
            data.failure_rate === null
              ? "aucun job terminé — rien à mesurer"
              : `${(data.failure_rate * 100).toFixed(1)} % d'échec sur les jobs terminés`
          }
        />
        <StatCard label="Modèles produits" value={String(c.models)} />
        <StatCard label="Prédictions servies" value={String(c.predictions)} />
        <StatCard
          label="Comptes en attente"
          value={String(c.users_pending_password)}
          hint="mot de passe provisoire non encore remplacé"
        />
        <StatCard label="Jobs en échec" value={String(data.jobs_failed)} />
      </div>

      <Card className="p-5">
        <div className="flex items-center justify-between mb-2 gap-3">
          <h2 className="text-h3 text-foreground">Activité par pilier</h2>
          <span className="text-xs text-muted-foreground">terminés · en cours · en file · en échec</span>
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

  if (error) return <ErrorNote message={error} reference={errorRef} />;
  if (!data) return <p className="text-sm text-muted-foreground">Chargement…</p>;

  return (
    <Card className="p-0 overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">Organisations de la plateforme et leur volumétrie</caption>
        <thead className="text-xs text-muted-foreground border-b border-border">
          <tr>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Organisation</th>
            <th scope="col" className="text-right font-medium px-4 py-2.5">Membres</th>
            <th scope="col" className="text-right font-medium px-4 py-2.5">Datasets</th>
            <th scope="col" className="text-right font-medium px-4 py-2.5">Jobs</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Créée le</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Dernière activité</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {data.map((org) => (
            <tr key={org.id}>
              <td className="px-4 py-2.5 text-foreground/90">{org.name}</td>
              <td className="px-4 py-2.5 text-right tabular-nums">
                {org.active_members}
                {org.members !== org.active_members && (
                  <span className="text-muted-foreground"> / {org.members}</span>
                )}
              </td>
              <td className="px-4 py-2.5 text-right tabular-nums">{org.datasets}</td>
              <td className="px-4 py-2.5 text-right tabular-nums">{org.jobs}</td>
              <td className="px-4 py-2.5 text-muted-foreground">{formatDateTime(org.created_at)}</td>
              <td className="px-4 py-2.5 text-muted-foreground">
                {org.last_activity_at ? formatDateTime(org.last_activity_at) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.length === 0 && <p className="text-sm text-muted-foreground p-4">Aucune organisation.</p>}
    </Card>
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

  if (error) return <ErrorNote message={error} reference={errorRef} />;
  if (!data) return <p className="text-sm text-muted-foreground">Chargement…</p>;

  return (
    <Card className="p-0 overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">Tous les comptes de la plateforme, toutes organisations confondues</caption>
        <thead className="text-xs text-muted-foreground border-b border-border">
          <tr>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Compte</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Organisation</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Rôle</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">État</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Dernière connexion</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {data.map((row) => (
            <tr key={row.id}>
              <td className="px-4 py-2.5">
                <div className="text-foreground/90">{row.nom}</div>
                <div className="text-xs text-muted-foreground">{row.email}</div>
              </td>
              <td className="px-4 py-2.5 text-muted-foreground">{row.organization_name}</td>
              <td className="px-4 py-2.5">
                <div className="flex items-center gap-1.5">
                  <Badge variant={row.role === "owner" ? "accent" : "neutral"}>
                    {row.role === "owner" ? "Propriétaire" : "Membre"}
                  </Badge>
                  {row.is_platform_admin && <Badge variant="primary">Plateforme</Badge>}
                </div>
              </td>
              <td className="px-4 py-2.5"><AccountState user={row} /></td>
              <td className="px-4 py-2.5 text-muted-foreground">
                {row.last_login ? formatDateTime(row.last_login) : "jamais"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function ActivityTab() {
  const { data, error, errorRef } = useAdminResource<PlatformAuditRow[]>(() => api.admin.activity(150));

  if (error) return <ErrorNote message={error} reference={errorRef} />;
  if (!data) return <p className="text-sm text-muted-foreground">Chargement…</p>;

  return (
    <Card className="p-0 overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">Journal d'activité de toutes les organisations</caption>
        <thead className="text-xs text-muted-foreground border-b border-border">
          <tr>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Quand</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Organisation</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Auteur</th>
            <th scope="col" className="text-left font-medium px-4 py-2.5">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {data.map((entry) => (
            <tr key={entry.id}>
              <td className="px-4 py-2.5 text-muted-foreground whitespace-nowrap">
                {formatDateTime(entry.created_at)}
              </td>
              <td className="px-4 py-2.5 text-muted-foreground">{entry.organization_name}</td>
              <td className="px-4 py-2.5 text-foreground/90">{entry.actor_name ?? "—"}</td>
              <td className="px-4 py-2.5">
                <span className="font-mono text-xs text-foreground/90">{entry.action}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.length === 0 && <p className="text-sm text-muted-foreground p-4">Aucune activité enregistrée.</p>}
    </Card>
  );
}
