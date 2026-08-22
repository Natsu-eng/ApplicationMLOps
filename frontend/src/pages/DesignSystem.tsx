import { useState } from "react";
import {
  AlertTriangle,
  Bell,
  Check,
  CircleCheck,
  Download,
  Grid3x3,
  LayoutList,
  Palette,
  Rows3,
  Sparkles,
  Trash2,
} from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import AppShell from "../components/AppShell";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Field } from "../components/ui/Field";
import { Input } from "../components/ui/Input";
import { Select } from "../components/ui/Select";
import { Table, type TableColumn } from "../components/ui/Table";
import { Tabs, type TabItem } from "../components/ui/Tabs";
import { Badge } from "../components/ui/Badge";
import { JobStatusBadge } from "../components/ui/StatusBadge";
import { StatTile, StatTileRow } from "../components/dashboard/StatTile";
import { ChartSkeleton, EmptyChartState } from "../components/ui/ChartStates";
import { IsolatableLegend, useSeriesIsolation } from "../components/ui/ChartLegend";
import { CHART_GRID_STROKE, CHART_HEIGHT_SM, CHART_SERIES_COLORS, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../theme/charts";
import type { Density } from "../theme/density";
import { DENSITY_LABELS } from "../theme/density";
import { Skeleton, SkeletonGroup } from "../components/ui/Skeleton";
import { ProgressBar, IndeterminateProgressBar } from "../components/ui/ProgressBar";
import { Segmented } from "../components/ui/Segmented";
import { Alert } from "../components/ui/Alert";
import { Stepper } from "../components/ui/Stepper";
import { Popover } from "../components/ui/Popover";
import { Breadcrumb } from "../components/ui/Breadcrumb";
import { useToast } from "../components/ui/Toast";
import { Drawer } from "../components/ui/Drawer";

/** Page de style guide (Lot 2A, AUDIT_DATALAB_2026-08-16.md §J.4) —
 * protégée, jamais liée depuis la navigation (accès direct par URL
 * uniquement). Montre tous les tokens et primitives reconstruits ce lot,
 * en clair ET en sombre (bascule déjà disponible page Profil). Aucune page
 * métier n'est modifiée par ce lot — cette page est la seule surface
 * neuve. */
export default function DesignSystem() {
  return (
    <AppShell>
      <PageHeader
        eyebrow="Système de conception"
        title="Fondation visuelle — Lot 2A"
        description="Tokens, échelles et primitives reconstruits. Page de référence, non liée depuis la navigation — vérifiez ce que vous voyez ici en clair et en sombre (Profil > Préférences) avant de valider."
        icon={Palette}
        color="violet"
      />
      <div className="space-y-10 pb-16">
        <ColorSection />
        <TypographySection />
        <SpacingRadiusElevationSection />
        <MotionSection />
        <ButtonSection />
        <FieldSection />
        <CardSection />
        <TabsSection />
        <StatTileSection />
        <TableSection />
        <ChartSection />
        <BadgeSection />
        <SkeletonSection />
        <ProgressBarSection />
        <SegmentedSection />
        <AlertSection />
        <StepperSection />
        <PopoverBreadcrumbSection />
        <ToastDrawerCommandSection />
      </div>
    </AppShell>
  );
}

function Section({ title, description, children }: { title: string; description?: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-h2 text-foreground">{title}</h2>
        {description && <p className="text-body text-muted-foreground mt-1 max-w-2xl">{description}</p>}
      </div>
      {children}
    </section>
  );
}

// ── Couleurs ─────────────────────────────────────────────────────────────

const NEUTRAL_SWATCHES = [
  { name: "background", var: "--color-background" },
  { name: "card", var: "--color-card" },
  { name: "border", var: "--color-border" },
  { name: "muted", var: "--color-muted" },
  { name: "muted-foreground", var: "--color-muted-foreground" },
  { name: "foreground", var: "--color-foreground" },
];

const SEMANTIC_SWATCHES = [
  { name: "primary", contrastLight: "6.70:1", contrastDark: "4.70:1" },
  { name: "info", contrastLight: "5.42:1", contrastDark: "4.69:1" },
  { name: "success", contrastLight: "4.82:1", contrastDark: "4.71:1" },
  { name: "warning", contrastLight: "4.82:1", contrastDark: "4.77:1" },
  { name: "destructive", contrastLight: "4.76:1", contrastDark: "4.65:1" },
];

// Classes littérales (jamais construites par interpolation de chaîne) — Tailwind
// v4 ne génère la variable CSS d'un jeton de thème QUE s'il détecte une classe
// qui l'utilise, littéralement, dans le code source scanné. `var(--color-${x}-solid)`
// en style inline n'est PAS détecté par ce scanner : --color-primary-solid/
// -warning-solid/-success-solid/-info-solid n'étaient jamais générés, le fond
// retombait silencieusement sur celui de la carte ambiante derrière (contraste
// ~1:1 avec le texte dessus — trouvé par axe-core, Lot 2, voir _design/JOURNAL.md).
// Seul --color-destructive-solid existait, par accident : Button.tsx l'utilise
// littéralement (variant="destructive").
const SOLID_SWATCH_CLASSES: Record<string, string> = {
  primary: "bg-primary-solid text-primary-foreground",
  info: "bg-info-solid text-info-foreground",
  success: "bg-success-solid text-success-foreground",
  warning: "bg-warning-solid text-warning-foreground",
  destructive: "bg-destructive-solid text-destructive-foreground",
};

function ColorSection() {
  return (
    <Section
      title="Couleurs"
      description="La couleur est un outil de sens (statut, série, sévérité), jamais une décoration. Tous les couples texte/fond ci-dessous atteignent 4.5:1 (AA, texte normal) — vérifié par le calcul (OKLCH → sRGB → luminance relative WCAG), pas à l'œil. Détail dans DECISIONS.md, D2.1."
    >
      <Card>
        <p className="text-overline uppercase text-muted-foreground mb-3">Neutre</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
          {NEUTRAL_SWATCHES.map((s) => (
            <div key={s.name} className="space-y-1.5">
              <div className="h-14 rounded-control border border-border" style={{ background: `var(${s.var})` }} />
              <p className="text-caption text-foreground font-medium">{s.name}</p>
            </div>
          ))}
        </div>

        <p className="text-overline uppercase text-muted-foreground mb-3">Sémantique — texte sur carte</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
          {SEMANTIC_SWATCHES.map((s) => (
            <div key={s.name} className="space-y-1.5">
              <div
                className="h-14 rounded-control border border-border flex items-center justify-center text-body font-semibold"
                style={{ color: `var(--color-${s.name})`, background: "var(--color-card)" }}
              >
                Aa
              </div>
              <p className="text-caption text-foreground font-medium">{s.name}</p>
              <p className="text-caption text-muted-foreground tabular-nums">
                clair {s.contrastLight} · sombre {s.contrastDark}
              </p>
            </div>
          ))}
        </div>

        <p className="text-overline uppercase text-muted-foreground mb-3">Sémantique — remplissage plein ("-solid") + texte clair</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
          {SEMANTIC_SWATCHES.map((s) => (
            <div key={s.name} className="space-y-1.5">
              <div className={`h-14 rounded-control flex items-center justify-center text-body font-semibold ${SOLID_SWATCH_CLASSES[s.name]}`}>
                Aa
              </div>
              <p className="text-caption text-foreground font-medium">{s.name}-solid</p>
            </div>
          ))}
        </div>

        <p className="text-overline uppercase text-muted-foreground mb-3">Palette catégorielle (graphiques)</p>
        <div className="flex flex-wrap gap-3">
          {CHART_SERIES_COLORS.map((c, i) => (
            <div key={c} className="space-y-1.5">
              <div className="h-10 w-16 rounded-control border border-border" style={{ background: c }} />
              <p className="text-caption text-muted-foreground text-center">série {i + 1}</p>
            </div>
          ))}
        </div>
      </Card>
    </Section>
  );
}

// ── Typographie ──────────────────────────────────────────────────────────

const TYPE_ROLES = [
  { cls: "text-h1", name: "h1", usage: "Titre de page" },
  { cls: "text-h2", name: "h2", usage: "Titre de section" },
  { cls: "text-h3", name: "h3", usage: "En-tête de carte, sous-section" },
  { cls: "text-body", name: "body", usage: "Texte courant" },
  { cls: "text-caption", name: "caption", usage: "Aide, méta-donnée" },
  { cls: "text-overline", name: "overline", usage: "Étiquette majuscule, en-tête de colonne" },
] as const;

function TypographySection() {
  return (
    <Section
      title="Typographie"
      description="6 rôles nommés. Les chiffres de métrique s'écrivent toujours avec tabular-nums en plus du rôle — jamais d'exception (voir la tuile ci-dessous)."
    >
      <Card className="divide-y divide-border/60">
        {TYPE_ROLES.map((r) => (
          <div key={r.name} className="py-3 first:pt-0 last:pb-0 flex items-baseline justify-between gap-4 flex-wrap">
            <p className={`${r.cls} ${r.name === "overline" ? "uppercase" : ""} text-foreground`}>
              {r.name === "overline" ? "Étiquette de colonne" : "Aperçu du rôle typographique"}
            </p>
            <p className="text-caption text-muted-foreground whitespace-nowrap">
              <code className="text-foreground">{r.cls}</code> — {r.usage}
            </p>
          </div>
        ))}
        <div className="pt-3">
          <p className="text-caption text-muted-foreground mb-1">Chiffre de métrique (tabular-nums obligatoire)</p>
          <p className="text-h1 text-foreground tabular-nums">1 284,5</p>
        </div>
      </Card>
    </Section>
  );
}

// ── Espacement, densité, rayons, élévation ───────────────────────────────

function SpacingRadiusElevationSection() {
  return (
    <Section
      title="Espacement, densité, rayons, élévation"
      description="Échelle Tailwind par défaut (base 4px) — déjà un système 4pt, la règle est de s'y tenir, pas d'en inventer un nouveau. 3 densités pour Table/Card, 3 rayons, 3 élévations maximum, usage prescrit."
    >
      <Card className="space-y-6">
        <div>
          <p className="text-overline uppercase text-muted-foreground mb-3">Rayons</p>
          <div className="flex flex-wrap gap-4">
            <div className="space-y-1.5">
              <div className="h-14 w-24 bg-muted rounded-control" />
              <p className="text-caption text-muted-foreground">control (0.5rem)</p>
            </div>
            <div className="space-y-1.5">
              <div className="h-14 w-24 bg-muted rounded-card" />
              <p className="text-caption text-muted-foreground">card (0.75rem)</p>
            </div>
            <div className="space-y-1.5">
              <div className="h-14 w-24 bg-muted rounded-sheet" />
              <p className="text-caption text-muted-foreground">sheet (1rem)</p>
            </div>
          </div>
        </div>

        <div>
          <p className="text-overline uppercase text-muted-foreground mb-3">Élévation</p>
          <div className="flex flex-wrap gap-4">
            <div className="space-y-1.5">
              <div className="h-14 w-24 bg-card rounded-control shadow-control border border-border/60" />
              <p className="text-caption text-muted-foreground">control</p>
            </div>
            <div className="space-y-1.5">
              <div className="h-14 w-24 bg-card rounded-card shadow-card border border-border/60" />
              <p className="text-caption text-muted-foreground">card</p>
            </div>
            <div className="space-y-1.5">
              <div className="h-14 w-24 bg-card rounded-sheet shadow-overlay border border-border/60" />
              <p className="text-caption text-muted-foreground">overlay</p>
            </div>
          </div>
        </div>

        <DensityDemo />
      </Card>
    </Section>
  );
}

function DensityDemo() {
  const [density, setDensity] = useState<Density>("default");
  const columns: TableColumn<{ id: number; algo: string; score: number }>[] = [
    { key: "algo", header: "Algorithme" },
    { key: "score", header: "Score", align: "right", render: (r) => <span className="tabular-nums">{r.score.toFixed(3)}</span> },
  ];
  const rows = [
    { id: 1, algo: "LightGBM", score: 0.912 },
    { id: 2, algo: "XGBoost", score: 0.905 },
    { id: 3, algo: "CatBoost", score: 0.898 },
  ];
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-overline uppercase text-muted-foreground">Densité (Table/Card)</p>
        <div className="flex gap-1">
          {(["compact", "default", "spacious"] as Density[]).map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => setDensity(d)}
              className={`text-caption px-2.5 py-1 rounded-control border transition-colors ${
                density === d ? "bg-primary/10 text-primary border-primary/30" : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {DENSITY_LABELS[d]}
            </button>
          ))}
        </div>
      </div>
      <Table columns={columns} rows={rows} rowKey={(r) => r.id} density={density} caption="Exemple de densité" />
    </div>
  );
}

// ── Mouvement ────────────────────────────────────────────────────────────

function MotionSection() {
  return (
    <Section
      title="Mouvement"
      description="Deux durées sanctionnées (duration-150, duration-200), une seule courbe (ease-out). Aucune animation décorative — uniquement un retour d'état. Survolez la carte."
    >
      <Card
        interactive
        className="w-fit cursor-default"
        onMouseEnter={() => {}}
      >
        <p className="text-body text-foreground">duration-200, ease-out (survol de cette carte)</p>
      </Card>
    </Section>
  );
}

// ── Boutons ──────────────────────────────────────────────────────────────

function ButtonSection() {
  const [loading, setLoading] = useState(false);
  return (
    <Section title="Boutons" description="6 variantes, état loading intégré (remplace isSubmitting ? '…' : '…' répété partout).">
      <Card className="flex flex-wrap items-center gap-3">
        <Button variant="primary">Primaire</Button>
        <Button variant="secondary">Secondaire</Button>
        <Button variant="ghost">Discret</Button>
        <Button variant="danger">Danger (teinté)</Button>
        <Button variant="destructive">
          <Trash2 size={14} /> Destructif (plein)
        </Button>
        <Button variant="link">Lien</Button>
        <Button
          variant="primary"
          loading={loading}
          onClick={() => {
            setLoading(true);
            setTimeout(() => setLoading(false), 1500);
          }}
        >
          {loading ? "Lancement…" : "Tester le chargement"}
        </Button>
        <Button variant="secondary" size="sm">
          Petit
        </Button>
      </Card>
    </Section>
  );
}

// ── Champs de formulaire ─────────────────────────────────────────────────

function FieldSection() {
  return (
    <Section title="Formulaires" description="Field compose label + aide + erreur + requis autour d'Input/Select.">
      <Card className="grid sm:grid-cols-3 gap-4 max-w-2xl">
        <Field label="Nom du dataset" help="Visible par toute l'organisation." required>
          <Input placeholder="ex. Résistance béton" />
        </Field>
        <Field label="Colonne cible" error="Cette colonne n'existe pas dans le dataset.">
          <Select>
            <option>colonne_a</option>
            <option>colonne_b</option>
          </Select>
        </Field>
        <Field label="Algorithme (désactivé)">
          <Select disabled>
            <option>LightGBM</option>
          </Select>
        </Field>
      </Card>
    </Section>
  );
}

// ── Cartes ───────────────────────────────────────────────────────────────

function CardSection() {
  return (
    <Section title="Cartes" description="4 variantes + slots header/footer normalisés.">
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card variant="elevated">
          <p className="text-h3 text-foreground mb-1">elevated</p>
          <p className="text-caption text-muted-foreground">Défaut — bordure fine + ombre discrète.</p>
        </Card>
        <Card variant="outlined">
          <p className="text-h3 text-foreground mb-1">outlined</p>
          <p className="text-caption text-muted-foreground">Aucune ombre — listes denses.</p>
        </Card>
        <Card variant="interactive">
          <p className="text-h3 text-foreground mb-1">interactive</p>
          <p className="text-caption text-muted-foreground">Survolez — élévation + léger déplacement.</p>
        </Card>
        <Card
          variant="stat"
          header={<p className="text-h3 text-foreground">En-tête</p>}
          footer={
            <Button variant="ghost" size="sm">
              Voir le détail
            </Button>
          }
        >
          <p className="text-caption text-muted-foreground">Corps avec header/footer normalisés.</p>
        </Card>
      </div>
    </Section>
  );
}

// ── Onglets ──────────────────────────────────────────────────────────────

type DemoTab = "apercu" | "detail" | "historique";
const DEMO_TABS: TabItem<DemoTab>[] = [
  { id: "apercu", label: "Aperçu" },
  { id: "detail", label: "Détail" },
  { id: "historique", label: "Historique" },
];

function TabsSection() {
  const [active, setActive] = useState<DemoTab>("apercu");
  return (
    <Section
      title="Onglets"
      description="urlParam synchronise l'onglet actif avec l'URL (?design_tab=...) — partageable par lien. Changez d'onglet puis regardez l'URL."
    >
      <Card>
        <Tabs items={DEMO_TABS} active={active} onChange={setActive} urlParam="design_tab" />
        <p className="text-body text-muted-foreground mt-3">Onglet actif : {active}</p>
      </Card>
    </Section>
  );
}

// ── StatTile ─────────────────────────────────────────────────────────────

function StatTileSection() {
  return (
    <Section title="Tuiles de statistique" description="Tendance, chargement et lien vers la vue détaillée.">
      <StatTileRow>
        <StatTile icon={Sparkles} label="Modèles entraînés" value={128} color="blue" trend={{ direction: "up", label: "+12 %" }} />
        <StatTile icon={CircleCheck} label="Taux de succès" value={94} color="teal" trend={{ direction: "flat", label: "stable" }} />
        <StatTile icon={AlertTriangle} label="Échecs (7 jours)" value={3} color="rose" trend={{ direction: "down", label: "−2" }} />
        <StatTile icon={Bell} label="Chargement…" value={undefined} color="violet" loading />
      </StatTileRow>
    </Section>
  );
}

// ── Table ────────────────────────────────────────────────────────────────

interface DemoRow {
  id: number;
  dataset: string;
  algorithme: string;
  score: number;
  statut: "completed" | "running" | "failed" | "queued";
}

const DEMO_ROWS: DemoRow[] = [
  { id: 1, dataset: "Résistance béton", algorithme: "LightGBM", score: 0.912, statut: "completed" },
  { id: 2, dataset: "Iris", algorithme: "XGBoost", score: 0.98, statut: "completed" },
  { id: 3, dataset: "Défauts soudure", algorithme: "CatBoost", score: 0.734, statut: "running" },
  { id: 4, dataset: "Résistance béton", algorithme: "RandomForest", score: 0.887, statut: "failed" },
  { id: 5, dataset: "Iris", algorithme: "LightGBM", score: 0.97, statut: "queued" },
];

function TableSection() {
  const [selected, setSelected] = useState<Set<string | number>>(new Set());
  const [showEmpty, setShowEmpty] = useState(false);
  const [showLoading, setShowLoading] = useState(false);
  const [showError, setShowError] = useState(false);

  const columns: TableColumn<DemoRow>[] = [
    { key: "dataset", header: "Dataset", sticky: true, sortable: true },
    { key: "algorithme", header: "Algorithme", sortable: true },
    {
      key: "score",
      header: "Score",
      align: "right",
      sortable: true,
      sortValue: (r) => r.score,
      render: (r) => <span className="tabular-nums">{r.score.toFixed(3)}</span>,
    },
    { key: "statut", header: "Statut", render: (r) => <JobStatusBadge status={r.statut} /> },
  ];

  return (
    <Section
      title="Tableau"
      description="Tri, pagination, sélection avec liseré d'accent, colonne figée (Dataset), en-tête collant, ligne repliable, état vide/erreur avec action, squelette de chargement — le composant à plus fort effet de levier de ce lot."
    >
      <Card>
        <div className="flex flex-wrap gap-2 mb-3">
          <Button variant="ghost" size="sm" onClick={() => setShowEmpty((v) => !v)}>
            {showEmpty ? "Afficher les données" : "Voir l'état vide"}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setShowLoading((v) => !v)}>
            {showLoading ? "Arrêter le squelette" : "Voir le squelette"}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setShowError((v) => !v)}>
            {showError ? "Revenir aux données" : "Voir l'état d'erreur"}
          </Button>
        </div>
        <Table
          caption="Entraînements récents (démonstration)"
          columns={columns}
          rows={showEmpty ? [] : DEMO_ROWS}
          rowKey={(r) => r.id}
          pageSize={3}
          selectable
          selectedKeys={selected}
          onSelectionChange={setSelected}
          loading={showLoading}
          error={showError ? { message: "Impossible de charger les entraînements.", action: <Button size="sm" onClick={() => setShowError(false)}>Réessayer</Button> } : undefined}
          highlightRow={(r) => r.score > 0.95}
          emptyMessage="Aucun entraînement pour l'instant."
          emptyAction={<Button size="sm">Lancer un entraînement</Button>}
          renderExpanded={(r) => (
            <p className="text-caption text-muted-foreground">
              Détail (ligne repliable) — dataset « {r.dataset} », algorithme {r.algorithme}, score {r.score.toFixed(3)}.
            </p>
          )}
        />
        {selected.size > 0 && (
          <p className="text-caption text-muted-foreground mt-2">{selected.size} ligne(s) sélectionnée(s)</p>
        )}
      </Card>
    </Section>
  );
}

// ── Graphiques ───────────────────────────────────────────────────────────

function ChartSection() {
  const isolation = useSeriesIsolation();
  const data = Array.from({ length: 10 }).map((_, i) => ({
    epoch: i + 1,
    "Série A": 0.5 + i * 0.04 + Math.sin(i) * 0.03,
    "Série B": 0.4 + i * 0.03,
  }));

  return (
    <Section
      title="Graphiques"
      description="Hauteurs normalisées (CHART_HEIGHT_SM/MD/LG), état vide, squelette de chargement, légende isolable réutilisable (extraite d'EvaluationCharts.tsx)."
    >
      <div className="grid lg:grid-cols-3 gap-4">
        <Card>
          <SectionHeader icon={Sparkles} color="blue" label="Avec légende isolable" />
          <ResponsiveContainer width="100%" height={CHART_HEIGHT_SM}>
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} vertical={false} />
              <XAxis dataKey="epoch" tick={CHART_TICK_STYLE} />
              <YAxis tick={CHART_TICK_STYLE} />
              <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
              <IsolatableLegend isolation={isolation} />
              {["Série A", "Série B"].map((name, i) => (
                <Line
                  key={name}
                  dataKey={name}
                  stroke={CHART_SERIES_COLORS[i]}
                  dot={false}
                  strokeOpacity={isolation.hidden.has(name) ? 0 : 1}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card>
          <SectionHeader icon={Sparkles} color="teal" label="État vide" />
          <EmptyChartState height={CHART_HEIGHT_SM} message="Pas encore de données pour ce graphique." />
        </Card>
        <Card>
          <SectionHeader icon={Sparkles} color="amber" label="Squelette de chargement" />
          <ChartSkeleton height={CHART_HEIGHT_SM} />
        </Card>
      </div>
    </Section>
  );
}

// ── Badges ───────────────────────────────────────────────────────────────

function BadgeSection() {
  return (
    <Section title="Badges" description="Variante dot pulsante pour un statut réellement en cours — remplace un spinner par ligne.">
      <Card className="flex flex-wrap items-center gap-3">
        <Badge variant="neutral">Neutre</Badge>
        <Badge variant="primary" dot>
          Info
        </Badge>
        <Badge variant="success" dot>
          <Check size={10} /> Terminé
        </Badge>
        <Badge variant="warning" dot>
          Attention
        </Badge>
        <Badge variant="danger" dot>
          Échec
        </Badge>
        <Badge variant="primary" dot pulse>
          En cours
        </Badge>
        <JobStatusBadge status="running" />
        <JobStatusBadge status="completed" />
        <JobStatusBadge status="failed" />
        <JobStatusBadge status="queued" />
        <Button variant="ghost" size="sm">
          <Download size={14} /> Export (icône + libellé)
        </Button>
      </Card>
    </Section>
  );
}

// ── Squelettes ───────────────────────────────────────────────────────────

function SkeletonSection() {
  return (
    <Section
      title="Squelettes"
      description="Reprennent la FORME du contenu réel, jamais un bloc générique. role=&quot;status&quot; posé une seule fois sur le groupe, aria-hidden sur chaque bloc — un lecteur d'écran entend « chargement en cours », pas 5 rectangles."
    >
      <Card>
        <SkeletonGroup label="Chargement d'une fiche" className="space-y-2 max-w-sm">
          <Skeleton className="h-3 w-20" />
          <Skeleton className="h-6 w-40" />
          <Skeleton className="h-3 w-56" />
        </SkeletonGroup>
      </Card>
    </Section>
  );
}

// ── Progression ──────────────────────────────────────────────────────────

function ProgressBarSection() {
  return (
    <Section title="Barre de progression" description="Déterminée (valeur connue) ou indéterminée (durée inconnue, ex. file d'attente).">
      <Card className="space-y-5 max-w-md">
        <ProgressBar label="Entraînement — époque 14/20" value={14} max={20} />
        <ProgressBar label="Compact" value={62} size="sm" />
        <IndeterminateProgressBar label="En file d'attente…" />
      </Card>
    </Section>
  );
}

// ── Contrôle segmenté ────────────────────────────────────────────────────

function SegmentedSection() {
  const [view, setView] = useState<"table" | "grid" | "list">("table");
  return (
    <Section
      title="Contrôle segmenté"
      description="Bascule entre modes mutuellement exclusifs (role=&quot;radiogroup&quot;) — jamais de la navigation entre pages, qui reste le rôle de Tabs (role=&quot;tablist&quot;)."
    >
      <Card>
        <Segmented
          aria-label="Mode d'affichage"
          value={view}
          onChange={setView}
          options={[
            { id: "table", label: "Tableau", icon: Rows3 },
            { id: "grid", label: "Grille", icon: Grid3x3 },
            { id: "list", label: "Liste", icon: LayoutList },
          ]}
        />
        <p className="text-caption text-muted-foreground mt-3">Mode actif : {view}</p>
      </Card>
    </Section>
  );
}

// ── Alertes ──────────────────────────────────────────────────────────────

function AlertSection() {
  // État réel (pas un onDismiss no-op) — une démonstration qui prétend
  // fermer l'alerte doit vraiment la retirer de l'affichage (retrouvé par
  // le test clavier automatisé, Lot 2 : le bouton cliquait mais rien ne
  // disparaissait, voir _design/JOURNAL.md).
  const [dangerDismissed, setDangerDismissed] = useState(false);

  return (
    <Section
      title="Alertes"
      description="4 sémantiques. Règle de fond n°3 : un avertissement dit quoi faire — « ne rien changer » est toujours une action valable."
    >
      <div className="space-y-3">
        <Alert variant="info" title="12 valeurs manquantes détectées">
          Colonne « epaisseur_mm » — 4 % des lignes. Cela peut biaiser légèrement l'entraînement.
        </Alert>
        <Alert
          variant="warning"
          title="Colonne quasi-constante"
          actions={
            <>
              <Button size="sm" variant="secondary">
                Exclure la colonne
              </Button>
              <Button size="sm" variant="ghost">
                Garder telle quelle
              </Button>
            </>
          }
        >
          « site_id » ne varie que sur 2 % des lignes — apporte peu d'information au modèle.
        </Alert>
        {dangerDismissed ? (
          <Button variant="ghost" size="sm" onClick={() => setDangerDismissed(false)}>
            Réafficher l'alerte fermée (démonstration)
          </Button>
        ) : (
          <Alert variant="danger" title="Échec de l'entraînement" onDismiss={() => setDangerDismissed(true)}>
            La colonne cible contient des valeurs manquantes. Corrigez le jeu de données puis relancez.
          </Alert>
        )}
        <Alert variant="success" title="Modèle prêt">
          R² test 0,912 — au-dessus du seuil recommandé (0,80) pour cet usage.
        </Alert>
      </div>
    </Section>
  );
}

// ── Fil d'étapes ─────────────────────────────────────────────────────────

function StepperSection() {
  const [active, setActive] = useState(2);
  return (
    <Section title="Fil d'étapes" description="Assistant multi-écrans (entraînement, wizard Vision).">
      <Card>
        <Stepper
          ariaLabel="Étapes de démonstration"
          activeStep={active}
          maxReachedStep={3}
          onSelect={setActive}
          steps={[
            { number: 1, label: "Données" },
            { number: 2, label: "Cible" },
            { number: 3, label: "Configuration" },
            { number: 4, label: "Résultat" },
          ]}
        />
      </Card>
    </Section>
  );
}

// ── Popover & fil d'Ariane ───────────────────────────────────────────────

function PopoverBreadcrumbSection() {
  return (
    <Section title="Popover & fil d'Ariane" description="Panneau flottant déclenché au clic (fermeture au clic extérieur ou à Échap) ; navigation contextuelle de la barre haute.">
      <Card className="space-y-5">
        <Breadcrumb items={[{ label: "Prédire une valeur", to: "/training" }, { label: "Entraînement", to: "/training" }, { label: "Résultat" }]} />
        <Popover
          trigger={({ toggle, triggerProps }) => (
            <Button variant="secondary" size="sm" onClick={toggle} {...triggerProps}>
              Ouvrir le popover
            </Button>
          )}
        >
          <p className="text-body text-foreground font-medium mb-1">Titre du panneau</p>
          <p className="text-caption text-muted-foreground">Contenu flottant posé sur --popover, jamais --surface.</p>
        </Popover>
      </Card>
    </Section>
  );
}

// ── Toast, tiroir, palette de commandes ─────────────────────────────────

function ToastDrawerCommandSection() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { push } = useToast();

  return (
    <Section
      title="Toast, tiroir, palette de commandes"
      description="Notification empilée (auto-disparition 5s), panneau latéral (même a11y que Modal), et ⌘K/Ctrl+K monté globalement sur toute route authentifiée (essayez le raccourci maintenant)."
    >
      <Card className="flex flex-wrap gap-2">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => push({ variant: "success", title: "Modèle enregistré", description: "Version v3 promue en production." })}
        >
          Déclencher un toast
        </Button>
        <Button variant="secondary" size="sm" onClick={() => setDrawerOpen(true)}>
          Ouvrir le tiroir
        </Button>
        <Badge variant="neutral">Essayez ⌘K / Ctrl+K</Badge>
      </Card>
      {drawerOpen && (
        <Drawer title="Détail (démonstration)" onClose={() => setDrawerOpen(false)}>
          <p className="text-body text-muted-foreground">
            Piège de focus, Échap pour fermer, focus restauré sur le bouton qui a ouvert ce tiroir — même mécanique que Modal.
          </p>
        </Drawer>
      )}
    </Section>
  );
}
