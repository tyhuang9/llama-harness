import { FormEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import { tauriConsoleApi } from "./bridge";
import type {
  AgentDefinition,
  CommandPreview,
  CommandResult,
  ConsoleApi,
  ConsoleEvent,
  ConsoleModels,
  ConsolePreferences,
  ConsoleRun,
  EvalLaunchRequest,
  EvaluationArtifacts,
  PromptfooArtifact,
  ProjectWorkspace,
} from "./types";

type View = "project" | "models" | "agents" | "runs" | "evaluations" | "settings";
type LoadState<T> = { value?: T; loading: boolean; error?: string };

const navigation: Array<{ id: View; label: string; hint: string }> = [
  { id: "project", label: "Project", hint: "Local paths" },
  { id: "models", label: "Models", hint: "Loopback Ollama" },
  { id: "agents", label: "Agents", hint: "Project manifest" },
  { id: "runs", label: "Runs", hint: "Redacted SQLite traces" },
  { id: "evaluations", label: "Evaluations", hint: "Saved artifacts" },
  { id: "settings", label: "Settings", hint: "Console preferences" },
];

export function App({ api = tauriConsoleApi }: { api?: ConsoleApi }) {
  const [preferences, setPreferences] = useState<ConsolePreferences>();
  const [view, setView] = useState<View>("project");
  const [startupError, setStartupError] = useState<string>();
  const [models, setModels] = useState<LoadState<ConsoleModels>>({ loading: false });
  const [agents, setAgents] = useState<LoadState<AgentDefinition[]>>({ loading: false });
  const [runs, setRuns] = useState<LoadState<ConsoleRun[]>>({ loading: false });
  const [artifacts, setArtifacts] = useState<LoadState<EvaluationArtifacts>>({ loading: false });
  const [promptfooArtifacts, setPromptfooArtifacts] = useState<LoadState<PromptfooArtifact[]>>({ loading: false });

  const workspace = preferences?.workspace;
  const refreshWorkspaceData = async () => {
    if (!workspace) return;
    setModels((current) => ({ ...current, loading: true, error: undefined }));
    setRuns((current) => ({ ...current, loading: true, error: undefined }));
    setAgents((current) => ({ ...current, loading: true, error: undefined }));
    setArtifacts((current) => ({ ...current, loading: true, error: undefined }));
    setPromptfooArtifacts((current) => ({ ...current, loading: true, error: undefined }));
    const [nextModels, nextRuns, nextAgents, nextArtifacts, nextPromptfooArtifacts] = await Promise.allSettled([
      api.listModels(),
      api.listRuns({}),
      api.listAgents(),
      api.listEvaluationArtifacts(),
      api.listPromptfooArtifacts(),
    ]);
    setModels(toLoadState(nextModels));
    setRuns(toLoadState(nextRuns));
    setAgents(toLoadState(nextAgents));
    setArtifacts(toLoadState(nextArtifacts));
    setPromptfooArtifacts(toLoadState(nextPromptfooArtifacts));
  };

  useEffect(() => {
    api
      .getPreferences()
      .then((loaded) => {
        setPreferences(loaded);
        if (loaded.workspace) setView("models");
      })
      .catch((error: unknown) => setStartupError(describeError(error)));
  }, [api]);

  useEffect(() => {
    void refreshWorkspaceData();
    // Refresh only when the selected workspace changes; the API object is stable in production.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace?.projectRoot, workspace?.traceDbPath, workspace?.evaluationResultsPath, workspace?.agentManifestPath, workspace?.ollamaUrl]);

  const connectedLabel = workspace ? workspace.projectRoot : "No local project connected";
  const updatePreferences = (next: ConsolePreferences) => setPreferences(next);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand" aria-label="Llama Harness Developer Console">
          <span className="brand-mark" aria-hidden="true">LH</span>
          <div>
            <p className="eyebrow">LOCAL DEVELOPER CONSOLE</p>
            <h1>Llama Harness</h1>
          </div>
        </div>
        <div className="connection-status" title={connectedLabel}>
          <span className={workspace ? "status-dot good" : "status-dot"} aria-hidden="true" />
          {workspace ? "Project connected" : "No project connected"}
        </div>
      </header>

      <aside className="sidebar">
        <p className="workspace-label">Workspace</p>
        <p className="workspace-path">{connectedLabel}</p>
        <nav aria-label="Console navigation">
          {navigation.map((item) => (
            <button
              className={view === item.id ? "nav-item active" : "nav-item"}
              key={item.id}
              onClick={() => setView(item.id)}
              aria-current={view === item.id ? "page" : undefined}
              type="button"
            >
              <span>{item.label}</span>
              <small>{item.hint}</small>
            </button>
          ))}
        </nav>
        <div className="sidebar-footnote">Direct local files and loopback Ollama only.</div>
      </aside>

      <main className="main-content">
        {startupError && <Notice kind="error">Could not load console preferences: {startupError}</Notice>}
        {!preferences && !startupError ? <Loading label="Loading local preferences" /> : null}
        {preferences && view === "project" && (
          <ProjectScreen
            api={api}
            preferences={preferences}
            onSaved={(next) => {
              updatePreferences(next);
              setView("models");
            }}
          />
        )}
        {preferences && workspace && view === "models" && (
          <ModelsScreen state={models} onRefresh={() => void refreshWorkspaceData()} />
        )}
        {preferences && workspace && view === "agents" && <AgentsScreen state={agents} />}
        {preferences && workspace && view === "runs" && (
          <RunsScreen api={api} initialState={runs} />
        )}
        {preferences && workspace && view === "evaluations" && (
          <EvaluationsScreen api={api} state={artifacts} promptfooArtifacts={promptfooArtifacts} />
        )}
        {preferences && workspace && view === "settings" && (
          <SettingsScreen api={api} preferences={preferences} onSaved={updatePreferences} />
        )}
        {preferences && !workspace && view !== "project" && (
          <EmptyState
            title="Connect a project first"
            detail="Choose a project root, an existing SQLite trace database, and an optional evaluation results location."
            actionLabel="Open project setup"
            onAction={() => setView("project")}
          />
        )}
      </main>
    </div>
  );
}

function ProjectScreen({
  api,
  preferences,
  onSaved,
}: {
  api: ConsoleApi;
  preferences: ConsolePreferences;
  onSaved: (preferences: ConsolePreferences) => void;
}) {
  const [workspace, setWorkspace] = useState<ProjectWorkspace>(
    preferences.workspace ?? {
      projectRoot: "",
      traceDbPath: "",
      evaluationResultsPath: "",
      agentManifestPath: "",
      ollamaUrl: "http://127.0.0.1:11434",
    },
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string>();

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError(undefined);
    try {
      const saved = await api.connectWorkspace({
        ...workspace,
        evaluationResultsPath: workspace.evaluationResultsPath?.trim() || undefined,
        agentManifestPath: workspace.agentManifestPath?.trim() || undefined,
      });
      onSaved(saved);
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="page-grid project-page" aria-labelledby="project-title">
      <div>
        <p className="eyebrow">LOCAL WORKSPACE</p>
        <h2 id="project-title">Connect project data</h2>
        <p className="lead">The console reads an existing Harness SQLite trace store and saved evaluation JSON from this machine. No daemon, account, or remote telemetry is involved.</p>
      </div>
      <form className="panel form-panel" onSubmit={submit} aria-describedby="project-help">
        <p id="project-help" className="muted">All paths must be absolute and already exist. Ollama must use a loopback URL.</p>
        <PathField label="Project root" value={workspace.projectRoot} onChange={(projectRoot) => setWorkspace((current) => ({ ...current, projectRoot }))} required />
        <PathField label="SQLite trace database" value={workspace.traceDbPath} onChange={(traceDbPath) => setWorkspace((current) => ({ ...current, traceDbPath }))} required />
        <PathField label="Evaluation results path (optional)" value={workspace.evaluationResultsPath ?? ""} onChange={(evaluationResultsPath) => setWorkspace((current) => ({ ...current, evaluationResultsPath }))} />
        <PathField label="Agent manifest path (optional)" value={workspace.agentManifestPath ?? ""} onChange={(agentManifestPath) => setWorkspace((current) => ({ ...current, agentManifestPath }))} />
        <PathField label="Local Ollama URL" value={workspace.ollamaUrl} onChange={(ollamaUrl) => setWorkspace((current) => ({ ...current, ollamaUrl }))} required />
        {error && <Notice kind="error">{error}</Notice>}
        <button className="button primary" disabled={saving} type="submit">{saving ? "Connecting…" : "Connect local project"}</button>
      </form>
      <div className="info-grid" aria-label="Console boundaries">
        <InfoCard title="Trace privacy" detail="Only structured redacted events are displayed. Raw trace payloads are never sent to the console." />
        <InfoCard title="Ollama scope" detail="The native provider rejects non-loopback addresses. This console does not operate a model service." />
        <InfoCard title="Command scope" detail="The console can only construct project-relative Harness eval and replay CLI commands; it does not run arbitrary shell input." />
      </div>
    </section>
  );
}

function ModelsScreen({ state, onRefresh }: { state: LoadState<ConsoleModels>; onRefresh: () => void }) {
  return (
    <section className="page" aria-labelledby="models-title">
      <PageHeading eyebrow="LOOPBACK PROVIDER" title="Local Ollama models" actionLabel="Refresh models" onAction={onRefresh} />
      {state.loading && <Loading label="Checking local Ollama" />}
      {state.error && <Notice kind="error">{state.error}</Notice>}
      {state.value && (
        <>
          <div className="health-row panel">
            <span className={state.value.health.healthy ? "status-dot good" : "status-dot bad"} aria-hidden="true" />
            <div><strong>{state.value.health.healthy ? "Ollama reachable" : "Ollama unavailable"}</strong><p>{state.value.health.detail ?? "Direct loopback check completed."}</p></div>
          </div>
          {state.value.models.length === 0 ? (
            <EmptyState title="No local models reported" detail="Start Ollama and install a model locally, then refresh this view." />
          ) : (
            <div className="card-list" aria-label="Local Ollama models">
              {state.value.models.map((model) => <ModelCard key={model.id} model={model} />)}
            </div>
          )}
        </>
      )}
    </section>
  );
}

function AgentsScreen({ state }: { state: LoadState<AgentDefinition[]> }) {
  return (
    <section className="page" aria-labelledby="agents-title">
      <PageHeading eyebrow="PROJECT-OWNED DEFINITIONS" title="Agents" />
      {state.loading && <Loading label="Reading validated agent manifest" />}
      {state.error && <Notice kind="error">{state.error}</Notice>}
      {state.value && state.value.length === 0 && (
        <EmptyState
          title="No agent manifest configured"
          detail="Add an existing project-relative YAML or JSON agent manifest in Project. The console only inspects definitions; your application still owns tool registration, policy, and approvals."
        />
      )}
      {state.value?.map((agent) => (
        <article className="agent-card panel" key={agent.id}>
          <div className="agent-title"><div><p className="eyebrow">{agent.id}</p><h3>{agent.name}</h3></div><span className="tag active">v{agent.version}</span></div>
          <dl className="agent-grid"><div><dt>Default model</dt><dd>{agent.defaultModel}</dd></div><div><dt>Limits</dt><dd>{agent.limits.maxModelCalls} model / {agent.limits.maxToolCalls} tool calls</dd></div><div><dt>Allowed tools</dt><dd>{agent.toolAllowlist.length ? agent.toolAllowlist.join(", ") : "None declared"}</dd></div><div><dt>Prompt version</dt><dd>{String(agent.metadata.prompt_version ?? "Not declared")}</dd></div></dl>
          {agent.systemInstructions && <details><summary>System instructions</summary><p>{agent.systemInstructions}</p></details>}
          {agent.outputSchema && <details><summary>Structured output schema</summary><pre>{JSON.stringify(agent.outputSchema, null, 2)}</pre></details>}
        </article>
      ))}
    </section>
  );
}

function RunsScreen({ api, initialState }: { api: ConsoleApi; initialState: LoadState<ConsoleRun[]> }) {
  const [query, setQuery] = useState({ traceId: "", status: "" });
  const [state, setState] = useState(initialState);
  const [selected, setSelected] = useState<ConsoleRun>();
  const [events, setEvents] = useState<LoadState<ConsoleEvent[]>>({ loading: false });

  useEffect(() => setState(initialState), [initialState]);
  useEffect(() => {
    if (!selected) return;
    setEvents({ loading: true });
    api.listRunEvents(selected.executionId)
      .then((value) => setEvents({ value, loading: false }))
      .catch((error: unknown) => setEvents({ loading: false, error: describeError(error) }));
  }, [api, selected]);

  const search = async (event: FormEvent) => {
    event.preventDefault();
    setState((current) => ({ ...current, loading: true, error: undefined }));
    try {
      setState({ value: await api.listRuns(query), loading: false });
    } catch (error) {
      setState({ loading: false, error: describeError(error) });
    }
  };

  return (
    <section className="page run-layout" aria-labelledby="runs-title">
      <div className="runs-main">
        <PageHeading eyebrow="REDACTED TRACE STORE" title="Runs" />
        <form className="filter-bar panel" onSubmit={search}>
          <label>Trace ID<input value={query.traceId} onChange={(event) => setQuery((current) => ({ ...current, traceId: event.target.value }))} /></label>
          <label>Status<select value={query.status} onChange={(event) => setQuery((current) => ({ ...current, status: event.target.value }))}><option value="">All statuses</option><option value="completed">Completed</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option><option value="limit_reached">Limit reached</option></select></label>
          <button className="button secondary" type="submit">Filter runs</button>
        </form>
        {state.loading && <Loading label="Reading local trace database" />}
        {state.error && <Notice kind="error">{state.error}</Notice>}
        {state.value && state.value.length === 0 && <EmptyState title="No matching runs" detail="This trace database has no runs matching the current filter." />}
        {state.value && state.value.length > 0 && <RunTable runs={state.value} selected={selected?.executionId} onSelect={setSelected} />}
      </div>
      <aside className="run-detail panel" aria-live="polite">
        {selected ? <RunDetail run={selected} events={events} /> : <EmptyState title="Select a run" detail="Choose a local trace run to inspect its redacted event timeline." />}
      </aside>
    </section>
  );
}

function EvaluationsScreen({ api, state, promptfooArtifacts }: { api: ConsoleApi; state: LoadState<EvaluationArtifacts>; promptfooArtifacts: LoadState<PromptfooArtifact[]> }) {
  const [request, setRequest] = useState<EvalLaunchRequest>({ suitePath: "", models: [], repeat: undefined });
  const [preview, setPreview] = useState<CommandPreview>();
  const [result, setResult] = useState<CommandResult>();
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const reportSummary = useMemo(() => state.value?.reports ?? [], [state.value]);
  const normalizedRequest = () => ({ ...request, models: request.models.filter(Boolean) });
  const previewCommand = async () => {
    setBusy(true); setError(undefined); setResult(undefined);
    try { setPreview(await api.previewEvalCommand(normalizedRequest())); } catch (reason) { setError(describeError(reason)); } finally { setBusy(false); }
  };
  const launchCommand = async () => {
    setBusy(true); setError(undefined);
    try { setResult(await api.launchEvalCommand(normalizedRequest())); } catch (reason) { setError(describeError(reason)); } finally { setBusy(false); }
  };
  return (
    <section className="page" aria-labelledby="evaluations-title">
      <PageHeading eyebrow="ARTIFACTS AND CLI" title="Evaluations" />
      <div className="split-grid">
        <section className="panel" aria-labelledby="reports-title"><h3 id="reports-title">Saved reports</h3>
          {state.loading && <Loading label="Reading evaluation artifacts" />}
          {state.error && <Notice kind="error">{state.error}</Notice>}
          {reportSummary.length === 0 && !state.loading && <EmptyState title="No evaluation reports found" detail="Set an existing JSON artifact directory in Project, then run your embedding application to write a report." />}
          {reportSummary.map(({ path, report }) => <EvaluationReportCard key={path} path={path} report={report} />)}
          {state.value?.skippedFiles.length ? <p className="muted">Skipped {state.value.skippedFiles.length} JSON file(s) that were not Harness evaluation reports.</p> : null}
          <h3>Promptfoo artifacts</h3>
          {promptfooArtifacts.loading && <Loading label="Reading generated Promptfoo artifacts" />}
          {promptfooArtifacts.error && <Notice kind="error">{promptfooArtifacts.error}</Notice>}
          {promptfooArtifacts.value?.length === 0 && !promptfooArtifacts.loading && <p className="muted">No generated Promptfoo config or raw result exists under <code>.llama-harness</code>.</p>}
          {promptfooArtifacts.value?.map((artifact) => <details className="report-card" key={artifact.path}><summary>{artifact.kind === "generated_config" ? "Generated Promptfoo config" : "Raw Promptfoo result"}</summary><small title={artifact.path}>{artifact.path}</small>{artifact.truncated && <p className="muted">Preview is capped at 512 KiB.</p>}<pre>{artifact.content}</pre></details>)}
        </section>
        <section className="panel" aria-labelledby="cli-title"><h3 id="cli-title">Constrained Harness CLI</h3>
          <p className="muted">This creates only a project-relative <code>cargo run -p llama-harness-cli -- eval run</code> command. Standalone evaluation execution remains intentionally unavailable without the embedding application’s adapter.</p>
          <label>Suite path relative to project root<input value={request.suitePath} onChange={(event) => setRequest((current) => ({ ...current, suitePath: event.target.value }))} placeholder="evals/local-task-agent/suite.yaml" /></label>
          <label>Model override (optional)<input value={request.models.join(", ")} onChange={(event) => setRequest((current) => ({ ...current, models: event.target.value.split(",").map((value) => value.trim()) }))} placeholder="ollama/qwen3:latest" /></label>
          <label>Repeat (optional)<input type="number" min="1" value={request.repeat ?? ""} onChange={(event) => setRequest((current) => ({ ...current, repeat: event.target.value ? Number(event.target.value) : undefined }))} /></label>
          <div className="button-row"><button type="button" className="button secondary" disabled={busy || !request.suitePath} onClick={() => void previewCommand()}>Preview command</button><button type="button" className="button primary" disabled={busy || !request.suitePath} onClick={() => void launchCommand()}>Run constrained command</button></div>
          {error && <Notice kind="error">{error}</Notice>}
          {preview && <CommandOutput title="Command preview" command={preview} />}
          {result && <CommandResultView result={result} />}
        </section>
      </div>
    </section>
  );
}

function SettingsScreen({ api, preferences, onSaved }: { api: ConsoleApi; preferences: ConsolePreferences; onSaved: (value: ConsolePreferences) => void }) {
  const [rawPayloadPreference, setRawPayloadPreference] = useState(preferences.rawPayloadPreference);
  const [redactionKeys, setRedactionKeys] = useState(preferences.redactionKeyFragments.join(", "));
  const [retentionDays, setRetentionDays] = useState(preferences.retentionDays?.toString() ?? "");
  const [message, setMessage] = useState<string>();
  const save = async (event: FormEvent) => {
    event.preventDefault(); setMessage(undefined);
    try {
      const saved = await api.savePreferences({ ...preferences, rawPayloadPreference, redactionKeyFragments: redactionKeys.split(",").map((value) => value.trim()).filter(Boolean), retentionDays: retentionDays ? Number(retentionDays) : undefined });
      onSaved(saved); setMessage("Preferences saved locally.");
    } catch (error) { setMessage(describeError(error)); }
  };
  return <section className="page" aria-labelledby="settings-title"><PageHeading eyebrow="LOCAL CONSOLE" title="Settings" />
    <form className="panel form-panel" onSubmit={save}>
      <label className="checkbox-label"><input type="checkbox" checked={rawPayloadPreference} onChange={(event) => setRawPayloadPreference(event.target.checked)} /> Remember that raw payloads are not displayed</label>
      <p className="muted">This is a reminder preference only. It cannot enable capture, change an existing database, or recover raw payloads.</p>
      <label>Suggested redaction key fragments<input value={redactionKeys} onChange={(event) => setRedactionKeys(event.target.value)} placeholder="authorization, token, secret" /></label>
      <label>Suggested retention days<input type="number" min="1" value={retentionDays} onChange={(event) => setRetentionDays(event.target.value)} /></label>
      <p className="muted">These preferences are stored with the console and are not applied retroactively to project trace stores.</p>
      {message && <Notice kind={message === "Preferences saved locally." ? "success" : "error"}>{message}</Notice>}
      <button className="button primary" type="submit">Save local preferences</button>
    </form>
  </section>;
}

function RunTable({ runs, selected, onSelect }: { runs: ConsoleRun[]; selected?: string; onSelect: (run: ConsoleRun) => void }) {
  return <div className="table-wrap"><table><caption className="sr-only">Local redacted trace runs</caption><thead><tr><th>Run</th><th>Trace</th><th>Status</th><th>Events</th><th>Updated</th></tr></thead><tbody>{runs.map((run) => <tr className={selected === run.executionId ? "selected" : ""} key={run.executionId}><td><button type="button" className="table-link" onClick={() => onSelect(run)}>{run.runId}</button></td><td>{run.traceId}</td><td><Status value={run.status} /></td><td>{run.eventCount}</td><td>{formatTime(run.updatedAtMs)}</td></tr>)}</tbody></table></div>;
}

function RunDetail({ run, events }: { run: ConsoleRun; events: LoadState<ConsoleEvent[]> }) {
  return <><p className="eyebrow">RUN DETAIL</p><h3>{run.runId}</h3><dl className="detail-list"><div><dt>Trace</dt><dd>{run.traceId}</dd></div><div><dt>Status</dt><dd><Status value={run.status} /></dd></div><div><dt>Events</dt><dd>{run.eventCount}</dd></div></dl>
    <h4>Redacted event timeline</h4>{events.loading && <Loading label="Reading event timeline" />}{events.error && <Notice kind="error">{events.error}</Notice>}{events.value?.length === 0 && <p className="muted">No events were found for this run.</p>}<ol className="timeline">{events.value?.map((event) => <li key={event.sequence}><span>{event.sequence}</span><div><strong>{String(event.event.type ?? "event")}</strong><time>{formatTime(event.timestampMs)}</time><pre>{JSON.stringify(event.event, null, 2)}</pre></div></li>)}</ol></>;
}

function EvaluationReportCard({ path, report }: { path: string; report: EvaluationArtifacts["reports"][number]["report"] }) {
  const passed = report.results.filter((result) => result.passed).length;
  return <article className="report-card"><div><h4>{report.suiteId}</h4><p>{passed}/{report.results.length} passed · v{report.suiteVersion}</p></div><small title={path}>{path}</small>{report.results.filter((result) => !result.passed).map((result) => <p className="failure" key={`${result.caseId}-${result.repetition}`}>{result.caseId}: {result.failures.map((failure) => `${failure.rule}: ${failure.message}`).join("; ") || "failed"}</p>)}</article>;
}

function ModelCard({ model }: { model: ConsoleModels["models"][number] }) {
  return <article className="model-card panel"><h3>{model.id}</h3><div className="tag-row"><Tag label="Tools" active={model.capabilities.supportsTools} /><Tag label="Streaming" active={model.capabilities.supportsStreaming} /><Tag label="Structured output" active={model.capabilities.supportsStructuredOutput} /></div></article>;
}

function CommandOutput({ title, command }: { title: string; command: CommandPreview }) { return <div className="command-output"><h4>{title}</h4><code>{[command.program, ...command.args].join(" ")}</code><p className="muted">Working directory: {command.cwd}</p></div>; }
function CommandResultView({ result }: { result: CommandResult }) { return <div className={result.success ? "command-output success" : "command-output failure"}><h4>{result.success ? "Command completed" : "Command returned a diagnostic"}</h4><CommandOutput title="Executed command" command={result.command} /><pre>{result.stdout || result.stderr || "No output returned."}</pre>{result.stderr && result.stdout && <pre>{result.stderr}</pre>}</div>; }
function PageHeading({ eyebrow, title, actionLabel, onAction }: { eyebrow: string; title: string; actionLabel?: string; onAction?: () => void }) { return <div className="page-heading"><div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2></div>{actionLabel && onAction && <button type="button" className="button secondary" onClick={onAction}>{actionLabel}</button>}</div>; }
function PathField({ label, value, onChange, required = false }: { label: string; value: string; onChange: (value: string) => void; required?: boolean }) { return <label>{label}<input value={value} required={required} onChange={(event) => onChange(event.target.value)} /></label>; }
function Status({ value }: { value?: string }) { return <span className={`status ${value ?? "unknown"}`}>{value ?? "unknown"}</span>; }
function Tag({ label, active }: { label: string; active: boolean }) { return <span className={active ? "tag active" : "tag"}>{label}: {active ? "yes" : "no"}</span>; }
function Notice({ kind, children }: { kind: "error" | "success"; children: ReactNode }) { return <p role={kind === "error" ? "alert" : "status"} className={`notice ${kind}`}>{children}</p>; }
function Loading({ label }: { label: string }) { return <p className="loading" role="status">{label}…</p>; }
function EmptyState({ title, detail, actionLabel, onAction }: { title: string; detail: string; actionLabel?: string; onAction?: () => void }) { return <div className="empty-state"><h3>{title}</h3><p>{detail}</p>{actionLabel && onAction && <button type="button" className="button secondary" onClick={onAction}>{actionLabel}</button>}</div>; }
function InfoCard({ title, detail }: { title: string; detail: string }) { return <article className="info-card"><h3>{title}</h3><p>{detail}</p></article>; }
function toLoadState<T>(result: PromiseSettledResult<T>): LoadState<T> { return result.status === "fulfilled" ? { value: result.value, loading: false } : { loading: false, error: describeError(result.reason) }; }
function describeError(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function formatTime(timestamp: number): string { return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium" }).format(new Date(timestamp)); }
