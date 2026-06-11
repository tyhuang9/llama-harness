import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  api,
  ChatResponse,
  getApiBase,
  Health,
  ModelsResponse,
  OllamaModel,
  RunRecord,
  setApiBase,
  Settings,
} from "./api";

type Tab = "dashboard" | "models" | "runs" | "instructions" | "tools" | "settings";

const tabs: Array<{ id: Tab; label: string }> = [
  { id: "dashboard", label: "Dashboard" },
  { id: "models", label: "Models" },
  { id: "runs", label: "Runs" },
  { id: "instructions", label: "Instructions" },
  { id: "tools", label: "Tools" },
  { id: "settings", label: "Settings" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [apiBaseInput, setApiBaseInput] = useState(getApiBase());
  const [prompt, setPrompt] = useState("Summarize this local harness in one sentence.");
  const [testModel, setTestModel] = useState("");
  const [testResult, setTestResult] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const recentErrors = useMemo(
    () => runs.filter((run) => run.status === "failed").slice(0, 5),
    [runs],
  );
  const activeTab = tabs.find((item) => item.id === tab) || tabs[0];

  async function refreshAll() {
    setError(null);
    const [healthResult, settingsResult, runsResult] = await Promise.allSettled([
      api.health(),
      api.settings(),
      api.runs(25),
    ]);

    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
    } else {
      setError(healthResult.reason.message);
    }

    if (settingsResult.status === "fulfilled") {
      setSettings(settingsResult.value);
      setTestModel(settingsResult.value.default_model || "");
    }

    if (runsResult.status === "fulfilled") {
      setRuns(runsResult.value.runs);
    }

    try {
      setModels(await api.models());
    } catch (err) {
      if (!healthResult.status || healthResult.status === "fulfilled") {
        setError((err as Error).message);
      }
      setModels(null);
    }
  }

  useEffect(() => {
    refreshAll();
  }, []);

  async function runModelTest(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setTestResult(null);
    try {
      const result = await api.testModel(testModel || null, prompt);
      setTestResult(result);
      await refreshRunsOnly();
    } catch (err) {
      setError((err as Error).message);
      await refreshRunsOnly();
    } finally {
      setBusy(false);
    }
  }

  async function refreshRunsOnly() {
    const result = await api.runs(25);
    setRuns(result.runs);
  }

  async function selectDefaultModel(model: string) {
    setBusy(true);
    setError(null);
    try {
      const updated = await api.setDefaultModel(model);
      setSettings(updated);
      setTestModel(model);
      await refreshAll();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function saveSettings(event: FormEvent) {
    event.preventDefault();
    if (!settings) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      setApiBase(apiBaseInput);
      const updated = await api.updateSettings(settings);
      setSettings(updated);
      await refreshAll();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell" data-theme={settings?.theme || "dark"}>
      <aside className="sidebar">
        <div className="brand">
          <h1>llama-harness</h1>
          <span className={health?.ollama_reachable ? "status-pill good-pill" : "status-pill bad-pill"}>
            {health?.ollama_reachable ? "Ollama online" : "Ollama offline"}
          </span>
        </div>

        <nav className="tabs" aria-label="Sections">
          {tabs.map((item) => (
            <button
              key={item.id}
              className={tab === item.id ? "active" : ""}
              type="button"
              onClick={() => setTab(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span>API</span>
          <code>{getApiBase()}</code>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Local Ollama Harness</p>
            <h2>{activeTab.label}</h2>
          </div>
          <button className="primary" type="button" onClick={refreshAll} disabled={busy}>
            Refresh
          </button>
        </header>

        {error ? <div className="notice error">{error}</div> : null}

        <main>
          {tab === "dashboard" ? (
            <Dashboard
              health={health}
              runs={runs}
              recentErrors={recentErrors}
              prompt={prompt}
              setPrompt={setPrompt}
              testModel={testModel}
              setTestModel={setTestModel}
              testResult={testResult}
              onSubmit={runModelTest}
              busy={busy}
            />
          ) : null}

          {tab === "models" ? (
            <Models
              models={models?.models || []}
              defaultModel={models?.default_model || settings?.default_model || null}
              onSelect={selectDefaultModel}
              testModel={testModel}
              setTestModel={setTestModel}
              prompt={prompt}
              setPrompt={setPrompt}
              testResult={testResult}
              onSubmit={runModelTest}
              busy={busy}
            />
          ) : null}

          {tab === "runs" ? <Runs runs={runs} /> : null}
          {tab === "instructions" && settings ? (
            <InstructionsView settings={settings} setSettings={setSettings} onSubmit={saveSettings} busy={busy} />
          ) : null}
          {tab === "tools" ? <Tools /> : null}
          {tab === "settings" && settings ? (
            <SettingsView
              settings={settings}
              setSettings={setSettings}
              apiBaseInput={apiBaseInput}
              setApiBaseInput={setApiBaseInput}
              onSubmit={saveSettings}
              busy={busy}
            />
          ) : null}
        </main>
      </section>
    </div>
  );
}

function Dashboard(props: {
  health: Health | null;
  runs: RunRecord[];
  recentErrors: RunRecord[];
  prompt: string;
  setPrompt: (value: string) => void;
  testModel: string;
  setTestModel: (value: string) => void;
  testResult: ChatResponse | null;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
}) {
  return (
    <div className="stack">
      <section className="panel">
        <h2>Service Status</h2>
        <dl className="status-grid">
          <Metric label="Service" value={props.health?.running ? "running" : "unknown"} />
          <Metric label="Ollama" value={props.health?.ollama_reachable ? "reachable" : "not reachable"} />
          <Metric label="Endpoint" value={props.health?.ollama_endpoint || "-"} />
          <Metric label="Default model" value={props.health?.default_model || "not set"} />
          <Metric label="Local models" value={props.health?.model_count?.toString() || "-"} />
          <Metric label="Uptime" value={props.health ? `${props.health.uptime_seconds}s` : "-"} />
        </dl>
      </section>

      <section className="panel">
        <h2>Quick Model Test</h2>
        <ModelTestForm
          prompt={props.prompt}
          setPrompt={props.setPrompt}
          testModel={props.testModel}
          setTestModel={props.setTestModel}
          testResult={props.testResult}
          onSubmit={props.onSubmit}
          busy={props.busy}
        />
      </section>

      <section className="two-column">
        <div className="panel">
          <h2>Recent Runs</h2>
          <RunsTable runs={props.runs.slice(0, 6)} compact />
        </div>
        <div className="panel">
          <h2>Recent Errors</h2>
          {props.recentErrors.length ? <RunsTable runs={props.recentErrors} compact /> : <p className="empty">No recent errors.</p>}
        </div>
      </section>
    </div>
  );
}

function Models(props: {
  models: OllamaModel[];
  defaultModel: string | null;
  onSelect: (model: string) => void;
  testModel: string;
  setTestModel: (value: string) => void;
  prompt: string;
  setPrompt: (value: string) => void;
  testResult: ChatResponse | null;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
}) {
  return (
    <div className="stack">
      <section className="panel">
        <h2>Models</h2>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Family</th>
              <th>Size</th>
              <th>Quantization</th>
              <th>Default</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {props.models.map((model) => (
              <tr key={model.name}>
                <td>{model.name}</td>
                <td>{model.details?.family || "-"}</td>
                <td>{formatBytes(model.size)}</td>
                <td>{model.details?.quantization_level || "-"}</td>
                <td>{props.defaultModel === model.name ? "yes" : "no"}</td>
                <td>
                  <button type="button" onClick={() => props.onSelect(model.name)} disabled={props.busy}>
                    Set default
                  </button>
                </td>
              </tr>
            ))}
            {!props.models.length ? (
              <tr>
                <td colSpan={6} className="empty-cell">
                  No models returned by Ollama.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>Test Selected Model</h2>
        <ModelTestForm
          prompt={props.prompt}
          setPrompt={props.setPrompt}
          testModel={props.testModel}
          setTestModel={props.setTestModel}
          testResult={props.testResult}
          onSubmit={props.onSubmit}
          busy={props.busy}
        />
      </section>
    </div>
  );
}

function Runs({ runs }: { runs: RunRecord[] }) {
  return (
    <section className="panel">
      <h2>Runs</h2>
      <RunsTable runs={runs} />
    </section>
  );
}

function Tools() {
  return (
    <section className="panel">
      <h2>Tools</h2>
      <table>
        <thead>
          <tr>
            <th>Field</th>
            <th>MVP Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Status</td>
            <td>Placeholder only</td>
          </tr>
          <tr>
            <td>Registry</td>
            <td>Future local-only tools with name, description, input schema, enabled flag</td>
          </tr>
          <tr>
            <td>Execution</td>
            <td>Not implemented for MVP</td>
          </tr>
        </tbody>
      </table>
    </section>
  );
}

function InstructionsView(props: {
  settings: Settings;
  setSettings: (settings: Settings) => void;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
}) {
  const { settings, setSettings } = props;

  function updateInstructions(next: Partial<Settings["instructions"]>) {
    setSettings({
      ...settings,
      instructions: {
        ...settings.instructions,
        ...next,
      },
    });
  }

  return (
    <section className="panel">
      <div className="section-header">
        <h2>Global Instructions</h2>
        <label className="switch-row">
          <input
            type="checkbox"
            checked={settings.instructions.enabled}
            onChange={(event) => updateInstructions({ enabled: event.target.checked })}
          />
          Apply to every run
        </label>
      </div>
      <form className="settings-form" onSubmit={props.onSubmit}>
        <label>
          System instructions
          <textarea
            value={settings.instructions.system_prompt}
            onChange={(event) => updateInstructions({ system_prompt: event.target.value })}
            placeholder="You are a careful local assistant. Follow the user's instructions exactly."
            rows={8}
          />
        </label>
        <label>
          Tool instructions
          <textarea
            value={settings.instructions.tool_context}
            onChange={(event) => updateInstructions({ tool_context: event.target.value })}
            placeholder="summarize_note: summarize note text&#10;extract_actions: return action items from note content"
            rows={8}
          />
        </label>
        <button className="primary" type="submit" disabled={props.busy}>
          Save instructions
        </button>
      </form>
    </section>
  );
}

function SettingsView(props: {
  settings: Settings;
  setSettings: (settings: Settings) => void;
  apiBaseInput: string;
  setApiBaseInput: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
}) {
  const { settings, setSettings } = props;

  function updateGeneration<K extends keyof Settings["generation"]>(key: K, value: number) {
    setSettings({
      ...settings,
      generation: {
        ...settings.generation,
        [key]: value,
      },
    });
  }

  return (
    <section className="panel">
      <h2>Settings</h2>
      <form className="settings-form" onSubmit={props.onSubmit}>
        <label>
          API base URL
          <input value={props.apiBaseInput} onChange={(event) => props.setApiBaseInput(event.target.value)} />
        </label>
        <label>
          Ollama endpoint
          <input
            value={settings.ollama_endpoint}
            onChange={(event) => setSettings({ ...settings, ollama_endpoint: event.target.value })}
          />
        </label>
        <label>
          Default model
          <input
            value={settings.default_model || ""}
            onChange={(event) => setSettings({ ...settings, default_model: event.target.value || null })}
          />
        </label>
        <div className="field-row">
          <label>
            Temperature
            <input
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={settings.generation.temperature}
              onChange={(event) => updateGeneration("temperature", Number(event.target.value))}
            />
          </label>
          <label>
            Top P
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={settings.generation.top_p}
              onChange={(event) => updateGeneration("top_p", Number(event.target.value))}
            />
          </label>
          <label>
            Max tokens
            <input
              type="number"
              min="1"
              step="1"
              value={settings.generation.max_tokens}
              onChange={(event) => updateGeneration("max_tokens", Number(event.target.value))}
            />
          </label>
        </div>
        <label>
          API token
          <input
            value={settings.api_token || ""}
            onChange={(event) => setSettings({ ...settings, api_token: event.target.value || null })}
          />
        </label>
        <label>
          Theme
          <select value={settings.theme} onChange={(event) => setSettings({ ...settings, theme: event.target.value })}>
            <option value="dark">dark</option>
            <option value="light">light</option>
          </select>
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={settings.logging_enabled}
            onChange={(event) => setSettings({ ...settings, logging_enabled: event.target.checked })}
          />
          JSONL run logging
        </label>
        <button className="primary" type="submit" disabled={props.busy}>
          Save settings
        </button>
      </form>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function ModelTestForm(props: {
  prompt: string;
  setPrompt: (value: string) => void;
  testModel: string;
  setTestModel: (value: string) => void;
  testResult: ChatResponse | null;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
}) {
  return (
    <form className="model-test" onSubmit={props.onSubmit}>
      <label>
        Model override
        <input
          value={props.testModel}
          onChange={(event) => props.setTestModel(event.target.value)}
          placeholder="leave blank to use default"
        />
      </label>
      <label>
        Prompt
        <textarea value={props.prompt} onChange={(event) => props.setPrompt(event.target.value)} rows={4} />
      </label>
      <button className="primary" type="submit" disabled={props.busy}>
        Run test
      </button>
      {props.testResult ? (
        <pre className="result">{props.testResult.message.content || "(empty response)"}</pre>
      ) : null}
    </form>
  );
}

function RunsTable({ runs, compact = false }: { runs: RunRecord[]; compact?: boolean }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Model</th>
          <th>Status</th>
          {!compact ? <th>Source</th> : null}
          <th>Duration</th>
          {!compact ? <th>Summary</th> : null}
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => (
          <tr key={run.id}>
            <td>{formatTime(run.started_at)}</td>
            <td>{run.model}</td>
            <td className={run.status === "failed" ? "bad" : "good"}>{run.status}</td>
            {!compact ? <td>{run.source_app || "-"}</td> : null}
            <td>{run.duration_ms}ms</td>
            {!compact ? <td>{run.error || run.prompt_summary}</td> : null}
          </tr>
        ))}
        {!runs.length ? (
          <tr>
            <td colSpan={compact ? 4 : 6} className="empty-cell">
              No runs recorded.
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}

function formatTime(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function formatBytes(value?: number | null): string {
  if (!value) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
