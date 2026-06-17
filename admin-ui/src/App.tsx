import { FormEvent, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  api,
  ApplyLiteLlmProvidersResponse,
  ChatResponse,
  GenerateLiteLlmConfigResponse,
  getApiBase,
  Health,
  LiteLlmProviderConfig,
  LiteLlmServiceStartResponse,
  LiteLlmTestResponse,
  ModelsResponse,
  OllamaModel,
  ProviderModelsResponse,
  ProviderStatus,
  RunRecord,
  setApiBase,
  Settings,
} from "./api";
import {
  Agent,
  AgentPermissions,
  AuditEntry,
  Environment,
  ModelProvider,
  Sandbox,
  seedActivityFeed,
  seedAgents,
  seedApprovals,
  seedAuditLog,
  seedSandboxes,
  seedTasks,
  Task,
  TaskStatus,
} from "./sentinelData";

type Page =
  | "dashboard"
  | "tasks"
  | "new-task"
  | "task-detail"
  | "agents"
  | "sandboxes"
  | "models"
  | "permissions"
  | "logs"
  | "instructions"
  | "settings";

type Approval = (typeof seedApprovals)[number] & { state?: "pending" | "approved" | "rejected" };

type PermissionPolicy = Record<string, { allowed: boolean; approval: boolean }>;
type ProviderVerification = {
  state: "success" | "error";
  message: string;
};

const nav: Array<{ id: Page; label: string; detail?: string }> = [
  { id: "dashboard", label: "Dashboard", detail: "Live operations" },
  { id: "agents", label: "Agents", detail: "Prompts and defaults" },
  { id: "tasks", label: "Tasks", detail: "Queue and details" },
  { id: "sandboxes", label: "Sandboxes", detail: "Execution contexts" },
  { id: "models", label: "Providers", detail: "LiteLLM and Ollama" },
  { id: "permissions", label: "Permissions", detail: "Human approval gates" },
  { id: "logs", label: "Audit Log", detail: "Actions and runs" },
  { id: "instructions", label: "Instructions", detail: "Global prompts" },
  { id: "settings", label: "Settings", detail: "API and generation" },
];

const taskFilters: Array<TaskStatus | "all"> = [
  "all",
  "running",
  "waiting_approval",
  "planning",
  "queued",
  "completed",
  "failed",
];

const litellmProviderTypes = [
  "openai",
  "anthropic",
  "gemini",
  "openrouter",
  "ollama",
  "azure",
  "bedrock",
  "vertex_ai",
  "cohere",
  "mistral",
  "groq",
  "deepseek",
  "xai",
  "perplexity",
  "together_ai",
  "fireworks_ai",
  "huggingface",
  "replicate",
  "custom",
];
const REDACTED_SECRET = "__configured__";
const environments: Environment[] = ["planner", "browser", "computer-use", "local-desktop"];
const autonomyLevels: Agent["autonomy"][] = ["observe", "ask", "low-risk", "autonomous"];

const permissionRows = [
  { id: "browser", label: "Browser access", description: "Visit URLs and interact with web pages." },
  { id: "fileRead", label: "File read", description: "Read files inside an execution workspace." },
  { id: "fileWrite", label: "File write", description: "Create or modify files inside an execution workspace." },
  { id: "terminal", label: "Terminal access", description: "Execute shell commands inside a sandbox." },
  { id: "email", label: "Email sending", description: "Send mail through configured providers." },
  { id: "purchases", label: "Purchases / payments", description: "Complete checkouts or initiate payments." },
  { id: "submissions", label: "External submissions", description: "Submit forms or post to third-party services." },
];

const initialPolicy: PermissionPolicy = {
  browser: { allowed: true, approval: false },
  fileRead: { allowed: true, approval: false },
  fileWrite: { allowed: false, approval: true },
  terminal: { allowed: false, approval: true },
  email: { allowed: false, approval: true },
  purchases: { allowed: false, approval: true },
  submissions: { allowed: false, approval: true },
};

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [selectedTaskId, setSelectedTaskId] = useState(seedTasks[0]?.id || "");
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [providerStatuses, setProviderStatuses] = useState<ProviderStatus[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [apiBaseInput, setApiBaseInput] = useState(getApiBase());
  const [prompt, setPrompt] = useState("Summarize the current agent task queue in one sentence.");
  const [testModel, setTestModel] = useState("");
  const [testResult, setTestResult] = useState<ChatResponse | null>(null);
  const localModels = useMemo(() => models?.models || [], [models]);
  const [litellmTestResult, setLiteLlmTestResult] = useState<LiteLlmTestResponse | null>(null);
  const [litellmConfigResult, setLiteLlmConfigResult] = useState<GenerateLiteLlmConfigResponse | null>(null);
  const [litellmServiceResult, setLiteLlmServiceResult] = useState<LiteLlmServiceStartResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [tasks, setTasks] = useState<Task[]>(seedTasks);
  const [agents, setAgents] = useState<Agent[]>(seedAgents);
  const [sandboxes, setSandboxes] = useState<Sandbox[]>(seedSandboxes);
  const [auditLog, setAuditLog] = useState<AuditEntry[]>(seedAuditLog);
  const [approvals, setApprovals] = useState<Approval[]>(seedApprovals.map((approval) => ({ ...approval, state: "pending" })));
  const [policy, setPolicy] = useState<PermissionPolicy>(initialPolicy);
  const [estopOpen, setEstopOpen] = useState(false);

  const selectedTask = tasks.find((task) => task.id === selectedTaskId) || tasks[0];
  const pendingApprovals = approvals.filter((approval) => approval.state !== "approved" && approval.state !== "rejected");
  const activePage = page === "task-detail" ? "tasks" : page === "new-task" ? "tasks" : page;
  const activeNav = nav.find((item) => item.id === activePage) || nav[0];

  async function refreshAll() {
    setError(null);
    const [healthResult, settingsResult, runsResult, providersResult] = await Promise.allSettled([
      api.health(),
      api.settings(),
      api.runs(50),
      api.providers(),
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

    if (providersResult.status === "fulfilled") {
      setProviderStatuses(providersResult.value);
    }

    try {
      setModels(await api.models());
    } catch (err) {
      if (healthResult.status === "fulfilled") {
        setError((err as Error).message);
      }
      setModels(null);
    }
  }

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    function blurEditableOnOutsidePointer(event: PointerEvent) {
      const active = document.activeElement;
      const target = event.target;
      if (!isEditableElement(active) || !(target instanceof Element)) {
        return;
      }
      if (active.contains(target) || target.closest("input, textarea, select, [contenteditable='true'], .combobox")) {
        return;
      }
      active.blur();
    }

    document.addEventListener("pointerdown", blurEditableOnOutsidePointer, true);
    return () => document.removeEventListener("pointerdown", blurEditableOnOutsidePointer, true);
  }, []);

  async function refreshRunsOnly() {
    const result = await api.runs(50);
    setRuns(result.runs);
  }

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
      await refreshRunsOnly().catch(() => undefined);
    } finally {
      setBusy(false);
    }
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

  async function persistSettings(nextSettings: Settings) {
    setApiBase(apiBaseInput);
    const updated = await api.updateSettings(nextSettings);
    setSettings(updated);
    return updated;
  }

  async function saveSettings(eventOrSettings?: FormEvent | Settings) {
    if (eventOrSettings && "preventDefault" in eventOrSettings) {
      eventOrSettings.preventDefault();
    }
    const settingsToSave = eventOrSettings && !("preventDefault" in eventOrSettings) ? eventOrSettings : settings;
    if (!settingsToSave) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const updated = await persistSettings(settingsToSave);
      await refreshAll();
      return updated;
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function testLiteLlmConnection() {
    if (!settings) {
      return;
    }
    setBusy(true);
    setError(null);
    setLiteLlmTestResult(null);
    try {
      const updated = await persistSettings(settings);
      const model = updated.litellm.default_model || "";
      if (model) {
        const result = await api.testLiteLLMProvider(model, "Say hello from llama-harness.");
        setLiteLlmTestResult(result);
        setProviderStatuses(await api.providers());
      } else {
        const statuses = await api.providers();
        const litellm = statuses.find((provider) => provider.id === "litellm");
        setProviderStatuses(statuses);
        setLiteLlmTestResult({
          ok: Boolean(litellm?.healthy),
          content: litellm?.healthy
            ? `LiteLLM gateway is reachable at ${litellm.base_url || updated.litellm.base_url}.`
            : `LiteLLM gateway is not reachable at ${litellm?.base_url || updated.litellm.base_url}.`,
        });
      }
    } catch (err) {
      setError((err as Error).message);
      setProviderStatuses(await api.providers().catch(() => providerStatuses));
    } finally {
      setBusy(false);
    }
  }

  async function generateLiteLlmConfig() {
    if (!settings) {
      return;
    }
    setBusy(true);
    setError(null);
    setLiteLlmConfigResult(null);
    try {
      const updated = await persistSettings(settings);
      const result = await api.generateLiteLLMConfig(updated.litellm.managed_config_path);
      setLiteLlmConfigResult(result);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function startLiteLlmService() {
    if (!settings) {
      return;
    }
    setBusy(true);
    setError(null);
    setLiteLlmServiceResult(null);
    try {
      const updated = await persistSettings(settings);
      if (updated.litellm.managed_config_path) {
        const configResult = await api.generateLiteLLMConfig(updated.litellm.managed_config_path);
        setLiteLlmConfigResult(configResult);
      }
      const result = await api.startLiteLLMService();
      setLiteLlmServiceResult(result);
      await refreshAll();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function applyLiteLlmProviders(providers: LiteLlmProviderConfig[]): Promise<ApplyLiteLlmProvidersResponse> {
    setApiBase(apiBaseInput);
    const result = await api.applyLiteLLMProviders(providers);
    setSettings(result.settings);
    setProviderStatuses(result.provider_statuses);
    await refreshAll();
    return result;
  }

  function openTask(taskId: string) {
    setSelectedTaskId(taskId);
    setPage("task-detail");
  }

  function updateTask(taskId: string, patch: Partial<Task>) {
    setTasks((list) => list.map((task) => (task.id === taskId ? { ...task, ...patch } : task)));
  }

  function addAudit(entry: Omit<AuditEntry, "id" | "timestamp">) {
    const timestamp = new Date().toISOString().slice(0, 19).replace("T", " ");
    setAuditLog((list) => [
      {
        ...entry,
        id: `a_${Math.random().toString(36).slice(2, 8)}`,
        timestamp,
      },
      ...list,
    ]);
  }

  function createTask(input: {
    title: string;
    instructions: string;
    agentId: string;
    providerId: string;
    provider: ModelProvider;
    model: string;
    environment: Environment;
  }) {
    const agent = agents.find((item) => item.id === input.agentId);
    const createdAt = new Date().toISOString().slice(0, 16).replace("T", " ");
    const task: Task = {
      id: `t_${Math.random().toString(36).slice(2, 6)}`,
      name: input.title,
      status: "planning",
      environment: input.environment,
      providerId: input.providerId,
      provider: input.provider,
      model: input.model || agent?.defaultModel || settings?.default_model || "default",
      createdAt,
      duration: "00:00:00",
      subgoal: "Drafting execution plan",
      reasoning: "The task has been created locally in the operator console and is ready to be wired to a backend task runner.",
      instructions: input.instructions,
      agentId: agent?.id,
    };
    setTasks((list) => [task, ...list]);
    if (agent) {
      setAgents((list) => list.map((item) => (item.id === agent.id ? { ...item, tasksRun: item.tasksRun + 1 } : item)));
    }
    addAudit({
      taskId: task.id,
      agent: agent?.name || "manual",
      action: `Created task '${task.name}'`,
      environment: task.environment,
      result: "success",
      risk: "low",
      approval: "n/a",
    });
    setSelectedTaskId(task.id);
    setPage("task-detail");
  }

  function applyEmergencyStop() {
    setTasks((list) =>
      list.map((task) =>
        task.status === "running" || task.status === "planning" || task.status === "queued" || task.status === "waiting_approval"
          ? { ...task, status: "failed", subgoal: "Stopped by emergency halt", reasoning: "Operator emergency stop halted this task." }
          : task,
      ),
    );
    setSandboxes((list) =>
      list.map((sandbox) => (sandbox.status === "running" || sandbox.status === "idle" ? { ...sandbox, status: "stopped", cpu: 0 } : sandbox)),
    );
    addAudit({
      taskId: "all",
      agent: "operator",
      action: "Emergency stop halted active tasks and sandboxes",
      environment: "local-desktop",
      result: "blocked",
      risk: "high",
      approval: "approved",
    });
    setEstopOpen(false);
  }

  return (
    <div className="app-shell" data-theme={settings?.theme || "dark"}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">LH</div>
          <div>
            <h1>Llama Harness</h1>
            <p>Agent Ops</p>
          </div>
          <StatusBadge status={health?.ollama_reachable ? "online" : "offline"} />
        </div>

        <nav className="tabs" aria-label="Sections">
          {nav.map((item) => (
            <button
              key={item.id}
              className={activePage === item.id ? "active" : ""}
              type="button"
              onClick={() => setPage(item.id)}
            >
              <span>{item.label}</span>
              <small>{item.detail}</small>
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
            <p className="eyebrow">/{activeNav.label.toLowerCase().replace(/\s+/g, "-")}</p>
            <h2>{page === "new-task" ? "New Task" : page === "task-detail" ? selectedTask?.name || "Task" : activeNav.label}</h2>
          </div>
          <div className="topbar-actions">
            <button className="danger-outline" type="button" onClick={() => setEstopOpen((value) => !value)}>
              Emergency Stop All
            </button>
            <button className="secondary" type="button" onClick={refreshAll} disabled={busy}>
              Refresh
            </button>
            <button className="primary" type="button" onClick={() => setPage("new-task")}>
              New Task
            </button>
          </div>
        </header>

        {estopOpen ? (
          <div className="notice danger-notice">
            <div>
              <strong>Confirm emergency stop</strong>
              <p>Active tasks will be marked failed and running sandboxes will be stopped in this local console state.</p>
            </div>
            <div className="button-row">
              <button type="button" onClick={() => setEstopOpen(false)}>
                Cancel
              </button>
              <button className="danger" type="button" onClick={applyEmergencyStop}>
                Stop everything
              </button>
            </div>
          </div>
        ) : null}

        {error ? <div className="notice error">{error}</div> : null}

        <main>
          {page === "dashboard" ? (
            <Dashboard
              health={health}
              runs={runs}
              tasks={tasks}
              agents={agents}
              sandboxes={sandboxes}
              approvals={pendingApprovals}
              activityFeed={seedActivityFeed}
              prompt={prompt}
              setPrompt={setPrompt}
              testModel={testModel}
              setTestModel={setTestModel}
              testResult={testResult}
              onSubmit={runModelTest}
              busy={busy}
              openTask={openTask}
            />
          ) : null}

          {page === "tasks" ? <TasksPage tasks={tasks} openTask={openTask} goNew={() => setPage("new-task")} /> : null}
          {page === "new-task" ? (
            <NewTaskPage
              agents={agents}
              settings={settings}
              localModels={localModels}
              createTask={createTask}
              cancel={() => setPage("tasks")}
            />
          ) : null}
          {page === "task-detail" && selectedTask ? (
            <TaskDetailPage
              task={selectedTask}
              auditLog={auditLog}
              approvals={pendingApprovals.filter((approval) => approval.taskId === selectedTask.id)}
              back={() => setPage("tasks")}
              updateTask={updateTask}
              addAudit={addAudit}
              resolveApproval={(approvalId, state) => {
                setApprovals((list) => list.map((approval) => (approval.id === approvalId ? { ...approval, state } : approval)));
                addAudit({
                  taskId: selectedTask.id,
                  agent: "operator",
                  action: `${state === "approved" ? "Approved" : "Rejected"} approval ${approvalId}`,
                  environment: selectedTask.environment,
                  result: state === "approved" ? "success" : "blocked",
                  risk: state === "approved" ? "medium" : "low",
                  approval: state,
                });
              }}
            />
          ) : null}

          {page === "agents" ? (
            <AgentsPage agents={agents} setAgents={setAgents} settings={settings} localModels={localModels} />
          ) : null}
          {page === "sandboxes" ? <SandboxesPage sandboxes={sandboxes} setSandboxes={setSandboxes} /> : null}
          {page === "models" ? (
            <ModelsPage
              models={models?.models || []}
              defaultModel={models?.default_model || settings?.default_model || null}
              settings={settings}
              setSettings={setSettings}
              providerStatuses={providerStatuses}
              selectDefaultModel={selectDefaultModel}
              saveSettings={saveSettings}
              applyLiteLlmProviders={applyLiteLlmProviders}
              testModel={testModel}
              setTestModel={setTestModel}
              prompt={prompt}
              setPrompt={setPrompt}
              testResult={testResult}
              onSubmit={runModelTest}
              busy={busy}
            />
          ) : null}
          {page === "permissions" ? <PermissionsPage policy={policy} setPolicy={setPolicy} /> : null}
          {page === "logs" ? <LogsPage auditLog={auditLog} runs={runs} /> : null}
          {page === "instructions" && settings ? (
            <InstructionsView settings={settings} setSettings={setSettings} onSubmit={saveSettings} busy={busy} />
          ) : null}
          {page === "settings" && settings ? (
            <SettingsView
              settings={settings}
              setSettings={setSettings}
              providerStatuses={providerStatuses}
              apiBaseInput={apiBaseInput}
              setApiBaseInput={setApiBaseInput}
              onSubmit={saveSettings}
              testLiteLlmConnection={testLiteLlmConnection}
              generateLiteLlmConfig={generateLiteLlmConfig}
              startLiteLlmService={startLiteLlmService}
              litellmTestResult={litellmTestResult}
              litellmConfigResult={litellmConfigResult}
              litellmServiceResult={litellmServiceResult}
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
  tasks: Task[];
  agents: Agent[];
  sandboxes: Sandbox[];
  approvals: Approval[];
  activityFeed: Array<{ id: string; time: string; text: string }>;
  prompt: string;
  setPrompt: (value: string) => void;
  testModel: string;
  setTestModel: (value: string) => void;
  testResult: ChatResponse | null;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
  openTask: (taskId: string) => void;
}) {
  const runningTasks = props.tasks.filter((task) => task.status === "running").length;
  const completedTasks = props.tasks.filter((task) => task.status === "completed").length;
  const failedTasks = props.tasks.filter((task) => task.status === "failed").length;
  const runningSandboxes = props.sandboxes.filter((sandbox) => sandbox.status === "running").length;

  return (
    <div className="dashboard-grid">
      <section className="hero-panel dashboard-hero">
        <div>
          <p className="eyebrow">Local Agent Operations</p>
          <h3>Tasks, agents, approvals, models, and live Ollama status in one console.</h3>
        </div>
        <dl className="status-grid compact">
          <Metric label="Service" value={props.health?.running ? "running" : "unknown"} />
          <Metric label="Ollama" value={props.health?.ollama_reachable ? "reachable" : "offline"} />
          <Metric label="Endpoint" value={props.health?.ollama_endpoint || "-"} />
          <Metric label="Default model" value={props.health?.default_model || "not set"} />
        </dl>
      </section>

      <section className="metric-strip dashboard-metrics">
        <StatCard label="Active agents" value={props.agents.filter((agent) => agent.status === "active").length} hint="configured locally" />
        <StatCard label="Running tasks" value={runningTasks} hint="agent queue" />
        <StatCard label="Running sandboxes" value={runningSandboxes} hint="execution contexts" />
        <StatCard label="Pending approvals" value={props.approvals.length} hint="human review" />
        <StatCard label="Completed" value={completedTasks} hint="local task state" />
        <StatCard label="Failed" value={failedTasks} hint="local task state" />
      </section>

      <section className="two-column wide-left dashboard-secondary">
        <div className="panel">
          <div className="section-header">
            <h2>Recent Activity</h2>
          </div>
          <ul className="activity-list">
            {props.activityFeed.map((item) => (
              <li key={item.id}>
                <span>{item.text}</span>
                <code>{item.time}</code>
              </li>
            ))}
          </ul>
        </div>
        <div className="panel">
          <div className="section-header">
            <h2>Resource Snapshot</h2>
          </div>
          <ResourceBar label="CPU" value={Math.min(95, 18 + runningTasks * 15)} />
          <ResourceBar label="RAM" value={Math.min(96, 25 + runningSandboxes * 11)} />
          <Metric label="Recent model runs" value={props.runs.length.toString()} />
          <Metric label="Local models" value={props.health?.model_count?.toString() || "-"} />
        </div>
      </section>

      <section className="panel dashboard-tasks">
        <div className="section-header">
          <h2>Recent Tasks</h2>
        </div>
        <TaskTable tasks={props.tasks.slice(0, 6)} openTask={props.openTask} />
      </section>

      <section className="panel dashboard-test">
        <div className="section-header">
          <h2>Quick Model Test</h2>
        </div>
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

function TasksPage({
  tasks,
  openTask,
  goNew,
}: {
  tasks: Task[];
  openTask: (taskId: string) => void;
  goNew: () => void;
}) {
  const [filter, setFilter] = useState<TaskStatus | "all">("all");
  const [query, setQuery] = useState("");
  const filtered = tasks.filter(
    (task) =>
      (filter === "all" || task.status === filter) &&
      [task.name, task.id, task.environment, task.provider, task.model].join(" ").toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <div className="stack">
      <section className="panel toolbar-panel">
        <div>
          <h2>Tasks</h2>
          <p>Planner, browser, computer-use, and local desktop tasks.</p>
        </div>
        <div className="filter-row">
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search tasks" />
          <button className="primary" type="button" onClick={goNew}>
            New Task
          </button>
        </div>
        <div className="chip-row">
          {taskFilters.map((item) => (
            <button key={item} className={filter === item ? "chip active" : "chip"} type="button" onClick={() => setFilter(item)}>
              {labelize(item)}
            </button>
          ))}
        </div>
      </section>

      <section className="panel">
        <TaskTable tasks={filtered} openTask={openTask} />
      </section>
    </div>
  );
}

function TaskTable({ tasks, openTask }: { tasks: Task[]; openTask: (taskId: string) => void }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Task</th>
          <th>Status</th>
          <th>Environment</th>
          <th>Provider</th>
          <th>Created</th>
          <th>Duration</th>
        </tr>
      </thead>
      <tbody>
        {tasks.map((task) => (
          <tr key={task.id} className="clickable-row" onClick={() => openTask(task.id)}>
            <td>
              <strong>{task.name}</strong>
              <code>{task.id}</code>
            </td>
            <td>
              <StatusBadge status={task.status} />
            </td>
            <td>{task.environment}</td>
            <td>
              {task.provider}
              <code>{task.model}</code>
            </td>
            <td>
              <code>{task.createdAt}</code>
            </td>
            <td>
              <code>{task.duration}</code>
            </td>
          </tr>
        ))}
        {!tasks.length ? (
          <tr>
            <td colSpan={6} className="empty-cell">
              No tasks match the current filter.
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}

function NewTaskPage({
  agents,
  settings,
  localModels,
  createTask,
  cancel,
}: {
  agents: Agent[];
  settings: Settings | null;
  localModels: OllamaModel[];
  createTask: (input: {
    title: string;
    instructions: string;
    agentId: string;
    providerId: string;
    provider: ModelProvider;
    model: string;
    environment: Environment;
  }) => void;
  cancel: () => void;
}) {
  const [agentId, setAgentId] = useState(agents[0]?.id || "");
  const agent = agents.find((item) => item.id === agentId) || agents[0];
  const providerOptions = modelProviderOptions(settings);
  const defaultProviderId = agentProviderId(agent, settings);
  const [title, setTitle] = useState("");
  const [instructions, setInstructions] = useState("");
  const [providerId, setProviderId] = useState(defaultProviderId);
  const [model, setModel] = useState(agent?.defaultModel || settings?.default_model || "");
  const modelOptions = useProviderModelOptions(providerId, settings, localModels);
  const [environment, setEnvironment] = useState<Environment>(agent?.defaultEnvironment || "planner");
  const [autonomy, setAutonomy] = useState<Agent["autonomy"]>(agent?.autonomy || "ask");
  const [permissions, setPermissions] = useState<AgentPermissions>(
    agent?.permissions || { browser: true, fileRead: true, fileWrite: false, terminal: false },
  );

  function selectAgent(id: string) {
    const next = agents.find((item) => item.id === id);
    if (!next) {
      return;
    }
    setAgentId(next.id);
    setProviderId(agentProviderId(next, settings));
    setModel(next.defaultModel);
    setEnvironment(next.defaultEnvironment);
    setAutonomy(next.autonomy);
    setPermissions(next.permissions);
  }

  useEffect(() => {
    if (!modelOptions.length) {
      return;
    }
    if (!modelOptions.includes(model)) {
      setModel(modelOptions[0]);
    }
  }, [model, modelOptions]);

  function updatePermission(key: keyof AgentPermissions, value: boolean) {
    setPermissions((current) => ({ ...current, [key]: value }));
  }

  return (
    <form
      className="stack"
      onSubmit={(event) => {
        event.preventDefault();
        const selectedProvider = providerOptions.find((option) => option.id === providerId) || providerOptions[0];
        createTask({
          title,
          instructions,
          agentId,
          providerId: selectedProvider.id,
          provider: selectedProvider.label,
          model,
          environment,
        });
      }}
    >
      <section className="panel">
        <div className="section-header">
          <h2>Agent</h2>
          <p>The agent prompt is prepended to this task.</p>
        </div>
        <div className="agent-card-grid">
          {agents.map((item) => (
            <button key={item.id} type="button" className={agentId === item.id ? "agent-card active" : "agent-card"} onClick={() => selectAgent(item.id)}>
              <strong>{item.name}</strong>
              <span>{item.role}</span>
              <p>{item.description}</p>
            </button>
          ))}
        </div>
        {agent ? (
          <pre className="prompt-preview">{agent.systemPrompt}</pre>
        ) : null}
      </section>

      <section className="panel">
        <div className="section-header">
          <h2>Task Details</h2>
        </div>
        <div className="settings-form">
          <label>
            Task title
            <input value={title} onChange={(event) => setTitle(event.target.value)} required placeholder="Reconcile vendor invoices" />
          </label>
          <label>
            Instructions
            <textarea value={instructions} onChange={(event) => setInstructions(event.target.value)} required rows={5} />
          </label>
        </div>
      </section>

      <section className="two-column">
        <div className="panel">
          <div className="section-header">
            <h2>Model Provider</h2>
          </div>
          <div className="chip-row wrap">
            {providerOptions.map((item) => (
              <button
                key={item.id}
                type="button"
                className={providerId === item.id ? "chip active" : "chip"}
                onClick={() => setProviderId(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
          <label className="field-block">
            Model
            <ProviderModelSelect
              value={model}
              models={modelOptions}
              onChange={setModel}
              emptyLabel={providerId === "ollama" ? "No local Ollama models loaded" : "No catalog models for this provider"}
              title="Models are loaded from the selected provider."
            />
          </label>
        </div>

        <div className="panel">
          <div className="section-header">
            <h2>Autonomy</h2>
          </div>
          <div className="chip-row wrap">
            {autonomyLevels.map((item) => (
              <button key={item} type="button" className={autonomy === item ? "chip active" : "chip"} onClick={() => setAutonomy(item)}>
                {labelize(item)}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="section-header">
          <h2>Execution Environment</h2>
        </div>
        <div className="agent-card-grid">
          {environments.map((item) => (
            <button key={item} type="button" className={environment === item ? "agent-card active" : "agent-card"} onClick={() => setEnvironment(item)}>
              <strong>{labelize(item)}</strong>
              <span>{environmentDescription(item)}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="section-header">
          <h2>File & Workspace Permissions</h2>
        </div>
        <div className="permission-list">
          {(
            [
              ["browser", "Browser access"],
              ["fileRead", "File read"],
              ["fileWrite", "File write"],
              ["terminal", "Terminal / shell access"],
            ] as const
          ).map(([key, label]) => (
            <label key={key} className="toggle-row">
              <span>{label}</span>
              <input type="checkbox" checked={permissions[key]} onChange={(event) => updatePermission(key, event.target.checked)} />
            </label>
          ))}
        </div>
      </section>

      <div className="form-actions">
        <button type="button" onClick={cancel}>
          Cancel
        </button>
        <button className="primary" type="submit">
          Start task
        </button>
      </div>
    </form>
  );
}

function TaskDetailPage(props: {
  task: Task;
  auditLog: AuditEntry[];
  approvals: Approval[];
  back: () => void;
  updateTask: (taskId: string, patch: Partial<Task>) => void;
  addAudit: (entry: Omit<AuditEntry, "id" | "timestamp">) => void;
  resolveApproval: (approvalId: string, state: "approved" | "rejected") => void;
}) {
  const taskLog = props.auditLog.filter((entry) => entry.taskId === props.task.id);

  function taskAction(status: TaskStatus, action: string) {
    props.updateTask(props.task.id, { status, subgoal: action });
    props.addAudit({
      taskId: props.task.id,
      agent: "operator",
      action,
      environment: props.task.environment,
      result: status === "failed" ? "blocked" : "success",
      risk: status === "failed" ? "medium" : "low",
      approval: "approved",
    });
  }

  return (
    <div className="stack">
      <section className="panel detail-heading">
        <button type="button" onClick={props.back}>
          Back to tasks
        </button>
        <div>
          <code>{props.task.id}</code>
          <h2>{props.task.name}</h2>
          <p>
            {props.task.environment} / {props.task.provider} / {props.task.model}
          </p>
        </div>
        <div className="button-row">
          <StatusBadge status={props.task.status} />
          <button type="button" onClick={() => taskAction("waiting_approval", "Paused by operator")}>
            Pause
          </button>
          <button type="button" onClick={() => taskAction("running", "Resumed by operator")}>
            Resume
          </button>
          <button className="danger-outline" type="button" onClick={() => taskAction("failed", "Stopped by operator")}>
            Stop
          </button>
        </div>
      </section>

      <section className="timeline">
        {["queued", "planning", "running", "waiting_approval", "completed"].map((step) => (
          <div key={step} className={timelineClass(props.task.status, step as TaskStatus)}>
            <span />
            <strong>{labelize(step)}</strong>
          </div>
        ))}
      </section>

      <section className="two-column wide-left">
        <div className="stack">
          <InfoPanel title="Current Subgoal">{props.task.subgoal}</InfoPanel>
          <InfoPanel title="Reasoning Summary">{props.task.reasoning}</InfoPanel>
          <section className="panel preview-panel">
            <div className="section-header">
              <h2>Browser / Computer Preview</h2>
            </div>
            <div className="preview-box">Live preview unavailable until a runner is connected</div>
          </section>
          <section className="panel">
            <div className="section-header">
              <h2>Action Log</h2>
            </div>
            <AuditTable entries={taskLog} compact />
          </section>
        </div>

        <div className="stack">
          <section className="panel">
            <div className="section-header">
              <h2>Approval Requests</h2>
            </div>
            {props.approvals.length ? (
              <div className="approval-list">
                {props.approvals.map((approval) => (
                  <div key={approval.id} className="approval-card">
                    <p>{approval.action}</p>
                    <div className="button-row">
                      <StatusBadge status={approval.risk} />
                      <code>{approval.requestedAt}</code>
                    </div>
                    <div className="button-row">
                      <button className="primary" type="button" onClick={() => props.resolveApproval(approval.id, "approved")}>
                        Approve
                      </button>
                      <button type="button" onClick={() => props.resolveApproval(approval.id, "rejected")}>
                        Reject
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty">No pending approvals.</p>
            )}
          </section>
          <InfoPanel title="Artifacts">
            <div className="artifact-grid">
              {[1, 2, 3, 4].map((item) => (
                <div key={item}>Screenshot {item}</div>
              ))}
            </div>
          </InfoPanel>
          <InfoPanel title="Instructions">{props.task.instructions}</InfoPanel>
        </div>
      </section>
    </div>
  );
}

function AgentsPage({
  agents,
  setAgents,
  settings,
  localModels,
}: {
  agents: Agent[];
  setAgents: (agents: Agent[] | ((agents: Agent[]) => Agent[])) => void;
  settings: Settings | null;
  localModels: OllamaModel[];
}) {
  const [activeId, setActiveId] = useState(agents[0]?.id || "");
  const [query, setQuery] = useState("");
  const providerOptions = modelProviderOptions(settings);
  const filtered = agents.filter((agent) => [agent.name, agent.role, agent.description].join(" ").toLowerCase().includes(query.toLowerCase()));
  const active = agents.find((agent) => agent.id === activeId) || agents[0];
  const activeProviderId = agentProviderId(active, settings);
  const activeModelOptions = useProviderModelOptions(activeProviderId, settings, localModels);

  function updateAgent(patch: Partial<Agent>) {
    if (!active) {
      return;
    }
    setAgents((list) => list.map((agent) => (agent.id === active.id ? { ...agent, ...patch, updatedAt: timestampNow() } : agent)));
  }

  function createAgent() {
    const defaultProviderId = agentProviderId(undefined, settings);
    const defaultProvider = providerOptions.find((option) => option.id === defaultProviderId) || providerOptions[0];
    const agent: Agent = {
      id: `ag_${Math.random().toString(36).slice(2, 7)}`,
      name: "New agent",
      role: "Draft role",
      description: "",
      systemPrompt: "",
      defaultProviderId: defaultProvider.id,
      defaultProvider: defaultProvider.label,
      defaultModel: "default",
      defaultEnvironment: "planner",
      autonomy: "ask",
      permissions: { browser: true, fileRead: true, fileWrite: false, terminal: false },
      status: "draft",
      tasksRun: 0,
      updatedAt: timestampNow(),
    };
    setAgents((list) => [agent, ...list]);
    setActiveId(agent.id);
  }

  function removeAgent(id: string) {
    setAgents((list) => list.filter((agent) => agent.id !== id));
    if (id === activeId) {
      setActiveId(agents.find((agent) => agent.id !== id)?.id || "");
    }
  }

  useEffect(() => {
    if (!active || !activeModelOptions.length) {
      return;
    }
    if (!activeModelOptions.includes(active.defaultModel)) {
      updateAgent({ defaultModel: activeModelOptions[0] });
    }
  }, [active?.id, activeProviderId, active?.defaultModel, activeModelOptions]);

  return (
    <div className="agents-layout">
      <aside className="agent-list">
        <div className="section-header">
          <div>
            <h2>Agents</h2>
            <p>{agents.length} configured</p>
          </div>
          <button className="primary" type="button" onClick={createAgent}>
            New
          </button>
        </div>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search agents" />
        <div className="agent-buttons">
          {filtered.map((agent) => (
            <button key={agent.id} type="button" className={agent.id === active?.id ? "active" : ""} onClick={() => setActiveId(agent.id)}>
              <strong>{agent.name || "Untitled"}</strong>
              <span>{agent.role}</span>
              <StatusBadge status={agent.status} />
            </button>
          ))}
        </div>
      </aside>

      {active ? (
        <section className="panel agent-editor">
          <div className="section-header">
            <div>
              <code>{active.id}</code>
              <h2>{active.name || "Untitled agent"}</h2>
              <p>Last updated {active.updatedAt} / {active.tasksRun} tasks run</p>
            </div>
            <div className="button-row">
              <button type="button" onClick={() => updateAgent({ status: active.status === "active" ? "paused" : "active" })}>
                {active.status === "active" ? "Pause" : "Activate"}
              </button>
              <button className="danger-outline" type="button" onClick={() => removeAgent(active.id)}>
                Delete
              </button>
            </div>
          </div>

          <div className="settings-form">
            <div className="field-row">
              <label>
                Name
                <input value={active.name} onChange={(event) => updateAgent({ name: event.target.value })} />
              </label>
              <label>
                Role
                <input value={active.role} onChange={(event) => updateAgent({ role: event.target.value })} />
              </label>
            </div>
            <label>
              Description
              <input value={active.description} onChange={(event) => updateAgent({ description: event.target.value })} />
            </label>
            <label>
              System prompt
              <textarea rows={9} value={active.systemPrompt} onChange={(event) => updateAgent({ systemPrompt: event.target.value })} />
            </label>
            <div className="field-row three">
              <label>
                Provider
                <select
                  title="Choose Ollama for the direct local provider, or a saved LiteLLM provider for gateway-routed models."
                  value={activeProviderId}
                  onChange={(event) => {
                    const selected = providerOptions.find((option) => option.id === event.target.value) || providerOptions[0];
                    updateAgent({ defaultProviderId: selected.id, defaultProvider: selected.label });
                  }}
                >
                  {providerOptions.map((item) => (
                    <option key={item.id} value={item.id}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label>
                Model
                <ProviderModelSelect
                  title="Model used by this agent by default. Ollama choices come from your local Ollama inventory."
                  value={active.defaultModel}
                  models={activeModelOptions}
                  onChange={(value) => updateAgent({ defaultModel: value })}
                  emptyLabel={activeProviderId === "ollama" ? "No local Ollama models loaded" : "No catalog models for this provider"}
                />
              </label>
              <label>
                Environment
                <select value={active.defaultEnvironment} onChange={(event) => updateAgent({ defaultEnvironment: event.target.value as Environment })}>
                  {environments.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className="chip-row wrap">
              {autonomyLevels.map((item) => (
                <button key={item} type="button" className={active.autonomy === item ? "chip active" : "chip"} onClick={() => updateAgent({ autonomy: item })}>
                  {labelize(item)}
                </button>
              ))}
            </div>
            <div className="permission-list">
              {(
                [
                  ["browser", "Browser access"],
                  ["fileRead", "File read"],
                  ["fileWrite", "File write"],
                  ["terminal", "Terminal / shell access"],
                ] as const
              ).map(([key, label]) => (
                <label key={key} className="toggle-row">
                  <span>{label}</span>
                  <input
                    type="checkbox"
                    checked={active.permissions[key]}
                    onChange={(event) => updateAgent({ permissions: { ...active.permissions, [key]: event.target.checked } })}
                  />
                </label>
              ))}
            </div>
          </div>
        </section>
      ) : (
        <section className="panel empty">No agent selected.</section>
      )}
    </div>
  );
}

function SandboxesPage({
  sandboxes,
  setSandboxes,
}: {
  sandboxes: Sandbox[];
  setSandboxes: (sandboxes: Sandbox[] | ((sandboxes: Sandbox[]) => Sandbox[])) => void;
}) {
  function updateSandbox(id: string, patch: Partial<Sandbox>) {
    setSandboxes((list) => list.map((sandbox) => (sandbox.id === id ? { ...sandbox, ...patch } : sandbox)));
  }

  return (
    <div className="stack">
      <section className="panel">
        <div className="section-header">
          <h2>Sandboxes</h2>
          <p>Isolated execution environments tied to running and historical tasks.</p>
        </div>
        <div className="sandbox-grid">
          {sandboxes.map((sandbox) => (
            <div key={sandbox.id} className="sandbox-card">
              <div className="section-header">
                <div>
                  <strong>{sandbox.id}</strong>
                  <p>{sandbox.environment} / {sandbox.isolation}</p>
                </div>
                <StatusBadge status={sandbox.status} />
              </div>
              <ResourceBar label="CPU" value={sandbox.cpu} />
              <ResourceBar label="RAM" value={Math.min(100, Math.round(sandbox.ram / 40))} detail={`${sandbox.ram} MB`} />
              <p>Created <code>{sandbox.createdAt}</code></p>
              <p>Linked task <code>{sandbox.taskId || "-"}</code></p>
              <div className="button-row">
                <button type="button" disabled={sandbox.status !== "running"}>
                  Preview
                </button>
                <button type="button" disabled={sandbox.status === "stopped" || sandbox.status === "destroyed"} onClick={() => updateSandbox(sandbox.id, { status: "stopped", cpu: 0 })}>
                  Stop
                </button>
                <button className="danger-outline" type="button" onClick={() => updateSandbox(sandbox.id, { status: "destroyed", cpu: 0, ram: 0 })}>
                  Destroy
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function ModelsPage(props: {
  models: OllamaModel[];
  defaultModel: string | null;
  settings: Settings | null;
  setSettings: (settings: Settings) => void;
  providerStatuses: ProviderStatus[];
  selectDefaultModel: (model: string) => void;
  saveSettings: (settings?: Settings) => Promise<Settings | void> | void;
  applyLiteLlmProviders: (providers: LiteLlmProviderConfig[]) => Promise<ApplyLiteLlmProvidersResponse>;
  testModel: string;
  setTestModel: (value: string) => void;
  prompt: string;
  setPrompt: (value: string) => void;
  testResult: ChatResponse | null;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
}) {
  const litellmProviders = props.settings?.litellm_providers || [];
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null);
  const [providerSaveError, setProviderSaveError] = useState<string | null>(null);
  const [providerSaveMessage, setProviderSaveMessage] = useState<string | null>(null);
  const [configWarning, setConfigWarning] = useState<string | null>(null);
  const [providerSaveBusy, setProviderSaveBusy] = useState(false);
  const [providerVerifyBusyId, setProviderVerifyBusyId] = useState<string | null>(null);
  const [providerVerifications, setProviderVerifications] = useState<Record<string, ProviderVerification>>({});
  const [deleteArmedId, setDeleteArmedId] = useState<string | null>(null);
  const [replacingProviderKeyId, setReplacingProviderKeyId] = useState<string | null>(null);
  const [replacementApiKey, setReplacementApiKey] = useState("");
  const litellmStatus = props.providerStatuses.find((provider) => provider.id === "litellm");
  const ollamaStatus = props.providerStatuses.find((provider) => provider.id === "ollama");
  const defaultProvider = props.settings?.default_provider || "ollama";
  const defaultModel =
    defaultProvider === "litellm"
      ? props.settings?.litellm.default_model || props.settings?.default_model || props.defaultModel
      : props.defaultModel || props.settings?.default_model;
  const enabledProviders = litellmProviders.filter((provider) => provider.enabled).length;

  function setLiteLlmProviders(providers: LiteLlmProviderConfig[]) {
    if (!props.settings) {
      return;
    }
    props.setSettings({ ...props.settings, litellm_providers: providers });
  }

  function addProvider() {
    const provider = createLiteLlmProvider();
    setLiteLlmProviders([...litellmProviders, provider]);
    setEditingProviderId(provider.id);
  }

  function updateProvider(id: string, patch: Partial<LiteLlmProviderConfig>) {
    setProviderVerifications((results) => {
      const next = { ...results };
      delete next[id];
      return next;
    });
    setProviderSaveError(null);
    setProviderSaveMessage(null);
    setDeleteArmedId(null);
    setLiteLlmProviders(
      litellmProviders.map((provider) => {
        if (provider.id !== id) {
          return provider;
        }
        return { ...provider, ...patch };
      }),
    );
  }

  function removeProvider(id: string) {
    if (!isDraftProviderId(id) && deleteArmedId !== id) {
      setDeleteArmedId(id);
      return;
    }
    setLiteLlmProviders(litellmProviders.filter((provider) => provider.id !== id));
    setProviderVerifications((results) => {
      const next = { ...results };
      delete next[id];
      return next;
    });
    setDeleteArmedId(null);
    if (editingProviderId === id) {
      setEditingProviderId(null);
    }
    if (replacingProviderKeyId === id) {
      setReplacingProviderKeyId(null);
      setReplacementApiKey("");
    }
  }

  function startReplacingProviderKey(id: string) {
    setProviderSaveError(null);
    setDeleteArmedId(null);
    setReplacingProviderKeyId(id);
    setReplacementApiKey("");
  }

  function cancelReplacingProviderKey() {
    setReplacingProviderKeyId(null);
    setReplacementApiKey("");
  }

  function applyReplacementProviderKey(id: string) {
    const value = replacementApiKey.trim();
    if (!value) {
      setProviderSaveError("Enter a replacement API key before applying it.");
      return;
    }
    updateProvider(id, { api_key: value });
    setReplacingProviderKeyId(null);
    setReplacementApiKey("");
  }

  async function verifyProvider(provider: LiteLlmProviderConfig) {
    const providerType = normalizeProviderType(provider.provider_type);
    if (!providerType) {
      setProviderVerifications((results) => ({
        ...results,
        [provider.id]: { state: "error", message: "Choose a provider type before verifying." },
      }));
      return;
    }
    const model = defaultProviderVerificationModel(
      providerType,
      props.models.map((model) => model.name),
    );
    if (!model) {
      setProviderVerifications((results) => ({
        ...results,
        [provider.id]: { state: "error", message: "No default test model is known for this provider type." },
      }));
      return;
    }

    const prepared = prepareProvidersForSave([provider]);
    if ("error" in prepared) {
      setProviderVerifications((results) => ({
        ...results,
        [provider.id]: { state: "error", message: prepared.error },
      }));
      return;
    }

    const draftProvider = prepared.providers[0];
    setProviderVerifyBusyId(provider.id);
    setProviderVerifications((results) => {
      const next = { ...results };
      delete next[provider.id];
      return next;
    });
    try {
      const result = await api.testLiteLLMProvider(
        model,
        "Reply with only OK.",
        draftProvider.id,
        { provider: draftProvider },
      );
      setProviderVerifications((results) => ({
        ...results,
        [provider.id]: {
          state: "success",
          message: result.content.trim() || `Verified with ${generatedLiteLlmModel(draftProvider, model)}.`,
        },
      }));
    } catch (err) {
      setProviderVerifications((results) => ({
        ...results,
        [provider.id]: { state: "error", message: (err as Error).message },
      }));
    } finally {
      setProviderVerifyBusyId(null);
    }
  }

  async function saveProviders() {
    if (!props.settings) {
      return;
    }

    const prepared = prepareProvidersForSave(litellmProviders);
    if ("error" in prepared) {
      setProviderSaveError(prepared.error);
      return;
    }

    setProviderSaveBusy(true);
    setProviderSaveError(null);
    setProviderSaveMessage(null);
    setConfigWarning(null);
    try {
      const result = await props.applyLiteLlmProviders(prepared.providers);
      if (result.litellm_ready) {
        setProviderSaveMessage("Providers saved. LiteLLM is ready.");
      } else {
        setProviderSaveMessage("Providers saved. LiteLLM is not ready yet.");
      }
      setConfigWarning(result.warning || null);
      setReplacingProviderKeyId(null);
      setReplacementApiKey("");
    } catch (err) {
      setProviderSaveError((err as Error).message);
    } finally {
      setProviderSaveBusy(false);
    }
  }

  return (
    <div className="stack">
      <section className="model-summary-strip">
        <div className="model-summary-tile">
          <span>Default provider</span>
          <strong>{defaultProviderLabel(defaultProvider, props.settings)}</strong>
          <small>{defaultModel || "not set"}</small>
        </div>
        <div className="model-summary-tile">
          <span>LiteLLM gateway</span>
          <div className="summary-status">
            <StatusBadge status={litellmStatus?.healthy ? "online" : props.settings?.litellm.enabled ? "offline" : "neutral"} />
          </div>
          <small>{props.settings?.litellm.base_url || "not configured"}</small>
        </div>
        <div className="model-summary-tile">
          <span>Providers</span>
          <strong>{enabledProviders}/{litellmProviders.length}</strong>
          <small>enabled</small>
        </div>
        <div className="model-summary-tile">
          <span>Ollama</span>
          <div className="summary-status">
            <StatusBadge status={ollamaStatus?.healthy ? "online" : "offline"} />
          </div>
          <small>{props.models.length} local models</small>
        </div>
      </section>

      {props.settings ? (
        <section className="panel">
          <div className="section-header">
            <div>
              <h2>
                LiteLLM Providers
                <FieldHelp text="Save writes provider secrets, regenerates LiteLLM config, and starts the app-managed gateway when needed." />
              </h2>
              <p>{enabledProviders} enabled / {litellmProviders.length} total</p>
            </div>
            <div className="button-row">
              <StatusBadge status={litellmStatus?.healthy ? "online" : props.settings.litellm.enabled ? "offline" : "neutral"} />
              <button type="button" onClick={addProvider} disabled={props.busy}>
                Add provider
              </button>
            </div>
          </div>

          <div className="provider-list">
            <div className="provider-list-header" aria-hidden="true">
              <span>Provider</span>
              <span>Credential</span>
              <span>Status</span>
            </div>
            {litellmProviders.map((provider) => {
              const expanded = editingProviderId === provider.id;
              const status = props.providerStatuses.find((item) => item.id === provider.id);
              const verification = providerVerifications[provider.id];
              const providerType = normalizeProviderType(provider.provider_type);
              const providerIsDraft = isDraftProvider(provider);
              const keyConfigured = provider.api_key === REDACTED_SECRET || Boolean(status?.api_key_configured);
              const apiKeyValue = provider.api_key && provider.api_key !== REDACTED_SECRET ? provider.api_key : "";
              const replacingProviderKey = replacingProviderKeyId === provider.id;
              const deleteLabel = providerIsDraft ? "Cancel provider" : deleteArmedId === provider.id ? "Confirm delete" : "Delete provider";
              return (
                <article key={provider.id} className={expanded ? "provider-row expanded" : "provider-row"}>
                  <button type="button" className="provider-row-summary" onClick={() => setEditingProviderId(expanded ? null : provider.id)}>
                    <span>
                      <strong>{provider.display_name || "Unnamed provider"}</strong>
                      <small>{provider.provider_type ? providerTypeLabel(provider.provider_type) : "Provider type not set"}</small>
                    </span>
                    <span>{providerCredentialLabel(provider, status)}</span>
                    <StatusBadge status={provider.enabled ? "active" : "neutral"} />
                  </button>
                  {expanded ? (
                    <div className="provider-row-editor">
                      <div className="field-row three">
                        <label>
                          Name
                          <input
                            title="A human-readable label used in provider lists and agent defaults."
                            value={provider.display_name}
                            onChange={(event) => updateProvider(provider.id, { display_name: event.target.value })}
                            placeholder="OpenAI work"
                          />
                        </label>
                        <label>
                          Provider type
                          {providerIsDraft ? (
                            <ProviderTypeCombobox
                              value={provider.provider_type}
                              onChange={(value) => {
                                updateProvider(provider.id, {
                                  provider_type: value,
                                  api_key: normalizeProviderType(value) === "ollama" ? null : provider.api_key,
                                });
                              }}
                            />
                          ) : (
                            <div className="readonly-field">{provider.provider_type ? providerTypeLabel(provider.provider_type) : "Provider type not set"}</div>
                          )}
                        </label>
                        <label>
                          <span className="field-label">
                            {providerIsDraft ? "API key" : "API Base"}
                            {providerIsDraft ? (
                              <FieldHelp text="Saved to the Llama Harness app data directory and passed to LiteLLM through environment variables." />
                            ) : (
                              <FieldHelp text="Optional provider-specific endpoint. Use it for Ollama behind LiteLLM, self-hosted providers, or a custom proxy; leave it blank for normal hosted provider defaults." />
                            )}
                          </span>
                          {providerIsDraft ? (
                            <input
                              type="password"
                              value={apiKeyValue}
                              onChange={(event) => updateProvider(provider.id, { api_key: event.target.value || null })}
                              placeholder={providerType === "ollama" ? "not required" : keyConfigured ? "configured" : "Paste API key"}
                              disabled={providerType === "ollama"}
                            />
                          ) : (
                            <input
                              value={provider.api_base || ""}
                              onChange={(event) => updateProvider(provider.id, { api_base: event.target.value || null })}
                              placeholder={normalizeProviderType(provider.provider_type) === "ollama" ? "http://localhost:11434" : "Optional"}
                            />
                          )}
                        </label>
                      </div>
                      {!providerIsDraft && providerType !== "ollama" && replacingProviderKey ? (
                        <div className="provider-key-replacement">
                          <label>
                            New API key
                            <input
                              type="password"
                              value={replacementApiKey}
                              onChange={(event) => setReplacementApiKey(event.target.value)}
                              placeholder="Paste replacement API key"
                            />
                          </label>
                          <div className="field-actions">
                            <button type="button" onClick={() => applyReplacementProviderKey(provider.id)} disabled={props.busy}>
                              Use new key
                            </button>
                            <button type="button" onClick={cancelReplacingProviderKey} disabled={props.busy}>
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : null}
                      <div className="field-row three">
                        {providerIsDraft ? (
                          <label>
                            <span className="field-label">
                              API Base
                              <FieldHelp text="Optional provider-specific endpoint. Use it for Ollama behind LiteLLM, self-hosted providers, or a custom proxy; leave it blank for normal hosted provider defaults." />
                            </span>
                            <input
                              value={provider.api_base || ""}
                              onChange={(event) => updateProvider(provider.id, { api_base: event.target.value || null })}
                              placeholder={normalizeProviderType(provider.provider_type) === "ollama" ? "http://localhost:11434" : "Optional"}
                            />
                          </label>
                        ) : (
                          <div />
                        )}
                        <label className="switch-row provider-enabled-toggle" title="Disabled providers stay saved but are omitted from generated LiteLLM config and agent choices.">
                          <input
                            type="checkbox"
                            checked={provider.enabled}
                            onChange={(event) => updateProvider(provider.id, { enabled: event.target.checked })}
                          />
                          <span className="switch-track" aria-hidden="true">
                            <span className="switch-thumb" />
                          </span>
                          <span>{provider.enabled ? "Enabled" : "Disabled"}</span>
                        </label>
                        <div className="field-actions">
                          {!providerIsDraft && providerType !== "ollama" && !replacingProviderKey ? (
                            <button type="button" onClick={() => startReplacingProviderKey(provider.id)} disabled={props.busy}>
                              {apiKeyValue ? "Edit pending key" : "Replace API key"}
                            </button>
                          ) : null}
                          <button
                            type="button"
                            onClick={() => verifyProvider(provider)}
                            disabled={props.busy || providerVerifyBusyId === provider.id}
                            title="Calls the LiteLLM gateway with the current provider values and a small test prompt."
                          >
                            {providerVerifyBusyId === provider.id ? "Testing..." : "Test Connection"}
                          </button>
                          <button
                            className={deleteArmedId === provider.id ? "danger" : "danger-outline"}
                            type="button"
                            onClick={() => removeProvider(provider.id)}
                            disabled={props.busy}
                            title="Requires a second click before this provider is removed."
                          >
                            {deleteLabel}
                          </button>
                        </div>
                      </div>
                      {verification ? (
                        <pre className={verification.state === "success" ? "result success-result" : "result error-result"}>
                          {verification.message}
                        </pre>
                      ) : null}
                    </div>
                  ) : null}
                </article>
              );
            })}
            {!litellmProviders.length ? <p className="empty">No providers configured.</p> : null}
          </div>

          <div className="form-actions route-save-row">
            <button className="primary" type="button" onClick={saveProviders} disabled={props.busy || providerSaveBusy}>
              Save providers
            </button>
          </div>
          {providerSaveError ? <pre className="result error-result">{providerSaveError}</pre> : null}
          {providerSaveMessage ? <pre className="result success-result">{providerSaveMessage}</pre> : null}
          {configWarning ? <pre className="result warning-result">Saved providers, but config generation failed: {configWarning}</pre> : null}
        </section>
      ) : null}

      <section className="panel">
        <div className="section-header">
          <div>
            <h2>Local Ollama Inventory</h2>
            <p>{props.models.length} models returned by Ollama</p>
          </div>
        </div>
        <div className="table-scroll">
          <table className="model-table">
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
                  <td data-label="Name">{model.name}</td>
                  <td data-label="Family">{model.details?.family || "-"}</td>
                  <td data-label="Size">{formatBytes(model.size)}</td>
                  <td data-label="Quantization">{model.details?.quantization_level || "-"}</td>
                  <td data-label="Default">{props.defaultModel === model.name ? "yes" : "no"}</td>
                  <td data-label="Action">
                    <button type="button" onClick={() => props.selectDefaultModel(model.name)} disabled={props.busy}>
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
        </div>
      </section>

      <section className="panel">
        <div className="section-header">
          <h2>Test Selected Model</h2>
        </div>
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

type ComboboxMenuPlacement = {
  top: number;
  left: number;
  width: number;
  maxHeight: number;
};

function ProviderTypeCombobox({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const [open, setOpen] = useState(false);
  const [menuPlacement, setMenuPlacement] = useState<ComboboxMenuPlacement | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) {
      return;
    }

    function closeOnOutsidePointer(event: PointerEvent) {
      const target = event.target;
      if (target instanceof Node && rootRef.current?.contains(target)) {
        return;
      }
      if (target instanceof Node && menuRef.current?.contains(target)) {
        return;
      }
      setOpen(false);
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("pointerdown", closeOnOutsidePointer, true);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer, true);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  useLayoutEffect(() => {
    if (!open) {
      return;
    }

    function updateMenuPlacement() {
      const rect = rootRef.current?.getBoundingClientRect();
      if (!rect) {
        return;
      }

      const viewportPadding = 12;
      const menuGap = 6;
      const availableBelow = window.innerHeight - rect.bottom - viewportPadding - menuGap;
      const availableAbove = rect.top - viewportPadding - menuGap;
      const openAbove = availableBelow < 220 && availableAbove > availableBelow;
      const rawAvailableHeight = openAbove ? availableAbove : availableBelow;
      const maxHeight = Math.max(120, Math.min(560, rawAvailableHeight));
      const width = Math.min(rect.width, window.innerWidth - viewportPadding * 2);
      const left = Math.min(
        Math.max(viewportPadding, rect.left),
        Math.max(viewportPadding, window.innerWidth - width - viewportPadding),
      );
      const top = openAbove
        ? Math.max(viewportPadding, rect.top - menuGap - maxHeight)
        : Math.min(rect.bottom + menuGap, window.innerHeight - viewportPadding - maxHeight);

      setMenuPlacement({ top, left, width, maxHeight });
    }

    updateMenuPlacement();
    window.addEventListener("resize", updateMenuPlacement);
    window.addEventListener("scroll", updateMenuPlacement, true);
    return () => {
      window.removeEventListener("resize", updateMenuPlacement);
      window.removeEventListener("scroll", updateMenuPlacement, true);
    };
  }, [open]);

  const menu = open && menuPlacement
    ? createPortal(
        <div className="combobox-menu" ref={menuRef} role="listbox" style={menuPlacement}>
          {litellmProviderTypes.map((providerType) => (
            <button
              key={providerType}
              type="button"
              className={normalizeProviderType(value) === providerType ? "active" : ""}
              onClick={() => {
                onChange(providerType);
                setOpen(false);
              }}
              role="option"
              aria-selected={normalizeProviderType(value) === providerType}
            >
              <span>{providerTypeLabel(providerType)}</span>
            </button>
          ))}
        </div>,
        document.body,
      )
    : null;

  return (
    <div className={open ? "combobox open" : "combobox"} ref={rootRef}>
      <button
        className="combobox-trigger"
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Show provider types"
      >
        <span>{value ? providerTypeLabel(value) : "Select provider"}</span>
        <span className="combobox-caret" aria-hidden="true">v</span>
      </button>
      {menu}
    </div>
  );
}

function ProviderModelSelect({
  value,
  models,
  onChange,
  emptyLabel,
  title,
}: {
  value: string;
  models: string[];
  onChange: (value: string) => void;
  emptyLabel: string;
  title?: string;
}) {
  return (
    <select value={value && models.includes(value) ? value : ""} onChange={(event) => onChange(event.target.value)} disabled={!models.length} title={title}>
      <option value="" disabled>
        {models.length ? "Select model" : emptyLabel}
      </option>
      {models.map((model) => (
        <option key={model} value={model}>
          {model}
        </option>
      ))}
    </select>
  );
}

function createLiteLlmProvider(): LiteLlmProviderConfig {
  return {
    id: `draft_${Date.now().toString(36)}`,
    enabled: true,
    provider_type: "",
    display_name: "",
    api_key_env_var: "",
    api_key: null,
    api_base: null,
  };
}

function generatedLiteLlmModel(provider: LiteLlmProviderConfig, model: string): string {
  const providerType = litellmModelPrefix(provider.provider_type);
  const trimmedModel = model.trim();
  if (!providerType || !trimmedModel) {
    return "";
  }
  if (trimmedModel.startsWith(`${providerType}/`)) {
    return trimmedModel;
  }
  return `${providerType}/${trimmedModel}`;
}

function litellmModelPrefix(providerType: string): string {
  const normalized = normalizeProviderType(providerType);
  return normalized === "ollama" ? "ollama_chat" : normalized;
}

function prepareProvidersForSave(providers: LiteLlmProviderConfig[]): { providers: LiteLlmProviderConfig[] } | { error: string } {
  const usedNames = new Set<string>();
  const usedIds = new Set<string>();
  const usedEnvVars = new Set<string>();
  const prepared: LiteLlmProviderConfig[] = [];

  for (const provider of providers) {
    const displayName = provider.display_name.trim();
    const providerType = normalizeProviderType(provider.provider_type);

    if (!displayName) {
      return { error: "Provider name is required." };
    }
    const nameKey = displayName.toLowerCase();
    if (usedNames.has(nameKey)) {
      return { error: `Provider name "${displayName}" is already used.` };
    }
    if (!providerType) {
      return { error: `Provider type is required for "${displayName}".` };
    }
    if (providerType !== "ollama" && provider.enabled) {
      const rawApiKey = provider.api_key && provider.api_key !== REDACTED_SECRET ? provider.api_key.trim() : "";
      const configuredApiKey = provider.api_key === REDACTED_SECRET;
      if (!rawApiKey && !configuredApiKey) {
        return { error: `API key is required for "${displayName}".` };
      }
    }

    usedNames.add(nameKey);
    const currentId = normalizeProviderId(provider.id);
    const id = currentId && !currentId.startsWith("draft_") ? currentId : uniqueProviderId(displayName, usedIds);
    if (usedIds.has(id)) {
      return { error: `Provider id "${id}" is already used.` };
    }
    const apiKeyEnvVar = providerType === "ollama" ? "" : provider.api_key_env_var.trim() || apiKeyEnvVarForProviderId(id);
    if (apiKeyEnvVar) {
      if (usedEnvVars.has(apiKeyEnvVar)) {
        return { error: `Provider API key environment variable "${apiKeyEnvVar}" is already used.` };
      }
      usedEnvVars.add(apiKeyEnvVar);
    }
    usedIds.add(id);
    prepared.push({
      ...provider,
      id,
      provider_type: providerType,
      display_name: displayName,
      api_key_env_var: apiKeyEnvVar,
      api_base: provider.api_base?.trim() || null,
    });
  }

  return { providers: prepared };
}

function uniqueProviderId(displayName: string, usedIds: Set<string>): string {
  const base = normalizeProviderId(displayName) || `provider_${Date.now().toString(36)}`;
  if (!usedIds.has(base)) {
    return base;
  }
  for (let index = 2; ; index += 1) {
    const candidate = `${base}_${index}`;
    if (!usedIds.has(candidate)) {
      return candidate;
    }
  }
}

function apiKeyEnvVarForProviderId(providerId: string): string {
  const slug = providerId
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return `${slug || "PROVIDER"}_API_KEY`;
}

function isDraftProvider(provider: LiteLlmProviderConfig): boolean {
  return isDraftProviderId(provider.id);
}

function isDraftProviderId(id: string): boolean {
  return normalizeProviderId(id).startsWith("draft_");
}

function localProviderModelCatalog(provider: LiteLlmProviderConfig): ProviderModelsResponse {
  const providerType = normalizeProviderType(provider.provider_type);
  const names = suggestedModelNames(providerType);
  return {
    provider_id: provider.id,
    provider_type: providerType,
    models: names.map((name) => ({
      name,
      litellm_model: generatedLiteLlmModel(provider, name),
      source: "catalog",
    })),
  };
}

function suggestedModelNames(providerType: string): string[] {
  switch (normalizeProviderType(providerType)) {
    case "openai":
      return ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3-mini"];
    case "anthropic":
      return ["claude-sonnet-4-0", "claude-opus-4-0", "claude-3-5-haiku-latest"];
    case "gemini":
      return ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"];
    case "openrouter":
      return ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-0", "google/gemini-2.5-pro"];
    case "ollama":
      return ["llama3.2", "qwen2.5:7b", "mistral"];
    default:
      return [];
  }
}

function defaultProviderVerificationModel(providerType: string, localModelNames: string[] = []): string {
  const normalized = normalizeProviderType(providerType);
  const catalog = normalized === "ollama" && localModelNames.length ? localModelNames : suggestedModelNames(normalized);
  const preferred = preferredVerificationModel(normalized);
  if (preferred && catalog.includes(preferred)) {
    return preferred;
  }
  return [...catalog].sort((left, right) => left.localeCompare(right))[0] || "";
}

function preferredVerificationModel(providerType: string): string {
  switch (normalizeProviderType(providerType)) {
    case "openai":
      return "gpt-4o-mini";
    case "anthropic":
      return "claude-3-5-haiku-latest";
    case "gemini":
      return "gemini-2.5-flash";
    case "openrouter":
      return "openai/gpt-4o-mini";
    default:
      return "";
  }
}

function providerCredentialLabel(provider: LiteLlmProviderConfig, status?: ProviderStatus): string {
  if (normalizeProviderType(provider.provider_type) === "ollama") {
    return "not required";
  }
  if (provider.api_key && provider.api_key !== REDACTED_SECRET) {
    return "pending save";
  }
  if (provider.api_key === REDACTED_SECRET || status?.api_key_configured) {
    return "configured";
  }
  return "not configured";
}

function providerTypeLabel(providerType: string): string {
  const normalized = normalizeProviderType(providerType);
  switch (normalized) {
    case "openai":
      return "OpenAI";
    case "anthropic":
      return "Anthropic";
    case "gemini":
      return "Gemini";
    case "openrouter":
      return "OpenRouter";
    case "ollama":
      return "Ollama";
    case "azure":
      return "Azure";
    case "bedrock":
      return "Bedrock";
    case "vertex_ai":
      return "Vertex AI";
    case "cohere":
      return "Cohere";
    case "mistral":
      return "Mistral";
    case "groq":
      return "Groq";
    case "deepseek":
      return "DeepSeek";
    case "xai":
      return "xAI";
    case "perplexity":
      return "Perplexity";
    case "together_ai":
      return "Together AI";
    case "fireworks_ai":
      return "Fireworks AI";
    case "huggingface":
      return "Hugging Face";
    case "replicate":
      return "Replicate";
    case "custom":
      return "Custom";
    default:
      return labelize(normalized);
  }
}

function defaultProviderLabel(providerId: string, settings: Settings | null): string {
  if (providerId === "ollama") {
    return "Ollama";
  }
  if (providerId === "litellm") {
    return "LiteLLM";
  }
  const provider = settings?.litellm_providers.find((item) => item.id === providerId);
  return provider?.display_name || providerTypeLabel(provider?.provider_type || providerId);
}

function normalizeProviderType(value: string): string {
  return value.trim().toLowerCase().replace(/[\s-]+/g, "_");
}

function normalizeProviderId(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function isEditableElement(element: Element | null): element is HTMLElement {
  if (!(element instanceof HTMLElement)) {
    return false;
  }
  if (element.isContentEditable) {
    return true;
  }
  return ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
}

function useProviderModelOptions(providerId: string, settings: Settings | null, localModels: OllamaModel[] = []): string[] {
  const [modelState, setModelState] = useState<{ providerId: string; models: string[] }>({ providerId: "", models: [] });

  useEffect(() => {
    let cancelled = false;
    if (!providerId) {
      setModelState({ providerId: "", models: [] });
      return;
    }

    const provider = settings?.litellm_providers.find((item) => item.id === providerId);
    const fallback =
      providerId === "ollama"
        ? localModels.map((model) => model.name).filter(Boolean)
        : provider
          ? localProviderModelCatalog(provider).models.map((model) => model.name)
          : [];
    setModelState({ providerId, models: fallback });

    api
      .providerModels(providerId)
      .then((response) => {
        if (!cancelled) {
          setModelState({ providerId, models: response.models.map((model) => model.name) });
        }
      })
      .catch(() => {
        if (!cancelled) {
          setModelState({ providerId, models: fallback });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [providerId, settings?.litellm_providers, localModels]);

  return modelState.providerId === providerId ? modelState.models : [];
}

function modelProviderOptions(settings: Settings | null): Array<{ id: string; label: string; providerType: string }> {
  const options = [{ id: "ollama", label: "Ollama", providerType: "ollama" }];
  if (!settings) {
    return options;
  }
  const reservedProviderIds = new Set(options.map((option) => option.id).concat("litellm"));

  return [
    ...options,
    ...settings.litellm_providers
      .filter((provider) => provider.enabled)
      .filter((provider) => !reservedProviderIds.has(provider.id))
      .map((provider) => ({
        id: provider.id,
        label: provider.display_name || providerTypeLabel(provider.provider_type),
        providerType: provider.provider_type,
      })),
  ];
}

function agentProviderId(agent: Agent | undefined, settings: Settings | null): string {
  const options = modelProviderOptions(settings);
  const isValidProviderId = (providerId: string) => options.some((option) => option.id === providerId);
  const fallback = isValidProviderId(settings?.default_provider || "") ? settings?.default_provider || "ollama" : "ollama";

  if (agent?.defaultProviderId) {
    return isValidProviderId(agent.defaultProviderId) ? agent.defaultProviderId : fallback;
  }
  if (agent?.defaultProvider) {
    const legacyProvider = normalizeProviderType(agent.defaultProvider);
    const configuredProvider = settings?.litellm_providers.find(
      (provider) => normalizeProviderType(provider.provider_type) === legacyProvider,
    );
    const providerId = configuredProvider?.id || (legacyProvider === "ollama" ? "ollama" : fallback);
    return isValidProviderId(providerId) ? providerId : fallback;
  }
  return fallback;
}

function modelPlaceholderForProvider(providerId: string, settings: Settings | null): string {
  if (!settings || providerId === "ollama") {
    return settings?.default_model || "default";
  }
  const provider = settings.litellm_providers.find((item) => item.id === providerId);
  if (!provider) {
    return settings.default_model || "default";
  }
  const suggestions = suggestedModelNames(provider.provider_type);
  return suggestions[0] || "model name";
}

function PermissionsPage({
  policy,
  setPolicy,
}: {
  policy: PermissionPolicy;
  setPolicy: (policy: PermissionPolicy | ((policy: PermissionPolicy) => PermissionPolicy)) => void;
}) {
  function update(id: string, patch: Partial<PermissionPolicy[string]>) {
    setPolicy((current) => ({
      ...current,
      [id]: {
        ...current[id],
        ...patch,
      },
    }));
  }

  return (
    <section className="panel">
      <div className="section-header">
        <h2>Permissions Policy</h2>
        <p>Default capabilities granted to agents, and which actions require human approval.</p>
      </div>
      <table>
        <thead>
          <tr>
            <th>Capability</th>
            <th>Allowed</th>
            <th>Require approval</th>
          </tr>
        </thead>
        <tbody>
          {permissionRows.map((row) => (
            <tr key={row.id}>
              <td>
                <strong>{row.label}</strong>
                <p>{row.description}</p>
              </td>
              <td>
                <input
                  type="checkbox"
                  checked={policy[row.id]?.allowed || false}
                  onChange={(event) => update(row.id, { allowed: event.target.checked })}
                />
              </td>
              <td>
                <input
                  type="checkbox"
                  checked={policy[row.id]?.approval || false}
                  disabled={!policy[row.id]?.allowed}
                  onChange={(event) => update(row.id, { approval: event.target.checked })}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function LogsPage({ auditLog, runs }: { auditLog: AuditEntry[]; runs: RunRecord[] }) {
  const [query, setQuery] = useState("");
  const filtered = auditLog.filter((entry) =>
    [entry.action, entry.taskId, entry.agent, entry.environment, entry.result, entry.risk, entry.approval]
      .join(" ")
      .toLowerCase()
      .includes(query.toLowerCase()),
  );

  function exportCsv() {
    const header = "timestamp,taskId,agent,action,environment,result,risk,approval";
    const rows = filtered.map((entry) =>
      [
        entry.timestamp,
        entry.taskId,
        entry.agent,
        entry.action,
        entry.environment,
        entry.result,
        entry.risk,
        entry.approval,
      ]
        .map((value) => `"${String(value).replace(/"/g, '""')}"`)
        .join(","),
    );
    const url = URL.createObjectURL(new Blob([[header, ...rows].join("\n")], { type: "text/csv" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "llama-harness-audit-log.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="stack">
      <section className="panel toolbar-panel">
        <div>
          <h2>Audit Log</h2>
          <p>Every agent action, classified by risk and approval state.</p>
        </div>
        <div className="filter-row">
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search actions, tasks, agents" />
          <button type="button" onClick={exportCsv}>
            Export CSV
          </button>
        </div>
      </section>
      <section className="panel">
        <AuditTable entries={filtered} />
      </section>
      <section className="panel">
        <div className="section-header">
          <h2>Model Runs</h2>
          <p>Existing llama-harness run log.</p>
        </div>
        <RunsTable runs={runs} />
      </section>
    </div>
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
        <label className="toggle-row inline">
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
  providerStatuses: ProviderStatus[];
  apiBaseInput: string;
  setApiBaseInput: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  testLiteLlmConnection: () => void;
  generateLiteLlmConfig: () => void;
  startLiteLlmService: () => void;
  litellmTestResult: LiteLlmTestResponse | null;
  litellmConfigResult: GenerateLiteLlmConfigResponse | null;
  litellmServiceResult: LiteLlmServiceStartResponse | null;
  busy: boolean;
}) {
  const { settings, setSettings } = props;
  const litellmStatus = props.providerStatuses.find((provider) => provider.id === "litellm");
  const apiKeyConfigured = settings.litellm.api_key === REDACTED_SECRET;
  const apiKeyValue = apiKeyConfigured ? "" : settings.litellm.api_key || "";

  function updateGeneration<K extends keyof Settings["generation"]>(key: K, value: number) {
    setSettings({
      ...settings,
      generation: {
        ...settings.generation,
        [key]: value,
      },
    });
  }

  function updateLiteLlm(next: Partial<Settings["litellm"]>) {
    setSettings({
      ...settings,
      litellm: {
        ...settings.litellm,
        ...next,
      },
    });
  }

  return (
    <div className="stack">
      <section className="panel">
        <div className="section-header">
          <div>
            <h2>App Runtime</h2>
            <p>Connection, display, logging, and fallback generation behavior.</p>
          </div>
        </div>
        <form className="settings-form" onSubmit={props.onSubmit}>
          <div className="settings-section">
            <div>
              <h3>App connection</h3>
              <p>Agent model selection lives on Agents. Provider credentials live on Providers.</p>
            </div>
            <div className="field-row three">
              <label>
                <span className="field-label">
                  API base URL
                  <FieldHelp text="The llama-harness API endpoint used by this admin UI." />
                </span>
                <input value={props.apiBaseInput} onChange={(event) => props.setApiBaseInput(event.target.value)} />
              </label>
              <label>
                Theme
                <select value={settings.theme} onChange={(event) => setSettings({ ...settings, theme: event.target.value })}>
                  <option value="dark">dark</option>
                  <option value="light">light</option>
                </select>
              </label>
              <label className="switch-row settings-switch">
                <input
                  type="checkbox"
                  checked={settings.logging_enabled}
                  onChange={(event) => setSettings({ ...settings, logging_enabled: event.target.checked })}
                />
                <span className="switch-track" aria-hidden="true">
                  <span className="switch-thumb" />
                </span>
                <span className="settings-switch-copy">
                  JSONL run logging
                  <small>Append local run history when enabled.</small>
                </span>
              </label>
            </div>
          </div>

          <div className="settings-section">
            <div>
              <h3>Generation fallback</h3>
              <p>Used only when a request or agent does not provide its own generation settings.</p>
            </div>
            <div className="field-row three">
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
          </div>

          <div className="form-actions">
            <button className="primary" type="submit" disabled={props.busy}>
              Save runtime settings
            </button>
          </div>
        </form>
      </section>

      <section className="panel">
        <div className="section-header">
          <div>
            <h2>LiteLLM Gateway Runtime</h2>
            <p>{litellmStatus?.base_url || settings.litellm.base_url}</p>
          </div>
          <StatusBadge status={litellmStatus?.healthy ? "online" : settings.litellm.enabled ? "offline" : "neutral"} />
        </div>
        <form className="settings-form" onSubmit={props.onSubmit}>
          <div className="settings-section">
            <div>
              <h3>Gateway controls</h3>
              <p>Provider API keys are configured on Providers. Local Ollama routes do not require provider keys.</p>
            </div>
            <div className="field-row">
              <label className="switch-row settings-switch">
                <input type="checkbox" checked={settings.litellm.enabled} onChange={(event) => updateLiteLlm({ enabled: event.target.checked })} />
                <span className="switch-track" aria-hidden="true">
                  <span className="switch-thumb" />
                </span>
                <span className="settings-switch-copy">
                  Enable LiteLLM
                  <small>Route configured gateway providers through the local proxy.</small>
                </span>
              </label>
              <label>
                Base URL
                <input value={settings.litellm.base_url} onChange={(event) => updateLiteLlm({ base_url: event.target.value })} />
              </label>
            </div>
            <div className="button-row">
              <button type="button" onClick={props.testLiteLlmConnection} disabled={props.busy}>
                Test connection
              </button>
              <button type="button" onClick={props.generateLiteLlmConfig} disabled={props.busy}>
                Generate config.yaml
              </button>
              <button type="button" onClick={props.startLiteLlmService} disabled={props.busy}>
                Start LiteLLM
              </button>
              <button className="primary" type="submit" disabled={props.busy}>
                Save gateway
              </button>
            </div>
          </div>

          <details className="advanced-settings">
            <summary>
              <span>
                Advanced LiteLLM
                <small>Proxy auth, process config, timeout, and raw model routing.</small>
              </span>
            </summary>
            <div className="advanced-settings-body">
              <div className="field-row three">
                <label>
                  <span className="field-label">
                    Proxy master key
                    <FieldHelp text="Optional auth for the local LiteLLM proxy. This is separate from provider API keys managed on Providers." />
                  </span>
                  <input
                    type="password"
                    value={apiKeyValue}
                    onChange={(event) => updateLiteLlm({ api_key: event.target.value || (apiKeyConfigured ? REDACTED_SECRET : null) })}
                    placeholder={apiKeyConfigured ? "configured" : "auto-managed"}
                  />
                </label>
                <label>
                  Timeout ms
                  <input
                    type="number"
                    min="1000"
                    step="1000"
                    value={settings.litellm.timeout_ms}
                    onChange={(event) => updateLiteLlm({ timeout_ms: Number(event.target.value) })}
                  />
                </label>
                <label className="switch-row settings-switch">
                  <input
                    type="checkbox"
                    checked={settings.litellm.allow_unconfigured_models}
                    onChange={(event) => updateLiteLlm({ allow_unconfigured_models: event.target.checked })}
                  />
                  <span className="switch-track" aria-hidden="true">
                    <span className="switch-thumb" />
                  </span>
                  <span className="settings-switch-copy">
                    Allow raw models
                    <small>Permit model names not covered by saved providers or routes.</small>
                  </span>
                </label>
              </div>
              <label>
                Managed config path
                <input
                  value={settings.litellm.managed_config_path || ""}
                  onChange={(event) => updateLiteLlm({ managed_config_path: event.target.value || null })}
                  placeholder="litellm.config.yaml"
                />
              </label>
            </div>
          </details>

          {props.litellmTestResult ? <pre className="result">{props.litellmTestResult.content || "(empty response)"}</pre> : null}
          {props.litellmConfigResult ? (
            <pre className="result">{`${props.litellmConfigResult.providers_written || 0} providers written\n${props.litellmConfigResult.entries_written || props.litellmConfigResult.routes_written} total entries\n${props.litellmConfigResult.path}`}</pre>
          ) : null}
          {props.litellmServiceResult ? (
            <pre className="result success-result">{`${labelize(props.litellmServiceResult.status)}\n${props.litellmServiceResult.base_url}\n${props.litellmServiceResult.config_path}`}</pre>
          ) : null}
        </form>
      </section>
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

function RunsTable({ runs }: { runs: RunRecord[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Provider</th>
          <th>Model</th>
          <th>Status</th>
          <th>Source</th>
          <th>Duration</th>
          <th>Usage</th>
          <th>Summary</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => (
          <tr key={run.id}>
            <td>{formatTime(run.started_at)}</td>
            <td>{run.provider || "ollama"}</td>
            <td>{run.model}</td>
            <td>
              <StatusBadge status={run.status} />
            </td>
            <td>{run.source_app || "-"}</td>
            <td>{run.duration_ms}ms</td>
            <td>{formatUsage(run.usage)}</td>
            <td>{run.error || run.prompt_summary}</td>
          </tr>
        ))}
        {!runs.length ? (
          <tr>
            <td colSpan={8} className="empty-cell">
              No runs recorded.
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}

function AuditTable({ entries, compact = false }: { entries: AuditEntry[]; compact?: boolean }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          {!compact ? <th>Task</th> : null}
          <th>Agent</th>
          <th>Action</th>
          {!compact ? <th>Env</th> : null}
          <th>Result</th>
          {!compact ? <th>Risk</th> : null}
          {!compact ? <th>Approval</th> : null}
        </tr>
      </thead>
      <tbody>
        {entries.map((entry) => (
          <tr key={entry.id}>
            <td>
              <code>{entry.timestamp}</code>
            </td>
            {!compact ? <td><code>{entry.taskId}</code></td> : null}
            <td>{entry.agent}</td>
            <td>{entry.action}</td>
            {!compact ? <td>{entry.environment}</td> : null}
            <td>
              <StatusBadge status={entry.result} />
            </td>
            {!compact ? <td><StatusBadge status={entry.risk} /></td> : null}
            {!compact ? <td><StatusBadge status={entry.approval} /></td> : null}
          </tr>
        ))}
        {!entries.length ? (
          <tr>
            <td colSpan={compact ? 4 : 8} className="empty-cell">
              No log entries match.
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}

function StatusBadge({ status }: { status: string }) {
  return <span className={`status-badge ${statusClass(status)}`}>{labelize(status)}</span>;
}

function FieldHelp({ text }: { text: string }) {
  return (
    <span className="help-dot" title={text} aria-label={text} tabIndex={0}>
      ?
    </span>
  );
}

function StatCard({ label, value, hint }: { label: string; value: number | string; hint?: string }) {
  return (
    <div className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint ? <small>{hint}</small> : null}
    </div>
  );
}

function ResourceBar({ label, value, detail }: { label: string; value: number; detail?: string }) {
  return (
    <div className="resource-row">
      <div>
        <span>{label}</span>
        <code>{detail || `${value}%`}</code>
      </div>
      <div className="meter">
        <span style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
      </div>
    </div>
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

function InfoPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="section-header">
        <h2>{title}</h2>
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

function labelize(value: string): string {
  return value.replace(/[_-]/g, " ");
}

function statusClass(status: string): string {
  if (["running", "success", "completed", "online", "active", "approved", "low"].includes(status)) {
    return "good";
  }
  if (["failed", "failure", "offline", "rejected", "high", "destroyed"].includes(status)) {
    return "bad";
  }
  if (["waiting_approval", "pending", "medium", "planning"].includes(status)) {
    return "warn";
  }
  return "neutral";
}

function timelineClass(current: TaskStatus, step: TaskStatus): string {
  const order: TaskStatus[] = ["queued", "planning", "running", "waiting_approval", "completed"];
  const currentIndex = current === "failed" ? 2 : order.indexOf(current);
  const stepIndex = order.indexOf(step);
  if (current === "failed" && step === "running") {
    return "bad";
  }
  if (stepIndex < currentIndex) {
    return "done";
  }
  if (stepIndex === currentIndex) {
    return "active";
  }
  return "";
}

function environmentDescription(environment: Environment): string {
  switch (environment) {
    case "planner":
      return "No execution. Produces a plan and reasoning artifacts.";
    case "browser":
      return "Isolated browser sandbox with web interaction.";
    case "computer-use":
      return "Desktop-style sandbox for screen and keyboard workflows.";
    case "local-desktop":
      return "Trusted local desktop. Highest risk.";
    default:
      return "";
  }
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

function formatUsage(usage?: { input_tokens?: number; output_tokens?: number; total_tokens?: number } | null): string {
  if (!usage) {
    return "-";
  }
  if (usage.total_tokens) {
    return usage.total_tokens.toString();
  }
  const parts = [usage.input_tokens ? `in ${usage.input_tokens}` : "", usage.output_tokens ? `out ${usage.output_tokens}` : ""].filter(Boolean);
  return parts.join(" / ") || "-";
}

function timestampNow(): string {
  return new Date().toISOString().slice(0, 16).replace("T", " ");
}
