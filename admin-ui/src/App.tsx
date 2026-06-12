import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  api,
  ChatResponse,
  GenerateLiteLlmConfigResponse,
  getApiBase,
  Health,
  LiteLlmTestResponse,
  ModelRoute,
  ModelsResponse,
  OllamaModel,
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

const nav: Array<{ id: Page; label: string; detail?: string }> = [
  { id: "dashboard", label: "Dashboard", detail: "Live operations" },
  { id: "agents", label: "Agents", detail: "Prompts and defaults" },
  { id: "tasks", label: "Tasks", detail: "Queue and details" },
  { id: "sandboxes", label: "Sandboxes", detail: "Execution contexts" },
  { id: "models", label: "Models", detail: "Providers and Ollama" },
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

const providers: ModelProvider[] = ["Gemini", "OpenAI", "Anthropic", "OpenRouter", "Ollama"];
const litellmFamilies = ["openai", "anthropic", "openrouter", "gemini", "custom"];
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
  const [litellmTestResult, setLiteLlmTestResult] = useState<LiteLlmTestResponse | null>(null);
  const [litellmConfigResult, setLiteLlmConfigResult] = useState<GenerateLiteLlmConfigResponse | null>(null);
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

  async function saveSettings(event?: FormEvent) {
    event?.preventDefault();
    if (!settings) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await persistSettings(settings);
      await refreshAll();
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
      const result = await api.testLiteLLMProvider(model, "Say hello from llama-harness.");
      setLiteLlmTestResult(result);
      setProviderStatuses(await api.providers());
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
          {page === "new-task" ? <NewTaskPage agents={agents} settings={settings} createTask={createTask} cancel={() => setPage("tasks")} /> : null}
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

          {page === "agents" ? <AgentsPage agents={agents} setAgents={setAgents} /> : null}
          {page === "sandboxes" ? <SandboxesPage sandboxes={sandboxes} setSandboxes={setSandboxes} /> : null}
          {page === "models" ? (
            <ModelsPage
              models={models?.models || []}
              defaultModel={models?.default_model || settings?.default_model || null}
              settings={settings}
              setSettings={setSettings}
              providerStatuses={providerStatuses}
              selectDefaultModel={selectDefaultModel}
              saveSettings={() => saveSettings()}
              generateLiteLlmConfig={generateLiteLlmConfig}
              litellmConfigResult={litellmConfigResult}
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
              litellmTestResult={litellmTestResult}
              litellmConfigResult={litellmConfigResult}
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
    <div className="stack">
      <section className="hero-panel">
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

      <section className="metric-strip">
        <StatCard label="Active agents" value={props.agents.filter((agent) => agent.status === "active").length} hint="configured locally" />
        <StatCard label="Running tasks" value={runningTasks} hint="agent queue" />
        <StatCard label="Running sandboxes" value={runningSandboxes} hint="execution contexts" />
        <StatCard label="Pending approvals" value={props.approvals.length} hint="human review" />
        <StatCard label="Completed" value={completedTasks} hint="local task state" />
        <StatCard label="Failed" value={failedTasks} hint="local task state" />
      </section>

      <section className="two-column wide-left">
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

      <section className="panel">
        <div className="section-header">
          <h2>Recent Tasks</h2>
        </div>
        <TaskTable tasks={props.tasks.slice(0, 6)} openTask={props.openTask} />
      </section>

      <section className="panel">
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
  createTask,
  cancel,
}: {
  agents: Agent[];
  settings: Settings | null;
  createTask: (input: {
    title: string;
    instructions: string;
    agentId: string;
    provider: ModelProvider;
    model: string;
    environment: Environment;
  }) => void;
  cancel: () => void;
}) {
  const [agentId, setAgentId] = useState(agents[0]?.id || "");
  const agent = agents.find((item) => item.id === agentId) || agents[0];
  const [title, setTitle] = useState("");
  const [instructions, setInstructions] = useState("");
  const [provider, setProvider] = useState<ModelProvider>(agent?.defaultProvider || "Ollama");
  const [model, setModel] = useState(agent?.defaultModel || settings?.default_model || "");
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
    setProvider(next.defaultProvider);
    setModel(next.defaultModel);
    setEnvironment(next.defaultEnvironment);
    setAutonomy(next.autonomy);
    setPermissions(next.permissions);
  }

  function updatePermission(key: keyof AgentPermissions, value: boolean) {
    setPermissions((current) => ({ ...current, [key]: value }));
  }

  return (
    <form
      className="stack"
      onSubmit={(event) => {
        event.preventDefault();
        createTask({ title, instructions, agentId, provider, model, environment });
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
            {providers.map((item) => (
              <button key={item} type="button" className={provider === item ? "chip active" : "chip"} onClick={() => setProvider(item)}>
                {item}
              </button>
            ))}
          </div>
          <label className="field-block">
            Model
            <input value={model} onChange={(event) => setModel(event.target.value)} placeholder={settings?.default_model || "default"} />
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

function AgentsPage({ agents, setAgents }: { agents: Agent[]; setAgents: (agents: Agent[] | ((agents: Agent[]) => Agent[])) => void }) {
  const [activeId, setActiveId] = useState(agents[0]?.id || "");
  const [query, setQuery] = useState("");
  const filtered = agents.filter((agent) => [agent.name, agent.role, agent.description].join(" ").toLowerCase().includes(query.toLowerCase()));
  const active = agents.find((agent) => agent.id === activeId) || agents[0];

  function updateAgent(patch: Partial<Agent>) {
    if (!active) {
      return;
    }
    setAgents((list) => list.map((agent) => (agent.id === active.id ? { ...agent, ...patch, updatedAt: timestampNow() } : agent)));
  }

  function createAgent() {
    const agent: Agent = {
      id: `ag_${Math.random().toString(36).slice(2, 7)}`,
      name: "New agent",
      role: "Draft role",
      description: "",
      systemPrompt: "",
      defaultProvider: "Ollama",
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
                <select value={active.defaultProvider} onChange={(event) => updateAgent({ defaultProvider: event.target.value as ModelProvider })}>
                  {providers.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </label>
              <label>
                Model
                <input value={active.defaultModel} onChange={(event) => updateAgent({ defaultModel: event.target.value })} />
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
  saveSettings: () => void;
  generateLiteLlmConfig: () => void;
  litellmConfigResult: GenerateLiteLlmConfigResponse | null;
  testModel: string;
  setTestModel: (value: string) => void;
  prompt: string;
  setPrompt: (value: string) => void;
  testResult: ChatResponse | null;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
}) {
  const routes = props.settings?.model_routes || [];
  const litellmStatus = props.providerStatuses.find((provider) => provider.id === "litellm");

  function setRoutes(modelRoutes: ModelRoute[]) {
    if (!props.settings) {
      return;
    }
    props.setSettings({ ...props.settings, model_routes: modelRoutes });
  }

  function addRoute() {
    setRoutes([...routes, createModelRoute("openai")]);
  }

  function updateRoute(id: string, patch: Partial<ModelRoute>) {
    setRoutes(routes.map((route) => (route.id === id ? { ...route, ...patch } : route)));
  }

  function removeRoute(id: string) {
    setRoutes(routes.filter((route) => route.id !== id));
  }

  return (
    <div className="stack">
      {props.settings ? (
        <section className="panel">
          <div className="section-header">
            <div>
              <h2>LiteLLM Model Routes</h2>
              <p>{routes.length} configured routes</p>
            </div>
            <div className="button-row">
              <StatusBadge status={litellmStatus?.healthy ? "online" : props.settings.litellm.enabled ? "offline" : "neutral"} />
              <button type="button" onClick={addRoute} disabled={props.busy}>
                Add route
              </button>
            </div>
          </div>

          <div className="example-grid">
            <code>openai/&lt;model&gt;</code>
            <code>anthropic/&lt;model&gt;</code>
            <code>openrouter/&lt;provider&gt;/&lt;model&gt;</code>
            <code>gemini/&lt;model&gt;</code>
          </div>

          <div className="route-grid">
            {routes.map((route) => (
              <div className="route-card" key={route.id}>
                <div className="section-header">
                  <div>
                    <code>{route.id}</code>
                    <h3>{route.display_name || "Untitled route"}</h3>
                    <p>{route.model_alias || "model alias required"}</p>
                  </div>
                  <div className="button-row">
                    <label className="toggle-row inline">
                      <input
                        type="checkbox"
                        checked={route.enabled}
                        onChange={(event) => updateRoute(route.id, { enabled: event.target.checked })}
                      />
                      Enabled
                    </label>
                    <button className="danger-outline" type="button" onClick={() => removeRoute(route.id)} disabled={props.busy}>
                      Delete
                    </button>
                  </div>
                </div>
                <div className="settings-form">
                  <div className="field-row three">
                    <label>
                      Provider family
                      <select
                        value={route.provider_family}
                        onChange={(event) => {
                          const family = event.target.value;
                          updateRoute(route.id, {
                            provider_family: family,
                            api_key_env_var: route.api_key_env_var || envVarForFamily(family),
                          });
                        }}
                      >
                        {litellmFamilies.map((family) => (
                          <option key={family} value={family}>
                            {family}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Display name
                      <input value={route.display_name} onChange={(event) => updateRoute(route.id, { display_name: event.target.value })} />
                    </label>
                    <label>
                      Model alias
                      <input
                        value={route.model_alias}
                        onChange={(event) => updateRoute(route.id, { model_alias: event.target.value })}
                        placeholder={`${route.provider_family}:model`}
                      />
                    </label>
                  </div>
                  <div className="field-row three">
                    <label>
                      LiteLLM model
                      <input
                        value={route.litellm_model}
                        onChange={(event) => updateRoute(route.id, { litellm_model: event.target.value })}
                        placeholder={litellmModelPlaceholder(route.provider_family)}
                      />
                    </label>
                    <label>
                      API key env var
                      <input value={route.api_key_env_var} onChange={(event) => updateRoute(route.id, { api_key_env_var: event.target.value })} />
                    </label>
                    <label>
                      API base
                      <input value={route.api_base || ""} onChange={(event) => updateRoute(route.id, { api_base: event.target.value || null })} />
                    </label>
                  </div>
                  <label>
                    Notes
                    <input value={route.notes || ""} onChange={(event) => updateRoute(route.id, { notes: event.target.value || null })} />
                  </label>
                </div>
              </div>
            ))}
            {!routes.length ? <p className="empty">No LiteLLM routes configured.</p> : null}
          </div>

          <div className="form-actions">
            <button type="button" onClick={props.generateLiteLlmConfig} disabled={props.busy}>
              Generate config.yaml
            </button>
            <button className="primary" type="button" onClick={props.saveSettings} disabled={props.busy}>
              Save model routes
            </button>
          </div>
          {props.litellmConfigResult ? (
            <pre className="result">{`${props.litellmConfigResult.routes_written} routes written\n${props.litellmConfigResult.path}`}</pre>
          ) : null}
        </section>
      ) : null}

      <section className="panel">
        <div className="section-header">
          <h2>Ollama Models</h2>
          <p>Backed by the existing llama-harness API.</p>
        </div>
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
      </section>

      {props.settings ? (
        <section className="panel">
          <div className="section-header">
            <h2>Default Model Per Environment</h2>
          </div>
          <div className="settings-form">
            {["Planner", "Browser sandbox", "Computer-use sandbox", "Local desktop"].map((environment, index) => (
              <label key={environment}>
                {environment}
                <input
                  value={index === 3 ? props.settings?.default_model || "" : defaultEnvironmentModel(environment)}
                  onChange={(event) => {
                    if (index === 3 && props.settings) {
                      props.setSettings({ ...props.settings, default_model: event.target.value || null });
                    }
                  }}
                />
              </label>
            ))}
          </div>
        </section>
      ) : null}

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

function createModelRoute(family: string): ModelRoute {
  const id = `route_${family}_${Date.now().toString(36)}`;
  return {
    id,
    enabled: true,
    display_name: `${labelize(family)} route`,
    provider: "litellm",
    provider_family: family,
    model_alias: `${family}:model`,
    litellm_model: litellmModelPlaceholder(family).replace("<model>", "model").replace("<provider>", "provider"),
    api_key_env_var: envVarForFamily(family),
    api_base: null,
    notes: null,
  };
}

function envVarForFamily(family: string): string {
  switch (family) {
    case "openai":
      return "OPENAI_API_KEY";
    case "anthropic":
      return "ANTHROPIC_API_KEY";
    case "openrouter":
      return "OPENROUTER_API_KEY";
    case "gemini":
      return "GEMINI_API_KEY";
    default:
      return "PROVIDER_API_KEY";
  }
}

function litellmModelPlaceholder(family: string): string {
  switch (family) {
    case "openai":
      return "openai/<model>";
    case "anthropic":
      return "anthropic/<model>";
    case "openrouter":
      return "openrouter/<provider>/<model>";
    case "gemini":
      return "gemini/<model>";
    default:
      return "provider/<model>";
  }
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
  litellmTestResult: LiteLlmTestResponse | null;
  litellmConfigResult: GenerateLiteLlmConfigResponse | null;
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
          <h2>Settings</h2>
        </div>
        <form className="settings-form" onSubmit={props.onSubmit}>
          <label>
            API base URL
            <input value={props.apiBaseInput} onChange={(event) => props.setApiBaseInput(event.target.value)} />
          </label>
          <div className="field-row three">
            <label>
              Default provider
              <select value={settings.default_provider} onChange={(event) => setSettings({ ...settings, default_provider: event.target.value })}>
                <option value="ollama">ollama</option>
                <option value="litellm">litellm</option>
              </select>
            </label>
            <label>
              Ollama endpoint
              <input
                value={settings.ollama_endpoint}
                onChange={(event) => setSettings({ ...settings, ollama_endpoint: event.target.value })}
              />
            </label>
            <label>
              Default Ollama model
              <input
                value={settings.default_model || ""}
                onChange={(event) => setSettings({ ...settings, default_model: event.target.value || null })}
              />
            </label>
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
          <label>
            API token
            <input value={settings.api_token || ""} onChange={(event) => setSettings({ ...settings, api_token: event.target.value || null })} />
          </label>
          <label>
            Theme
            <select value={settings.theme} onChange={(event) => setSettings({ ...settings, theme: event.target.value })}>
              <option value="dark">dark</option>
              <option value="light">light</option>
            </select>
          </label>
          <label className="toggle-row inline">
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

      <section className="panel">
        <div className="section-header">
          <div>
            <h2>LiteLLM Gateway</h2>
            <p>{litellmStatus?.base_url || settings.litellm.base_url}</p>
          </div>
          <StatusBadge status={litellmStatus?.healthy ? "online" : settings.litellm.enabled ? "offline" : "neutral"} />
        </div>
        <form className="settings-form" onSubmit={props.onSubmit}>
          <label className="toggle-row">
            <span>Enable LiteLLM</span>
            <input type="checkbox" checked={settings.litellm.enabled} onChange={(event) => updateLiteLlm({ enabled: event.target.checked })} />
          </label>
          <div className="field-row three">
            <label>
              Base URL
              <input value={settings.litellm.base_url} onChange={(event) => updateLiteLlm({ base_url: event.target.value })} />
            </label>
            <label>
              API key / master key
              <input
                type="password"
                value={apiKeyValue}
                onChange={(event) => updateLiteLlm({ api_key: event.target.value || (apiKeyConfigured ? REDACTED_SECRET : null) })}
                placeholder={apiKeyConfigured ? "configured" : "optional"}
              />
            </label>
            <label>
              Default LiteLLM model
              <input
                value={settings.litellm.default_model || ""}
                onChange={(event) => updateLiteLlm({ default_model: event.target.value || null })}
                placeholder="openai:gpt-4o"
              />
            </label>
          </div>
          <div className="field-row three">
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
            <label>
              Managed config path
              <input
                value={settings.litellm.managed_config_path || ""}
                onChange={(event) => updateLiteLlm({ managed_config_path: event.target.value || null })}
                placeholder="litellm.config.yaml"
              />
            </label>
            <label className="toggle-row">
              <span>Allow raw models</span>
              <input
                type="checkbox"
                checked={settings.litellm.allow_unconfigured_models}
                onChange={(event) => updateLiteLlm({ allow_unconfigured_models: event.target.checked })}
              />
            </label>
          </div>
          <div className="button-row">
            <button type="button" onClick={props.testLiteLlmConnection} disabled={props.busy || !settings.litellm.default_model}>
              Test connection
            </button>
            <button type="button" onClick={props.generateLiteLlmConfig} disabled={props.busy}>
              Generate config.yaml
            </button>
            <button className="primary" type="submit" disabled={props.busy}>
              Save LiteLLM
            </button>
          </div>
          {props.litellmTestResult ? <pre className="result">{props.litellmTestResult.content || "(empty response)"}</pre> : null}
          {props.litellmConfigResult ? (
            <pre className="result">{`${props.litellmConfigResult.routes_written} routes written\n${props.litellmConfigResult.path}`}</pre>
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

function defaultEnvironmentModel(environment: string): string {
  switch (environment) {
    case "Planner":
      return "claude-sonnet-4.5";
    case "Browser sandbox":
      return "gpt-5";
    case "Computer-use sandbox":
      return "claude-opus-4";
    default:
      return "llama3.3:70b";
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
