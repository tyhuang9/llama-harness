export type TaskStatus = "queued" | "planning" | "running" | "waiting_approval" | "completed" | "failed";
export type Environment = "planner" | "browser" | "computer-use" | "local-desktop";
export type ModelProvider = "Gemini" | "OpenAI" | "Anthropic" | "OpenRouter" | "Ollama";

export type AgentPermissions = {
  browser: boolean;
  fileRead: boolean;
  fileWrite: boolean;
  terminal: boolean;
};

export type Task = {
  id: string;
  name: string;
  status: TaskStatus;
  environment: Environment;
  provider: ModelProvider;
  model: string;
  createdAt: string;
  duration: string;
  subgoal: string;
  reasoning: string;
  instructions: string;
  agentId?: string;
};

export type Agent = {
  id: string;
  name: string;
  role: string;
  description: string;
  systemPrompt: string;
  defaultProvider: ModelProvider;
  defaultModel: string;
  defaultEnvironment: Environment;
  autonomy: "observe" | "ask" | "low-risk" | "autonomous";
  permissions: AgentPermissions;
  status: "active" | "paused" | "draft";
  tasksRun: number;
  updatedAt: string;
};

export type Sandbox = {
  id: string;
  environment: Environment;
  status: "running" | "idle" | "stopped" | "destroyed";
  cpu: number;
  ram: number;
  createdAt: string;
  taskId: string | null;
  isolation: "shared-browser" | "container" | "microvm" | "local-desktop";
};

export type AuditEntry = {
  id: string;
  timestamp: string;
  taskId: string;
  agent: string;
  action: string;
  environment: Environment;
  result: "success" | "blocked" | "failure" | "pending";
  risk: "low" | "medium" | "high";
  approval: "n/a" | "approved" | "rejected" | "pending";
};

export const seedTasks: Task[] = [
  {
    id: "t_8a1f",
    name: "Reconcile Q4 vendor invoices",
    status: "running",
    environment: "browser",
    provider: "Anthropic",
    model: "claude-sonnet-4.5",
    createdAt: "2026-06-11 09:14",
    duration: "00:12:41",
    subgoal: "Cross-check invoice #4471 against PO 9823",
    reasoning: "Comparing line items between the SAP export and supplier PDF. Detected a discrepancy on the shipping line.",
    instructions: "Download Q4 invoices from the accounting portal and reconcile them against purchase orders.",
    agentId: "ag_finance",
  },
  {
    id: "t_2c9d",
    name: "Scrape competitor pricing",
    status: "waiting_approval",
    environment: "browser",
    provider: "OpenAI",
    model: "gpt-5",
    createdAt: "2026-06-11 08:42",
    duration: "00:08:11",
    subgoal: "Request approval to slow crawl after rate limit",
    reasoning: "Target site returned 429. Continuing requires a slower cadence and explicit operator approval.",
    instructions: "Collect pricing data across 12 SKUs from three competitor sites.",
    agentId: "ag_research",
  },
  {
    id: "t_5b3a",
    name: "Draft contract redlines",
    status: "planning",
    environment: "planner",
    provider: "Anthropic",
    model: "claude-opus-4",
    createdAt: "2026-06-11 08:30",
    duration: "00:02:55",
    subgoal: "Outline clauses requiring revision",
    reasoning: "Reviewing the MSA template against the company playbook. Identifying liability and IP clauses.",
    instructions: "Compare contract draft v3 against the standard playbook.",
  },
  {
    id: "t_9e4c",
    name: "Provision staging environment",
    status: "completed",
    environment: "computer-use",
    provider: "OpenAI",
    model: "gpt-5",
    createdAt: "2026-06-11 07:55",
    duration: "00:21:33",
    subgoal: "Done",
    reasoning: "Terraform plan applied. Health checks are green.",
    instructions: "Spin up staging cluster mirroring production.",
    agentId: "ag_devops",
  },
  {
    id: "t_1f7b",
    name: "Triage support inbox",
    status: "running",
    environment: "browser",
    provider: "Gemini",
    model: "gemini-2.5-pro",
    createdAt: "2026-06-11 07:20",
    duration: "00:48:02",
    subgoal: "Classify ticket #8821",
    reasoning: "Customer reports billing mismatch. Routing to finance queue.",
    instructions: "Classify and tag all unread support tickets.",
    agentId: "ag_support",
  },
  {
    id: "t_3d2e",
    name: "Refresh dashboard snapshots",
    status: "queued",
    environment: "planner",
    provider: "OpenRouter",
    model: "auto",
    createdAt: "2026-06-11 07:02",
    duration: "-",
    subgoal: "Waiting in queue",
    reasoning: "Awaiting scheduler slot.",
    instructions: "Refresh executive dashboard PNG exports.",
  },
  {
    id: "t_6h8j",
    name: "Migrate legacy CSV data",
    status: "failed",
    environment: "local-desktop",
    provider: "Ollama",
    model: "llama3.3:70b",
    createdAt: "2026-06-10 22:10",
    duration: "00:04:19",
    subgoal: "Failed: schema mismatch",
    reasoning: "Source column 'cust_id' was missing in 3 of 14 files.",
    instructions: "Migrate archived CSVs into the warehouse.",
  },
  {
    id: "t_4k1m",
    name: "Generate weekly KPI report",
    status: "completed",
    environment: "planner",
    provider: "Anthropic",
    model: "claude-sonnet-4.5",
    createdAt: "2026-06-10 18:00",
    duration: "00:14:08",
    subgoal: "Done",
    reasoning: "Report delivered to #exec-updates.",
    instructions: "Compile and post weekly KPIs.",
  },
];

export const seedAgents: Agent[] = [
  {
    id: "ag_support",
    name: "Support Triage",
    role: "Customer support",
    description: "Classifies, tags, and drafts replies to inbound support tickets.",
    systemPrompt:
      "You are a customer support specialist at Llama Harness. Read each ticket carefully, identify the customer's intent, assign a category, and draft a clear reply. Escalate anything involving security, legal, or churn risk to a human.",
    defaultProvider: "Anthropic",
    defaultModel: "claude-sonnet-4.5",
    defaultEnvironment: "browser",
    autonomy: "ask",
    permissions: { browser: true, fileRead: true, fileWrite: false, terminal: false },
    status: "active",
    tasksRun: 48,
    updatedAt: "2026-06-10 14:22",
  },
  {
    id: "ag_finance",
    name: "Finance Reconciler",
    role: "Finance operations",
    description: "Reconciles invoices and purchase orders, then flags discrepancies for review.",
    systemPrompt:
      "You are a finance operations agent. Compare invoices against purchase orders line by line. Flag any mismatch over $50 or 2%. Never modify ledger entries directly; produce a reconciliation report and request approval before submitting corrections.",
    defaultProvider: "Anthropic",
    defaultModel: "claude-opus-4",
    defaultEnvironment: "browser",
    autonomy: "low-risk",
    permissions: { browser: true, fileRead: true, fileWrite: true, terminal: false },
    status: "active",
    tasksRun: 17,
    updatedAt: "2026-06-09 09:10",
  },
  {
    id: "ag_research",
    name: "Market Researcher",
    role: "Competitive research",
    description: "Gathers competitor pricing, feature, and positioning data.",
    systemPrompt:
      "You are a market research analyst. Collect public information about competitors from websites, pricing pages, and changelogs. Cite every source. Do not scrape behind login walls and respect robots.txt.",
    defaultProvider: "OpenAI",
    defaultModel: "gpt-5",
    defaultEnvironment: "browser",
    autonomy: "ask",
    permissions: { browser: true, fileRead: false, fileWrite: true, terminal: false },
    status: "active",
    tasksRun: 22,
    updatedAt: "2026-06-08 16:48",
  },
  {
    id: "ag_devops",
    name: "DevOps Operator",
    role: "Infrastructure",
    description: "Provisions environments and runs maintenance scripts.",
    systemPrompt:
      "You are a DevOps operator. Execute infrastructure changes via Terraform and approved scripts only. Always run a plan first and request human approval before applying changes to production.",
    defaultProvider: "OpenAI",
    defaultModel: "gpt-5",
    defaultEnvironment: "computer-use",
    autonomy: "ask",
    permissions: { browser: false, fileRead: true, fileWrite: true, terminal: true },
    status: "paused",
    tasksRun: 6,
    updatedAt: "2026-06-05 11:02",
  },
];

export const seedSandboxes: Sandbox[] = [
  { id: "sbx_001", environment: "browser", status: "running", cpu: 34, ram: 1280, createdAt: "2026-06-11 09:14", taskId: "t_8a1f", isolation: "container" },
  { id: "sbx_002", environment: "browser", status: "running", cpu: 12, ram: 860, createdAt: "2026-06-11 08:42", taskId: "t_2c9d", isolation: "container" },
  { id: "sbx_003", environment: "computer-use", status: "running", cpu: 61, ram: 3200, createdAt: "2026-06-11 07:55", taskId: null, isolation: "microvm" },
  { id: "sbx_004", environment: "browser", status: "running", cpu: 22, ram: 940, createdAt: "2026-06-11 07:20", taskId: "t_1f7b", isolation: "shared-browser" },
  { id: "sbx_005", environment: "local-desktop", status: "stopped", cpu: 0, ram: 0, createdAt: "2026-06-10 22:10", taskId: "t_6h8j", isolation: "local-desktop" },
  { id: "sbx_006", environment: "planner", status: "idle", cpu: 2, ram: 220, createdAt: "2026-06-10 18:00", taskId: null, isolation: "container" },
];

export const seedAuditLog: AuditEntry[] = [
  { id: "a_001", timestamp: "2026-06-11 09:26:11", taskId: "t_8a1f", agent: "browser-agent", action: "Opened https://erp.internal/invoices", environment: "browser", result: "success", risk: "low", approval: "n/a" },
  { id: "a_002", timestamp: "2026-06-11 09:25:48", taskId: "t_2c9d", agent: "browser-agent", action: "Requested approval: slow crawl after rate limit", environment: "browser", result: "pending", risk: "medium", approval: "pending" },
  { id: "a_003", timestamp: "2026-06-11 09:24:02", taskId: "t_1f7b", agent: "browser-agent", action: "Tagged ticket #8821 as billing", environment: "browser", result: "success", risk: "low", approval: "n/a" },
  { id: "a_004", timestamp: "2026-06-11 09:20:14", taskId: "t_9e4c", agent: "computer-use", action: "Executed terraform apply", environment: "computer-use", result: "success", risk: "high", approval: "approved" },
  { id: "a_005", timestamp: "2026-06-11 09:18:55", taskId: "t_8a1f", agent: "browser-agent", action: "Downloaded invoice_4471.pdf", environment: "browser", result: "success", risk: "low", approval: "n/a" },
  { id: "a_006", timestamp: "2026-06-10 22:13:40", taskId: "t_6h8j", agent: "local-desktop", action: "Read /archive/csv/2019/*.csv", environment: "local-desktop", result: "failure", risk: "medium", approval: "n/a" },
  { id: "a_007", timestamp: "2026-06-10 21:55:02", taskId: "t_4k1m", agent: "planner", action: "Posted report to #exec-updates", environment: "planner", result: "success", risk: "low", approval: "approved" },
  { id: "a_008", timestamp: "2026-06-10 21:30:18", taskId: "t_9e4c", agent: "computer-use", action: "Attempted to delete production bucket", environment: "computer-use", result: "blocked", risk: "high", approval: "rejected" },
];

export const seedApprovals = [
  { id: "ap_1", taskId: "t_2c9d", action: "Continue competitor pricing crawl at a slower cadence", risk: "medium" as const, requestedAt: "2026-06-11 09:25" },
  { id: "ap_2", taskId: "t_8a1f", action: "Submit invoice correction to vendor portal", risk: "high" as const, requestedAt: "2026-06-11 09:22" },
];

export const seedActivityFeed = [
  { id: "f1", time: "2 min ago", text: "Task 'Reconcile Q4 vendor invoices' downloaded 3 invoices" },
  { id: "f2", time: "4 min ago", text: "Approval requested on 'Scrape competitor pricing'" },
  { id: "f3", time: "11 min ago", text: "Sandbox sbx_006 went idle" },
  { id: "f4", time: "26 min ago", text: "Task 'Provision staging environment' completed" },
  { id: "f5", time: "1 hr ago", text: "Task 'Migrate legacy CSV data' failed: schema mismatch" },
  { id: "f6", time: "3 hr ago", text: "Weekly KPI report posted to #exec-updates" },
];
