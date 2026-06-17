export type TaskStatus = "queued" | "planning" | "running" | "waiting_approval" | "completed" | "failed";
export type Environment = "planner" | "browser" | "computer-use" | "local-desktop";
export type ModelProvider = string;

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
  providerId?: string;
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
  defaultProviderId: string;
  defaultProvider?: ModelProvider;
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

export type ApprovalSeed = {
  id: string;
  taskId: string;
  action: string;
  risk: "low" | "medium" | "high";
  requestedAt: string;
};

export type ActivityFeedItem = {
  id: string;
  time: string;
  text: string;
};

type LocalSeedData = {
  seedTasks: Task[];
  seedAgents: Agent[];
  seedSandboxes: Sandbox[];
  seedAuditLog: AuditEntry[];
  seedApprovals: ApprovalSeed[];
  seedActivityFeed: ActivityFeedItem[];
};

const localSeedModules = import.meta.glob<Partial<LocalSeedData>>("./sentinelData.local.ts", {
  eager: true,
});

const localSeedData = Object.values(localSeedModules)[0] || {};

export const seedTasks: Task[] = localSeedData.seedTasks || [];
export const seedAgents: Agent[] = localSeedData.seedAgents || [];
export const seedSandboxes: Sandbox[] = localSeedData.seedSandboxes || [];
export const seedAuditLog: AuditEntry[] = localSeedData.seedAuditLog || [];
export const seedApprovals: ApprovalSeed[] = localSeedData.seedApprovals || [];
export const seedActivityFeed: ActivityFeedItem[] = localSeedData.seedActivityFeed || [];
