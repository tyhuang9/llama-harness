export type ProjectWorkspace = {
  projectRoot: string;
  traceDbPath: string;
  evaluationResultsPath?: string;
  agentManifestPath?: string;
  ollamaUrl: string;
};

export type AgentDefinition = {
  id: string;
  name: string;
  version: string;
  systemInstructions: string;
  defaultModel: string;
  toolAllowlist: string[];
  limits: { maxModelCalls: number; maxToolCalls: number };
  outputSchema?: Record<string, unknown>;
  metadata: Record<string, unknown>;
};

export type ConsolePreferences = {
  workspace?: ProjectWorkspace;
  rawPayloadPreference: boolean;
  redactionKeyFragments: string[];
  retentionDays?: number;
};

export type ConsoleRun = {
  executionId: string;
  runId: string;
  traceId: string;
  startedAtMs: number;
  updatedAtMs: number;
  eventCount: number;
  status?: string;
};

export type ConsoleEvent = {
  sequence: number;
  timestampMs: number;
  event: Record<string, unknown>;
};

export type ModelInfo = {
  id: string;
  capabilities: {
    supportsTools: boolean;
    supportsStreaming: boolean;
    supportsStructuredOutput: boolean;
  };
};

export type ConsoleModels = {
  health: { healthy: boolean; detail?: string };
  models: ModelInfo[];
};

export type EvaluationReport = {
  formatVersion: number;
  id: string;
  suiteId: string;
  suiteVersion: number;
  results: Array<{
    caseId: string;
    model: string;
    repetition: number;
    passed: boolean;
    traceId?: string;
    failures: Array<{ rule: string; message: string }>;
  }>;
};

export type EvaluationArtifacts = {
  reports: Array<{ path: string; report: EvaluationReport }>;
  skippedFiles: string[];
};

export type PromptfooArtifact = {
  kind: "generated_config" | "raw_result";
  path: string;
  content: string;
  truncated: boolean;
};

export type CommandPreview = {
  program: string;
  args: string[];
  cwd: string;
};

export type CommandResult = {
  command: CommandPreview;
  success: boolean;
  exitCode?: number;
  stdout: string;
  stderr: string;
};

export type EvalLaunchRequest = {
  suitePath: string;
  models: string[];
  repeat?: number;
};

export type ReplayLaunchRequest = { regressionPath: string };

export type ConsoleApi = {
  getPreferences(): Promise<ConsolePreferences>;
  connectWorkspace(workspace: ProjectWorkspace): Promise<ConsolePreferences>;
  savePreferences(preferences: ConsolePreferences): Promise<ConsolePreferences>;
  listRuns(query: { traceId?: string; status?: string }): Promise<ConsoleRun[]>;
  listRunEvents(executionId: string): Promise<ConsoleEvent[]>;
  listModels(): Promise<ConsoleModels>;
  listAgents(): Promise<AgentDefinition[]>;
  listEvaluationArtifacts(): Promise<EvaluationArtifacts>;
  listPromptfooArtifacts(): Promise<PromptfooArtifact[]>;
  previewEvalCommand(request: EvalLaunchRequest): Promise<CommandPreview>;
  launchEvalCommand(request: EvalLaunchRequest): Promise<CommandResult>;
  previewReplayCommand(request: ReplayLaunchRequest): Promise<CommandPreview>;
  launchReplayCommand(request: ReplayLaunchRequest): Promise<CommandResult>;
};
