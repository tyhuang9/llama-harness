import { invoke } from "@tauri-apps/api/core";
import type {
  CommandPreview,
  CommandResult,
  ConsoleApi,
  AgentDefinition,
  ConsoleEvent,
  ConsoleModels,
  ConsolePreferences,
  ConsoleRun,
  EvalLaunchRequest,
  EvaluationArtifacts,
  PromptfooArtifact,
  ProjectWorkspace,
  ReplayLaunchRequest,
} from "./types";

export const tauriConsoleApi: ConsoleApi = {
  getPreferences: () => invoke<ConsolePreferences>("get_preferences"),
  connectWorkspace: (workspace: ProjectWorkspace) =>
    invoke<ConsolePreferences>("connect_workspace", { workspace }),
  savePreferences: (update: ConsolePreferences) =>
    invoke<ConsolePreferences>("save_preferences", { update }),
  listRuns: (query) => invoke<ConsoleRun[]>("list_runs", { query }),
  listRunEvents: (executionId: string) =>
    invoke<ConsoleEvent[]>("list_run_events", { executionId }),
  listModels: () => invoke<ConsoleModels>("list_models"),
  listAgents: () => invoke<AgentDefinition[]>("list_agents"),
  listEvaluationArtifacts: () =>
    invoke<EvaluationArtifacts>("list_evaluation_artifacts"),
  listPromptfooArtifacts: () => invoke<PromptfooArtifact[]>("list_promptfoo_artifacts"),
  previewEvalCommand: (request: EvalLaunchRequest) =>
    invoke<CommandPreview>("preview_eval_command", { request }),
  launchEvalCommand: (request: EvalLaunchRequest) =>
    invoke<CommandResult>("launch_eval_command", { request }),
  previewReplayCommand: (request: ReplayLaunchRequest) =>
    invoke<CommandPreview>("preview_replay_command", { request }),
  launchReplayCommand: (request: ReplayLaunchRequest) =>
    invoke<CommandResult>("launch_replay_command", { request }),
};
