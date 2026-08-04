import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import type { ConsoleApi, ConsolePreferences } from "./types";

const workspace = {
  projectRoot: "C:/work/llama-harness",
  traceDbPath: "C:/work/llama-harness/.llama-harness/traces.sqlite",
  evaluationResultsPath: "C:/work/llama-harness/results",
  ollamaUrl: "http://127.0.0.1:11434",
};

function apiWith(preferences: ConsolePreferences): ConsoleApi {
  return {
    getPreferences: vi.fn().mockResolvedValue(preferences),
    connectWorkspace: vi.fn().mockResolvedValue({ ...preferences, workspace }),
    savePreferences: vi.fn().mockResolvedValue(preferences),
    listRuns: vi.fn().mockResolvedValue([]),
    listRunEvents: vi.fn().mockResolvedValue([]),
    listModels: vi.fn().mockResolvedValue({ health: { healthy: true }, models: [] }),
    listEvaluationArtifacts: vi.fn().mockResolvedValue({ reports: [], skippedFiles: [] }),
    previewEvalCommand: vi.fn(),
    launchEvalCommand: vi.fn(),
    previewReplayCommand: vi.fn(),
    launchReplayCommand: vi.fn(),
  };
}

describe("developer console", () => {
  afterEach(() => cleanup());

  it("connects a real workspace form and renders honest empty local states", async () => {
    const api = apiWith({ rawPayloadPreference: false, redactionKeyFragments: [] });
    render(<App api={api} />);

    fireEvent.change(await screen.findByLabelText("Project root"), { target: { value: workspace.projectRoot } });
    fireEvent.change(screen.getByLabelText("SQLite trace database"), { target: { value: workspace.traceDbPath } });
    fireEvent.change(screen.getByLabelText("Evaluation results path (optional)"), { target: { value: workspace.evaluationResultsPath } });
    fireEvent.click(screen.getByRole("button", { name: "Connect local project" }));

    await screen.findByRole("heading", { name: "Local Ollama models" });
    expect(screen.getByText("No local models reported")).toBeInTheDocument();
    expect(screen.queryByText(/demo|sample data/i)).not.toBeInTheDocument();
    expect(api.connectWorkspace).toHaveBeenCalledWith(workspace);
  });

  it("keeps navigation focusable and shows a redacted event timeline", async () => {
    const api = apiWith({ workspace, rawPayloadPreference: false, redactionKeyFragments: [] });
    vi.mocked(api.listRuns).mockResolvedValue([
      { runId: "run-1", traceId: "trace-1", startedAtMs: 1, updatedAtMs: 2, eventCount: 1, status: "completed" },
    ]);
    vi.mocked(api.listRunEvents).mockResolvedValue([
      { sequence: 1, timestampMs: 2, event: { type: "tool_completed", tool_id: "task.list", ok: true } },
    ]);
    render(<App api={api} />);

    const runs = await screen.findByRole("button", { name: /^Runs/ });
    runs.focus();
    expect(runs).toHaveFocus();
    fireEvent.click(runs);
    const run = await screen.findByRole("button", { name: "run-1" });
    fireEvent.click(run);

    await screen.findByText("tool_completed");
    expect(screen.getByText("Redacted event timeline")).toBeInTheDocument();
    expect(screen.queryByText("rawPayload")).not.toBeInTheDocument();
    await waitFor(() => expect(api.listRunEvents).toHaveBeenCalledWith("run-1"));
  });
});
