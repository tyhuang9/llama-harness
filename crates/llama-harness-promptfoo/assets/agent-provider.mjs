import { spawn } from 'node:child_process';
import { appendFile } from 'node:fs/promises';

const MAX_RESPONSE_BYTES = 256 * 1024;

function executeCargo(request, cwd) {
  return new Promise((resolve, reject) => {
    const cargo = process.platform === 'win32' ? 'cargo.exe' : 'cargo';
    const child = spawn(cargo, ['run', '--quiet', '-p', 'local-task-agent', '--', 'promptfoo-adapter'], {
      cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
      if (Buffer.byteLength(stdout, 'utf8') > MAX_RESPONSE_BYTES) child.kill();
    });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) return reject(new Error(`Harness adapter exited ${code}: ${stderr.trim()}`));
      try { resolve(JSON.parse(stdout)); } catch (error) { reject(new Error(`Harness adapter returned invalid JSON: ${error}`)); }
    });
    child.stdin.end(JSON.stringify(request));
  });
}

export default class HarnessAgentProvider {
  constructor(options) {
    this.config = options.config || {};
  }

  id = () => `llama-harness:${this.config.model || 'unknown'}`;

  async callApi(prompt, context) {
    const caseId = context?.vars?.case_id;
    const repetition = context?.vars?.repetition;
    if (!caseId || typeof caseId !== 'string') return { error: 'Harness Promptfoo case_id is required.' };
    const observation = {
      case_id: caseId,
      model: this.config.model,
      repetition: Number.isInteger(repetition) ? repetition : 1,
    };
    try {
      const response = await executeCargo({
        suite_path: this.config.suitePath,
        case_id: caseId,
        input: prompt,
        model: this.config.model,
        ollama_url: this.config.ollamaUrl,
        trace_db: this.config.traceDbPath,
      }, this.config.projectRoot);
      await appendFile(this.config.observationPath, `${JSON.stringify({ ...observation, response })}\n`, 'utf8');
      return { output: response.output, metadata: { harness: response } };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await appendFile(this.config.observationPath, `${JSON.stringify({ ...observation, error: message })}\n`, 'utf8');
      return { error: message };
    }
  }
}
