// AgentClient — host-side glue to a user-chosen CLI coding agent (Claude Code, Codex,
// or any custom command). Spawns the agent like HelperClient spawns Python and streams
// its stdout. Piggybacks on the agent CLI's OWN auth — no API keys live here. The prompt
// is delivered over stdin (default, avoids argv length limits) or as an argv slot via a
// "{prompt}" placeholder. NO vscode/electron imports — pure Node, host-agnostic.
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

export type AgentKind = 'claude' | 'codex' | 'custom' | 'mock';

export interface AgentConfig {
  kind: AgentKind;
  command: string; // executable, e.g. "claude"
  args: string[]; // argv; a "{prompt}" entry/substring is replaced when promptVia==='arg'
  promptVia: 'stdin' | 'arg'; // how the prompt reaches the agent
  trust?: 'review' | 'auto'; // how agent-suggested directives are applied (default 'review')
  model?: string; // optional model; for claude/codex we append `--model <model>`
}

export interface AgentRunOptions {
  onChunk?: (delta: string) => void; // streamed stdout deltas (for the chat transcript)
  signal?: AbortSignal; // cancel a run
  timeoutMs?: number; // default 120s
  mockResponse?: string; // kind==='mock' only (tests / no-CLI dev)
}

// Sensible presets. The config page lets the user pick one and tweak the command/args.
export const AGENT_PRESETS: Record<AgentKind, AgentConfig> = {
  // `claude -p "<prompt>"` / `codex exec "<prompt>"` — prompt as an argv slot (spawn passes
  // it literally, no shell, so newlines/quotes in the code context are safe). Big files can
  // exceed ARG_MAX; switch a preset to promptVia:'stdin' in the config if that bites.
  claude: { kind: 'claude', command: 'claude', args: ['-p', '{prompt}'], promptVia: 'arg', trust: 'review' },
  codex: { kind: 'codex', command: 'codex', args: ['exec', '{prompt}'], promptVia: 'arg', trust: 'review' },
  custom: { kind: 'custom', command: '', args: [], promptVia: 'stdin', trust: 'review' },
  mock: { kind: 'mock', command: '', args: [], promptVia: 'stdin', trust: 'review' },
};

export function defaultAgentConfig(): AgentConfig {
  return { ...AGENT_PRESETS.claude };
}

const DEFAULT_MOCK = '# fusion: input = torch.randn(2, 8)\n';

/** Load the agent config from a JSON file, falling back to the default (claude preset). */
export function loadAgentConfig(file: string): AgentConfig {
  try {
    return { ...defaultAgentConfig(), ...JSON.parse(fs.readFileSync(file, 'utf8')) };
  } catch {
    return defaultAgentConfig();
  }
}

/** Persist the agent config (creates the parent dir). */
export function saveAgentConfig(file: string, cfg: AgentConfig): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(cfg, null, 2));
}

export class AgentClient {
  constructor(private readonly cfg: AgentConfig) {}

  /** Run the agent on `prompt`, streaming stdout via onChunk; resolves with the full text. */
  run(prompt: string, opts: AgentRunOptions = {}): Promise<string> {
    return this.cfg.kind === 'mock' ? this.runMock(opts) : this.runSpawn(prompt, opts);
  }

  private runMock(opts: AgentRunOptions): Promise<string> {
    const text = opts.mockResponse ?? DEFAULT_MOCK;
    const mid = Math.ceil(text.length / 2);
    const parts = [text.slice(0, mid), text.slice(mid)].filter((p) => p.length > 0);
    return new Promise((resolve) => {
      let i = 0;
      const tick = (): void => {
        if (i < parts.length) {
          // NB: advance `i` separately — `opts.onChunk?.(parts[i++])` would skip the
          // increment whenever onChunk is undefined (optional-chaining short-circuits
          // its arguments), spinning forever.
          const part = parts[i];
          i += 1;
          opts.onChunk?.(part);
          setImmediate(tick);
        } else {
          resolve(text);
        }
      };
      setImmediate(tick);
    });
  }

  private runSpawn(prompt: string, opts: AgentRunOptions): Promise<string> {
    const argv = this.cfg.args.map((a) => a.replace('{prompt}', prompt));
    // Optional model: claude/codex take `--model <model>`. For 'custom', put it in args yourself.
    const model = this.cfg.model?.trim();
    if (model && (this.cfg.kind === 'claude' || this.cfg.kind === 'codex')) argv.push('--model', model);
    return new Promise<string>((resolve, reject) => {
      let proc: ChildProcess;
      try {
        proc = spawn(this.cfg.command, argv, { stdio: ['pipe', 'pipe', 'pipe'] });
      } catch (e) {
        reject(e instanceof Error ? e : new Error(String(e)));
        return;
      }
      let out = '';
      let err = '';
      let settled = false;
      const to = opts.timeoutMs ?? 120_000;
      const onAbort = (): void => {
        proc.kill();
        finish(() => reject(new Error('agent aborted')));
      };
      const cleanup = (): void => {
        clearTimeout(timer);
        opts.signal?.removeEventListener('abort', onAbort);
      };
      const finish = (fn: () => void): void => {
        if (settled) return;
        settled = true;
        cleanup();
        fn();
      };
      const timer = setTimeout(() => {
        proc.kill();
        finish(() => reject(new Error(`agent timed out after ${to}ms`)));
      }, to);

      if (opts.signal) {
        if (opts.signal.aborted) {
          proc.kill();
          finish(() => reject(new Error('agent aborted')));
          return;
        }
        opts.signal.addEventListener('abort', onAbort);
      }

      proc.stdout?.on('data', (d) => {
        const s = String(d);
        out += s;
        opts.onChunk?.(s);
      });
      proc.stderr?.on('data', (d) => (err += String(d)));
      proc.on('error', (e) => finish(() => reject(e))); // e.g. ENOENT: agent CLI not installed
      proc.on('close', (code) =>
        finish(() =>
          code === 0
            ? resolve(out)
            : reject(new Error(`agent exited ${code}: ${err.trim() || out.trim() || 'no output'}`)),
        ),
      );

      if (this.cfg.promptVia === 'stdin' && proc.stdin) {
        proc.stdin.write(prompt);
        proc.stdin.end();
      }
    });
  }
}
