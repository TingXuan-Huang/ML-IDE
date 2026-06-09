// HelperClient — host-side glue to the warm Python sidecar (lens-helper).
//
// Spawns `python -m lens_helper` once, speaks JSON-RPC over stdio, matches responses
// to requests by id, and lazily (re)starts the process — so a crash mid-session is
// recovered on the next request instead of wedging the cockpit. NO `vscode` import,
// so it is unit-testable with plain Node (see helperClient.smoke.ts).
import { spawn, ChildProcess } from 'child_process';
import * as readline from 'readline';

export interface HelperOptions {
  python?: string; // interpreter (from the ms-python env API in the real extension)
  cwd?: string; // where the lens_helper package is importable
  onStderr?: (line: string) => void;
}

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
}

export class HelperClient {
  private proc?: ChildProcess;
  private rl?: readline.Interface;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private restarts = 0;

  constructor(private readonly opts: HelperOptions = {}) {}

  private start(): void {
    const python = this.opts.python ?? 'python3';
    this.proc = spawn(python, ['-m', 'lens_helper'], {
      cwd: this.opts.cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.proc.on('exit', (code) => this.onExit(code));
    this.proc.on('error', (e) => {
      // e.g. ENOENT when the python path is wrong — settle pending so requests don't hang
      this.opts.onStderr?.('spawn error: ' + e.message);
      this.onExit(null);
    });
    if (this.opts.onStderr && this.proc.stderr) {
      readline.createInterface({ input: this.proc.stderr }).on('line', this.opts.onStderr);
    }
    this.rl = readline.createInterface({ input: this.proc.stdout! });
    this.rl.on('line', (line) => this.onLine(line));
  }

  private onLine(line: string): void {
    let msg: { id?: number; result?: unknown; error?: string };
    try {
      msg = JSON.parse(line);
    } catch {
      return; // ignore non-JSON noise on stdout
    }
    if (msg.id == null) return;
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    if (msg.error) p.reject(new Error(msg.error));
    else p.resolve(msg.result);
  }

  private onExit(code: number | null): void {
    const err = new Error(`lens-helper exited (code ${code})`);
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
    this.proc = undefined;
    this.rl = undefined;
  }

  /** Send a request; lazily (re)starts the helper if it isn't running. */
  request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    if (!this.proc) {
      this.restarts++;
      this.start();
    }
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.proc!.stdin!.write(JSON.stringify({ id, method, params }) + '\n');
    });
  }

  ping(): Promise<{ pong: boolean }> {
    return this.request('ping');
  }
  version(): Promise<{ name: string; version: string; protocol: number }> {
    return this.request('version');
  }
  loadFile(path: string): Promise<Record<string, unknown>> {
    return this.request('load_file', { path });
  }

  /** Force-kill the process under the client (test hook). */
  killForTest(): void {
    this.proc?.kill();
  }

  dispose(): void {
    this.proc?.kill();
    this.proc = undefined;
    this.rl = undefined;
  }

  get restartCount(): number {
    return this.restarts;
  }
}
