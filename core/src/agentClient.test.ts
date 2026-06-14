import { describe, expect, it } from 'vitest';
import * as os from 'os';
import * as path from 'path';
import { AgentClient, AGENT_PRESETS, defaultAgentConfig, loadAgentConfig, saveAgentConfig } from './agentClient';

describe('AgentClient', () => {
  it('mock streams in pieces and returns the full canned response', async () => {
    const chunks: string[] = [];
    const out = await new AgentClient(AGENT_PRESETS.mock).run('hello', { onChunk: (d) => chunks.push(d) });
    expect(out).toContain('# fusion: input');
    expect(chunks.join('')).toBe(out); // streamed deltas reconstruct the whole
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('mock honors a custom response', async () => {
    const out = await new AgentClient(AGENT_PRESETS.mock).run('x', { mockResponse: 'RESP' });
    expect(out).toBe('RESP');
  });

  it('spawns a real process, delivers the prompt via stdin, streams stdout', async () => {
    // `node` stands in for the agent CLI: echo back "echo:" + whatever arrives on stdin.
    const out = await new AgentClient({
      kind: 'custom',
      command: process.execPath,
      args: ['-e', 'let s="";process.stdin.on("data",d=>s+=d);process.stdin.on("end",()=>process.stdout.write("echo:"+s))'],
      promptVia: 'stdin',
    }).run('hi there');
    expect(out).toBe('echo:hi there');
  });

  it('substitutes {prompt} into argv (arg mode)', async () => {
    const out = await new AgentClient({
      kind: 'custom',
      command: process.execPath,
      args: ['-e', 'process.stdout.write("arg:"+process.argv[1])', '{prompt}'],
      promptVia: 'arg',
    }).run('PAYLOAD');
    expect(out).toBe('arg:PAYLOAD');
  });

  it('rejects when the agent command is missing (ENOENT)', async () => {
    const c = new AgentClient({ kind: 'custom', command: 'definitely-not-a-real-cmd-xyz', args: [], promptVia: 'stdin' });
    await expect(c.run('x')).rejects.toThrow();
  });

  it('inactivity timeout fires on a hung (silent) process', async () => {
    const c = new AgentClient({
      kind: 'custom',
      command: process.execPath,
      args: ['-e', 'setTimeout(() => {}, 3000)'], // sleeps, never writes
      promptVia: 'stdin',
      timeoutMs: 150,
    });
    await expect(c.run('x')).rejects.toThrow(/no output|timed out/);
  });

  it('streaming output keeps re-arming the idle timer (slow model must NOT time out)', async () => {
    // Emits '.' every 80ms x4 (~320ms total) — never a 200ms gap, so the 200ms inactivity
    // timeout must NOT fire even though total runtime exceeds it. This is the 120B-model fix.
    const c = new AgentClient({
      kind: 'custom',
      command: process.execPath,
      args: ['-e', 'let n=0;const t=setInterval(()=>{process.stdout.write(".");if(++n>=4){clearInterval(t);process.exit(0);}},80)'],
      promptVia: 'stdin',
      timeoutMs: 200,
    });
    expect(await c.run('x')).toBe('....');
  });

  it('timeoutMs 0 disables the timeout', async () => {
    const c = new AgentClient({
      kind: 'custom',
      command: process.execPath,
      args: ['-e', 'setTimeout(() => process.stdout.write("ok"), 250)'], // silent 250ms, then output
      promptVia: 'stdin',
      timeoutMs: 0,
    });
    expect(await c.run('x')).toBe('ok');
  });

  it('config round-trips through disk; default is the claude preset', () => {
    expect(defaultAgentConfig().kind).toBe('claude');
    const file = path.join(os.tmpdir(), `fusion-agent-${process.pid}.json`);
    saveAgentConfig(file, AGENT_PRESETS.codex);
    expect(loadAgentConfig(file).kind).toBe('codex');
    expect(loadAgentConfig('/no/such/file.json').kind).toBe('claude'); // falls back to default
  });
});
