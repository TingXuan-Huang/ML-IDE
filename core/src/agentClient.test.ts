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

  it('config round-trips through disk; default is the claude preset', () => {
    expect(defaultAgentConfig().kind).toBe('claude');
    const file = path.join(os.tmpdir(), `fusion-agent-${process.pid}.json`);
    saveAgentConfig(file, AGENT_PRESETS.codex);
    expect(loadAgentConfig(file).kind).toBe('codex');
    expect(loadAgentConfig('/no/such/file.json').kind).toBe('claude'); // falls back to default
  });
});
