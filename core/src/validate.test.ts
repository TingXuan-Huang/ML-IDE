import { describe, expect, it } from 'vitest';
import { coerceAgentConfig, toDataMeta } from './index';

describe('toDataMeta', () => {
  it('passes a valid payload through, injecting the path when absent', () => {
    const m = toDataMeta({ kind: 'ndarray', shape: [2, 3], dtype: 'float32', rowSample: 2 }, '/x.npy');
    expect(m.kind).toBe('ndarray');
    expect(m.path).toBe('/x.npy');
    expect(m.shape).toEqual([2, 3]);
  });

  it('degrades a drifted payload (bad / missing / non-object kind) to a visible unknown', () => {
    expect(toDataMeta({ kind: 'bogus' }, '/x').kind).toBe('unknown');
    expect(toDataMeta(null, '/x')).toMatchObject({ kind: 'unknown', path: '/x' });
    expect(toDataMeta({}, '/x').note).toContain('unrecognized');
  });
});

describe('coerceAgentConfig', () => {
  it('keeps a valid kind and preset-defaults the rest', () => {
    const c = coerceAgentConfig({ kind: 'codex' });
    expect(c.kind).toBe('codex');
    expect(c.command).toBe('codex'); // from the codex preset
  });

  it('falls back to custom on an unknown kind and rejects a bad promptVia', () => {
    const c = coerceAgentConfig({ kind: 'evil', command: 'x', promptVia: 'pipe' });
    expect(c.kind).toBe('custom');
    expect(c.promptVia).toBe('stdin'); // custom preset default; 'pipe' rejected
    expect(c.command).toBe('x');
  });

  it('filters non-string args and drops a non-number timeout', () => {
    const c = coerceAgentConfig({ kind: 'custom', args: ['-p', 42, null], timeoutMs: 'soon' });
    expect(c.args).toEqual(['-p']);
    expect(c.timeoutMs).toBeUndefined();
  });
});
