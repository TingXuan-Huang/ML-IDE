import { describe, expect, it } from 'vitest';
import { classLineFor, placeDirectives } from './directives';

// A small model file (1-based line numbers in comments).
const SRC = [
  'import torch.nn as nn', // 1
  '', // 2
  'class Net(nn.Module):', // 3
  '    def __init__(s):', // 4
  '        super().__init__()', // 5
  '    def forward(s, x):', // 6
  '        return s.fc(x)', // 7
];

describe('classLineFor', () => {
  it('finds the nearest enclosing class above a method', () => {
    expect(classLineFor(SRC, 6)).toBe(3);
  });
  it('returns null for a top-level function (no class above)', () => {
    expect(classLineFor(['def f(x):', '    return x'], 1)).toBeNull();
  });
});

describe('placeDirectives', () => {
  it('puts `input` above the method and `model` above the class', () => {
    const out = placeDirectives(SRC, 6, '# fusion: model = Net()\n# fusion: input = torch.randn(2, 4)');
    const text = out.join('\n');
    // model directive sits directly above the class line
    expect(out[out.indexOf('class Net(nn.Module):') - 1]).toContain('# fusion: model = Net()');
    // input directive sits directly above the forward def, indented to match
    const fwd = out.indexOf('    def forward(s, x):');
    expect(out[fwd - 1]).toBe('    # fusion: input = torch.randn(2, 4)');
    expect(text).toContain('# fusion: model = Net()');
  });

  it('is idempotent — re-applying the same directive does not pile up', () => {
    const once = placeDirectives(SRC, 6, '# fusion: input = torch.randn(2, 4)');
    const fwdLine = findForward(once); // forward shifted down by the inserted directive
    const reapplied = placeDirectives(once, fwdLine, '# fusion: input = torch.randn(2, 4)');
    expect(reapplied).toEqual(once);
  });

  it('replaces a stale directive block instead of stacking a new one', () => {
    const first = placeDirectives(SRC, 6, '# fusion: input = torch.randn(2, 4)');
    const fwdLine = findForward(first);
    const second = placeDirectives(first, fwdLine, '# fusion: input = torch.zeros(8)');
    const text = second.join('\n');
    expect(text).toContain('torch.zeros(8)');
    expect(text).not.toContain('torch.randn(2, 4)'); // old one stripped, not stacked
    expect((text.match(/# fusion: input/g) ?? []).length).toBe(1);
  });
});

// 1-based line of the `forward` def in a buffer (it shifts as directives are inserted).
function findForward(lines: string[]): number {
  return lines.findIndex((l) => /def forward\(/.test(l)) + 1;
}
