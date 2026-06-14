import { describe, expect, test } from 'vitest';
import { appendMemoryFact, memoryPromptBlock, parseRemembers, stripRememberLines } from './memory';

describe('agent memory', () => {
  test('parseRemembers pulls fusion remember facts (with : or =), de-duped', () => {
    const reply = 'Here you go\n# fusion: remember the body is BodyConfig(d_model=128)\n' +
      '# fusion: remember: forward takes a dict {x_dense}\n# fusion: input = torch.randn(2, 8)\n' +
      '# fusion: remember the body is BodyConfig(d_model=128)';
    expect(parseRemembers(reply)).toEqual([
      'the body is BodyConfig(d_model=128)',
      'forward takes a dict {x_dense}',
    ]);
  });

  test('stripRememberLines removes the directives but keeps prose', () => {
    expect(stripRememberLines('the answer\n# fusion: remember x\n\nmore')).toBe('the answer\n\nmore');
    expect(parseRemembers('no facts here')).toEqual([]);
  });

  test('appendMemoryFact bullets + de-dupes (ignores leading marker)', () => {
    const a = appendMemoryFact('', 'fact one');
    expect(a).toBe('- fact one\n');
    expect(appendMemoryFact(a, 'fact one')).toBe(a); // exact dup
    expect(appendMemoryFact(a, '- fact one')).toBe(a); // dup modulo bullet
    expect(appendMemoryFact(a, 'fact two')).toBe('- fact one\n- fact two\n');
    expect(appendMemoryFact('  ', '   ')).toBe('  '); // empty fact -> unchanged
  });

  test('memoryPromptBlock empty -> empty, else a labeled block', () => {
    expect(memoryPromptBlock('   ')).toBe('');
    const b = memoryPromptBlock('- uses dict input');
    expect(b).toContain('Project memory');
    expect(b).toContain('uses dict input');
  });
});
