import { describe, expect, test } from 'vitest';
import { formatTranscript } from './conversation';

describe('conversations', () => {
  test('formatTranscript labels roles and bounds length', () => {
    const turns = [
      { role: 'user', text: 'first' },
      { role: 'agent', text: 'reply' },
      { role: 'user', text: 'second' },
    ];
    expect(formatTranscript(turns)).toBe('User: first\nAssistant: reply\nUser: second');
    expect(formatTranscript(turns, 1)).toBe('User: second'); // last-N only
    expect(formatTranscript([{ role: 'user', text: '  ' }])).toBe(''); // empty turns dropped
  });
});
