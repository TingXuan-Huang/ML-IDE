// Saved agent conversations — pure helpers. File I/O lives in the host; the chat UI lives
// in the webview. These are the shared, testable bits: deriving a title and formatting a
// prior-turns transcript for prompt replay (so a resumed conversation keeps its context).

export type Turn = { role: string; text: string };

/** Format prior turns as a `User:`/`Assistant:` transcript for the next prompt. Bounded to
 *  the last `max` non-empty turns so a long history can't blow up the agent's context. */
export function formatTranscript(turns: Turn[], max = 12): string {
  return turns
    .filter((t) => t.text.trim())
    .slice(-max)
    .map((t) => `${t.role === 'user' ? 'User' : 'Assistant'}: ${t.text.trim()}`)
    .join('\n');
}
