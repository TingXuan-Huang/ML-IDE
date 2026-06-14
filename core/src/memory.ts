// Agent memory — pure helpers for the durable memory store. The file I/O and prompt
// injection live in the host (fs is host-specific); these are the testable pure bits:
// parsing the agent's `# fusion: remember <fact>` lines, formatting the memory document,
// and building the prompt context block.

const REMEMBER_RE = /^#\s*fusion:\s*remember\s*[:=]?\s*(.+)$/i;

/** Pull `# fusion: remember <fact>` facts out of an agent reply (trimmed, de-duped). */
export function parseRemembers(text: string): string[] {
  const out: string[] = [];
  for (const line of text.split('\n')) {
    const m = REMEMBER_RE.exec(line.trim());
    const fact = m?.[1].trim();
    if (fact && !out.includes(fact)) out.push(fact);
  }
  return out;
}

/** The agent reply with its `# fusion: remember …` lines removed (they're captured to
 *  memory, not shown as chat prose). Collapses the blank lines they leave behind. */
export function stripRememberLines(text: string): string {
  return text
    .split('\n')
    .filter((l) => !REMEMBER_RE.test(l.trim()))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** Append a fact to the memory document as a `- bullet`, ignoring exact duplicates.
 *  Returns the new document (unchanged if the fact was empty or already present). */
export function appendMemoryFact(memory: string, fact: string): string {
  const f = fact.trim().replace(/^[-*]\s*/, '');
  if (!f) return memory;
  const present = memory.split('\n').some((l) => l.replace(/^[-*]\s*/, '').trim() === f);
  if (present) return memory;
  const base = memory.trim();
  return (base ? `${base}\n` : '') + `- ${f}\n`;
}

/** Format the memory document as a prompt-context block (empty string when no memory). */
export function memoryPromptBlock(memory: string): string {
  const m = memory.trim();
  return m ? `Project memory — durable facts you established earlier (honor these):\n${m}\n\n` : '';
}
