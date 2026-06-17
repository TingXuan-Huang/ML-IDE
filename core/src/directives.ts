// Pure source-text helpers for placing `# fusion:` directives. Extracted from the desktop host so
// this logic — which mutates the user's file on disk (insertDirective) on every trace-assist Insert
// — is unit-testable in isolation. No host (electron/vscode) dependencies.

/** Nearest top-level `class …:` above a method line (its 1-based def line), or null. */
export function classLineFor(lines: string[], methodLine: number): number | null {
  for (let i = methodLine - 1; i >= 1; i--) {
    if (/^class\s+\w/.test(lines[i - 1] ?? '')) return i;
  }
  return null;
}

/** Place directives at the RIGHT spot so trace_module finds them: `# fusion: model =` goes above
 *  the enclosing CLASS (where _pick_directive looks), `# fusion: input =` above the METHOD. Returns
 *  a NEW lines array (bottom-up inserts so line numbers don't shift). Idempotent: re-applying first
 *  strips the prior contiguous `# fusion:` block above each target, so directives don't pile up. */
export function placeDirectives(lines: string[], methodLine: number, text: string): string[] {
  const model: string[] = [];
  const input: string[] = [];
  for (const d of text.split('\n').map((s) => s.trim()).filter(Boolean)) {
    if (/^#\s*fusion:\s*model\s*=/.test(d)) model.push(d);
    else if (/^#\s*fusion:\s*input\s*=/.test(d)) input.push(d);
  }
  const out = [...lines];
  const edits: Array<{ at: number; dirs: string[] }> = [];
  if (input.length) edits.push({ at: methodLine, dirs: [...new Set(input)] });
  if (model.length) edits.push({ at: classLineFor(out, methodLine) ?? methodLine, dirs: [...new Set(model)] });
  edits.sort((a, b) => b.at - a.at); // bottom-up so the class edit's line number stays valid
  for (const e of edits) {
    // Strip any existing contiguous `# fusion:` lines directly above the target FIRST, so
    // re-inserting is idempotent — this is what stops directives piling up on every click.
    let lo = Math.max(0, e.at - 1); // 0-based index of the target line; we insert before it
    while (lo > 0 && /^\s*#\s*fusion:/.test(out[lo - 1] ?? '')) lo--;
    out.splice(lo, Math.max(0, e.at - 1) - lo); // remove the stale directive block
    const indent = (out[lo] ?? '').match(/^\s*/)?.[0] ?? '';
    out.splice(lo, 0, ...e.dirs.map((d) => indent + d));
  }
  return out;
}
