<script lang="ts">
  import { abstract, askTrace, caretLine, density, fmtOp, fmtShape, isDesktop, revealInEditor, structure, traceWithRealInput } from '../store';
  import { post } from '../vscode';
  import type { BlockLine, FunctionBlock } from '@fusion/shared';

  $: s = $structure;
  // The function the editor caret currently sits in (desktop) -> highlight its block.
  // Keyed by startLine (unique) — a transformer file has many methods named `forward`.
  $: activeFn = s?.functions.find((f) => $caretLine >= f.startLine && $caretLine <= f.endLine)?.startLine ?? null;
  // Display name: qualify methods with their class so the N `forward`s are distinguishable.
  const qual = (fn: FunctionBlock): string => (fn.className ? `${fn.className}.${fn.name}` : fn.name);

  function reveal(line: number): void {
    // Desktop: scroll the embedded Monaco editor. VS Code: ask the host to open it.
    if (s) revealInEditor(s.path, line);
  }
  function traceFn(fn: FunctionBlock): void {
    // Call THIS function directly (no __main__, no debugger). The host auto-synthesizes
    // an input and reports the exact call it ran, so the shapes are reproducible.
    if (s) post({ type: 'traceFunction', path: s.path, name: fn.name, line: fn.startLine });
  }
  function ask(fn: FunctionBlock): void {
    // Ask the agent to author a `# fusion:` directive that makes this function traceable.
    // Pass the bare name (for the trace lookup) + a qualified label (so the chat is clear).
    if (s) askTrace(s.path, fn.name, fn.startLine, qual(fn));
  }
  const io = (fn: FunctionBlock) =>
    `in(${fn.params.map((p) => p.name).join(',')})` + (fn.returns ? ` → ${fn.returns}` : '');
  // Auto-trace only data-path methods. Dunders (__init__, __call__, ...) are not a
  // forward pass — synthesizing a tensor input for them is meaningless.
  const canTrace = (fn: FunctionBlock) => !(fn.name.startsWith('__') && fn.name.endsWith('__'));
  // Density: 'all' shows every tensor; 'changed' (default) only shapes that changed.
  $: shapesOf = (l: BlockLine): string =>
    l.problem
      ? '✕ shape error'
      : ($density === 'all' ? l.shapes : l.shapes.filter((x) => x.changed))
          .map((x) => `${x.varName}[${fmtShape(x.shape, s?.dimNames, $abstract)}]`)
          .join('  ');
  $: isChanged = (l: BlockLine): boolean =>
    !!l.problem || l.shapes.some((x) => x.changed) || ($density === 'all' && l.shapes.length > 0);

  // First traced shape of a parameter inside this function (for the […, D] general form).
  const paramShape = (fn: FunctionBlock, name: string): number[] | null => {
    for (const l of fn.lines) {
      const hit = l.shapes.find((x) => x.varName === name);
      if (hit && hit.shape.length) return hit.shape;
    }
    return null;
  };
  // Input-shape requirements line: pinned ranks always shown (that's the warning the
  // trace can't express); flexible params shown as x[…, D] once traced — the traced
  // rank was just ONE valid choice, leading dims are free.
  $: reqLine = (fn: FunctionBlock): string => {
    const parts: string[] = [];
    for (const r of fn.shapeReqs ?? []) {
      if (r.kind === 'exact' || r.kind === 'min') {
        parts.push(`${r.name}: rank ${r.kind === 'exact' ? '=' : '≥'} ${r.rank}${r.via ? ` — ${r.via}` : ''}`);
      } else {
        const shp = paramShape(fn, r.name);
        if (shp) parts.push(`${r.name}[…, ${fmtShape(shp.slice(-1), s?.dimNames, $abstract)}] flexible`);
      }
    }
    return parts.join('  ·  ');
  };
</script>

{#if !s || !s.functions.length}
  <div class="empty">Open a Python file. Then <b>▶ Trace this file</b> to fill in shapes.</div>
{:else}
  {#each s.functions as fn}
    <div class="fn" class:active={activeFn === fn.startLine}>
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <div class="fnhd">
        <span class="fnname" role="button" tabindex="0" on:click={() => reveal(fn.startLine)}>{qual(fn)}</span>
        <span class="io">{io(fn)}</span>
        {#if canTrace(fn)}
          <span
            class="ftrace"
            role="button"
            tabindex="0"
            title="Run this function directly (auto-synthesized input) and fill in shapes — no __main__, no debugger"
            on:click={() => traceFn(fn)}>▶ trace</span>
        {/if}
        {#if isDesktop && canTrace(fn)}
          <span
            class="fask"
            role="button"
            tabindex="0"
            title="Trace this function with a REAL data file from the project (writes # fusion: input = load(...))"
            on:click={() => s && traceWithRealInput(s.path, fn.name, fn.startLine)}>⊞ real input</span>
        {/if}
        {#if isDesktop}
          <span
            class="fask"
            role="button"
            tabindex="0"
            title="Ask the agent to write a # fusion: directive that makes this traceable"
            on:click={() => ask(fn)}>✦ ask</span>
        {/if}
      </div>
      {#if fn.traceInput}
        <div class="prov" title="The exact call that produced these shapes">▶ {fn.traceInput}</div>
      {/if}
      {#if reqLine(fn)}
        <div class="reqs" title="Input-shape requirements inferred from this function's own ops — rank pins are hard requirements; 'flexible' means leading dims are free here">⛶ {reqLine(fn)}</div>
      {/if}
      {#each fn.lines as l}
        <div class="ln" class:bad={l.problem} class:changed={isChanged(l)}>
          <span>{l.text}</span><span class="sh">{shapesOf(l)}</span>
        </div>
        {#if l.problem}
          <div class="prob">⚠ {l.problem.message}</div>
        {/if}
        {#if l.op}
          <div class="op">∗ {fmtOp(l.op, s?.dimNames, $abstract)}</div>
        {/if}
      {/each}
    </div>
  {/each}
{/if}
