<script lang="ts">
  import { askTrace, caretLine, isDesktop, revealInEditor, structure } from '../store';
  import { post } from '../vscode';
  import type { BlockLine, FunctionBlock } from '@fusion/shared';

  $: s = $structure;
  // The function the editor caret currently sits in (desktop) -> highlight its block.
  $: activeFn = s?.functions.find((f) => $caretLine >= f.startLine && $caretLine <= f.endLine)?.name ?? null;

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
    if (s) askTrace(s.path, fn.name, fn.startLine);
  }
  const io = (fn: FunctionBlock) =>
    `in(${fn.params.map((p) => p.name).join(',')})` + (fn.returns ? ` → ${fn.returns}` : '');
  // Auto-trace only data-path methods. Dunders (__init__, __call__, ...) are not a
  // forward pass — synthesizing a tensor input for them is meaningless.
  const canTrace = (fn: FunctionBlock) => !(fn.name.startsWith('__') && fn.name.endsWith('__'));
  const shapesOf = (l: BlockLine) =>
    l.problem
      ? '✕ shape error'
      : l.shapes.filter((x) => x.changed).map((x) => `${x.varName}[${x.shape.join(', ')}]`).join('  ');
  const isChanged = (l: BlockLine) => !!l.problem || l.shapes.some((x) => x.changed);
</script>

{#if !s || !s.functions.length}
  <div class="empty">Open a Python file. Then <b>▶ Trace this file</b> to fill in shapes.</div>
{:else}
  {#each s.functions as fn}
    <div class="fn" class:active={activeFn === fn.name}>
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <div class="fnhd">
        <span class="fnname" role="button" tabindex="0" on:click={() => reveal(fn.startLine)}>{fn.name}</span>
        <span class="io">{io(fn)}</span>
        {#if canTrace(fn)}
          <span
            class="ftrace"
            role="button"
            tabindex="0"
            title="Run this function directly (auto-synthesized input) and fill in shapes — no __main__, no debugger"
            on:click={() => traceFn(fn)}>▶ trace</span>
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
      {#each fn.lines as l}
        <div class="ln" class:bad={l.problem} class:changed={isChanged(l)}>
          <span>{l.text}</span><span class="sh">{shapesOf(l)}</span>
        </div>
        {#if l.problem}
          <div class="prob">⚠ {l.problem.message}</div>
        {/if}
        {#if l.op}
          <div class="op">∗ {l.op}</div>
        {/if}
      {/each}
    </div>
  {/each}
{/if}
