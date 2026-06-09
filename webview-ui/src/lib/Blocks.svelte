<script lang="ts">
  import { structure } from '../store';
  import { post } from '../vscode';
  import type { BlockLine, FunctionBlock } from '@fusion/shared';

  $: s = $structure;

  function reveal(line: number): void {
    if (s) post({ type: 'revealSymbol', path: s.path, line });
  }
  const io = (fn: FunctionBlock) =>
    `in(${fn.params.map((p) => p.name).join(',')})` + (fn.returns ? ` → ${fn.returns}` : '');
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
    <div class="fn">
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <div class="fnhd" role="button" tabindex="0" on:click={() => reveal(fn.startLine)}>
        <span>{fn.name}</span><span class="io">{io(fn)}</span>
      </div>
      {#each fn.lines as l}
        <div class="ln" class:bad={l.problem} class:changed={isChanged(l)}>
          <span>{l.text}</span><span class="sh">{shapesOf(l)}</span>
        </div>
        {#if l.problem}
          <div class="prob">⚠ {l.problem.message}</div>
        {/if}
      {/each}
    </div>
  {/each}
{/if}
