<script lang="ts">
  // Design B — faithful-port compare. Pick a reference implementation; Fusion diffs it, module by
  // module, against the OPEN file (A = this file, B = reference). Each matched `ClassName.forward`
  // is aligned by op-kind + batch-normalized shape (LCS, so one inserted op is a single gap, not a
  // cascade); a row flags when a shape/op differs or an op is missing on one side. Every step shows
  // its SOURCE line, so activation/arg differences that don't change shape are still visible.
  import {
    abstract,
    compareFiles,
    compareResult,
    doc,
    fmtOp,
    fmtShape,
    lastReference,
    pickCompareFile,
    revealInEditor,
  } from '../store';
  import type { CompareModule, CompareStep } from '@fusion/shared';

  $: r = $compareResult;
  const base = (p: string): string => p.split('/').pop() ?? p;
  // Modules folded into `matched` show their crash inline, so suppress only THOSE problems from the
  // footer — a crash for a module that's only-in-A/B (not folded) must still surface.
  $: matchedNames = new Set((r?.matched ?? []).map((m) => m.module));

  // Diverging modules expand by default; converged ones collapse to a green header.
  let expanded = new Set<string>();
  $: if (r && r.matched) expanded = new Set(r.matched.filter((m) => m.diverges).map((m) => m.module));
  function toggle(name: string): void {
    if (expanded.has(name)) expanded.delete(name);
    else expanded.add(name);
    expanded = expanded;
  }

  // A cell shows the SOURCE line (primary) + a muted shape annotation (secondary).
  const stepCode = (st?: CompareStep | null): string =>
    !st ? '—' : st.crashed ? `✕ ${st.message ?? 'not traceable'}` : st.text || st.lhs || '';
  const stepShape = (st: CompareStep | null | undefined, dims: Record<string, string> | undefined, abs: boolean): string =>
    !st || st.crashed
      ? ''
      : st.op
        ? fmtOp(st.op, dims, abs)
        : st.shapes.map((sh) => `${sh.varName}[${fmtShape(sh.shape, dims, abs)}]`).join('  ');

  const status = (m: CompareModule): { cls: string; text: string } =>
    m.matchNote && m.matchNote.startsWith('not traceable')
      ? { cls: 'crash', text: `✕ ${m.matchNote}` }
      : m.diverges
        ? { cls: 'bad', text: `✕ diverges at step ${(m.divergeStep ?? 0) + 1}` }
        : { cls: 'ok', text: '✓ aligns' };

  // Re-compare ALWAYS uses the current open file as A (never a closed one) + the last reference.
  const recompare = (): void => {
    if ($doc && $lastReference) compareFiles($doc.path, $lastReference);
  };
</script>

<div class="cmpbar">
  {#if r && !r.error && !r.pending}<span class="sub">A: <b>{base(r.pathA)}</b> (this file) &nbsp;vs&nbsp; B: <b>{base(r.pathB)}</b> (reference)</span>{/if}
  <span class="spacer"></span>
  <span class="sub">abstract: {$abstract ? 'on' : 'off'} (display)</span>
  {#if $doc && $lastReference}<button class="ghost" on:click={recompare}>↻ re-compare</button>{/if}
  <button class="ghost" on:click={pickCompareFile}>{r ? 'pick another…' : 'pick reference…'}</button>
</div>

{#if r && r.pending}
  <div class="empty">Comparing <b>{base(r.pathA)}</b> vs <b>{base(r.pathB)}</b>…</div>
{:else if r && r.error}
  <div class="nanbar">⚠ compare failed — {r.error}</div>
{:else if !r && $lastReference}
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
  <div class="empty">
    {#if $doc}The file changed. <span class="link" role="button" tabindex="0" on:click={recompare}>↻ re-compare against {base($lastReference)}</span>{:else}Open a Python file, then re-compare.{/if}
  </div>
{:else if !r}
  <div class="empty">Pick a <b>reference implementation</b> to diff against this file, module by module.</div>
{:else}
  {#if !r.matched.length && !r.onlyA.length && !r.onlyB.length}
    <div class="empty">No comparable model forwards found{#if r.problems && r.problems.some((p) => !p.module)} — see below{/if}.</div>
  {/if}
  <div class="paper">
    {#each r.matched as m (m.module)}
      {@const st = status(m)}
      <div class="paper-sec">
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
        <div class="paper-h" role="button" tabindex="0" on:click={() => toggle(m.module)}>
          <span class="cmp-caret">{expanded.has(m.module) ? '▾' : '▸'}</span>
          {m.module}
          <span class="cmp-chip {st.cls}">{st.text}</span>
        </div>
        {#if expanded.has(m.module)}
          {#each m.rows as row, i}
            <div class="cmp-row" class:diverge={row.diverge} class:first={i === m.divergeStep}>
              {#if i === m.divergeStep}<span class="cmp-first">first divergence ↓</span>{/if}
              <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
              <span
                class="cmp-a"
                class:click={row.a && !row.a.crashed}
                role="button"
                tabindex="0"
                title="Reveal in editor"
                on:click={() => row.a && !row.a.crashed && r && revealInEditor(r.pathA, row.a.line)}>
                {stepCode(row.a)}<span class="cmp-shape">{stepShape(row.a, r.dimsA, $abstract)}</span>
              </span>
              <span class="cmp-b">{stepCode(row.b)}<span class="cmp-shape">{stepShape(row.b, r.dimsB, $abstract)}</span></span>
            </div>
          {/each}
        {/if}
      </div>
    {/each}

    {#if r.onlyA.length}<div class="cmp-only">Only in this file (A): {r.onlyA.join(', ')}</div>{/if}
    {#if r.onlyB.length}<div class="cmp-only">Only in reference (B): {r.onlyB.join(', ')}</div>{/if}
  </div>
{/if}

<!-- Problems that AREN'T already shown inline in a matched module (load failures, and crashes for
     only-in-A/B modules) ALWAYS surface, even when nothing matched. -->
{#if r && r.problems}
  {#each r.problems.filter((p) => !p.module || !matchedNames.has(p.module)) as p}<div class="prob">⚠ {p.message}</div>{/each}
{/if}
{#if r && r.note && !r.pending && !r.error}<div class="cmp-note">{r.note}</div>{/if}
