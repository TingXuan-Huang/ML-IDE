<script lang="ts">
  // #8 — paper-reading mode: the traced forward pass rendered as a paper-method-section
  // architecture description (symbolic shapes via the shared abstract toggle), plus an
  // agent "explain like a paper" prose panel.
  import {
    abstract,
    cancelPaperExplain,
    explainPaper,
    fmtOp,
    fmtShape,
    isDesktop,
    paper,
    paperExplain,
    requestPaper,
    revealInEditor,
  } from '../store';
  import type { PaperStep } from '@fusion/shared';

  // Re-request lazily whenever the cache is empty (mount, or after a re-trace nulls it).
  let pending = false;
  $: p = $paper;
  $: if (!p && !pending) {
    pending = true;
    requestPaper();
  }
  $: if (p) pending = false;
  // A step reads as "lhs = <op note>" (relabelled), or "lhs = var[shape] …" when no op.
  const stepRhs = (st: PaperStep, dims: Record<string, string> | undefined, abs: boolean): string =>
    st.op
      ? fmtOp(st.op, dims, abs)
      : st.shapes.map((sh) => `${sh.varName}[${fmtShape(sh.shape, dims, abs)}]`).join('  ');
</script>

{#if !p}
  <div class="empty">Reading the model as a paper…</div>
{:else if !p.sections.length}
  <div class="empty">
    Nothing traced yet. <b>▶ Trace this file</b> first{#if p.problems.length}: {p.problems[0].message}{/if}.
  </div>
{:else}
  <div class="paperbar">
    <span class="sub">abstract shapes: {$abstract ? 'on' : 'off (toggle in ⚙ Tracing)'}</span>
    <span class="spacer"></span>
    {#if isDesktop}
      <button class="ghost" on:click={explainPaper} disabled={$paperExplain?.streaming}>✦ Explain like a paper</button>
    {/if}
  </div>

  <div class="paper">
    {#each p.sections as sec (sec.module)}
      <div class="paper-sec">
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
        <div class="paper-h" role="button" tabindex="0" title="Reveal in editor" on:click={() => revealInEditor(p.path, sec.startLine)}>
          {sec.module}<span class="paper-note">  {sec.forwardNote}</span>
        </div>
        {#each sec.steps as st (st.line)}
          <div class="paper-step" class:changed={st.changed}>
            <span class="lhs">{st.lhs}</span> = <span class="rhs">{stepRhs(st, p.dims, $abstract)}</span>
          </div>
        {/each}
      </div>
    {/each}
  </div>

  {#if $paperExplain}
    <div class="paper-prose" class:err={$paperExplain.error}>
      <div class="paper-prose-h">
        ✦ Paper explanation
        {#if $paperExplain.streaming}<span class="caret">▋</span>
          <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
          <span class="prose-stop" role="button" tabindex="0" on:click={cancelPaperExplain}>stop</span>{/if}
      </div>
      <div class="paper-prose-body">{$paperExplain.text}</div>
    </div>
  {/if}
{/if}
