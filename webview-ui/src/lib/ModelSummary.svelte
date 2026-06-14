<script lang="ts">
  // #2 — torchinfo-style model summary: per-module output shape + params + share bar.
  // Lazy (requests on mount); shapes relabel via the shared abstract toggle.
  import { abstract, fmtShape, modelSummary, requestModelSummary } from '../store';

  // Re-request lazily whenever the cache is empty (mount, or after a re-trace nulls it while
  // this tab is open). `pending` prevents a re-fire storm before the response lands.
  let pending = false;
  $: s = $modelSummary;
  $: if (!s && !pending) {
    pending = true;
    requestModelSummary();
  }
  $: if (s) pending = false;
  const fmtN = (n: number): string => n.toLocaleString();
</script>

{#if !s}
  <div class="empty">Building the model summary… <span class="sub">(open a model file)</span></div>
{:else if !s.rows.length}
  <div class="empty">{s.error ?? 'No auto-buildable model in this file.'}{#if s.error && /constructor args/.test(s.error)} — add a <code># fusion: model = …</code> directive.{/if}</div>
{:else}
  <div class="prov" title="The exact call used to build + run the model">▶ {s.target}</div>
  {#if s.error}
    <div class="gsparse">forward crashed: {s.error} — params shown, output shapes omitted</div>
  {/if}
  <table class="msum">
    <thead>
      <tr><th>Module</th><th>Type</th><th>Output</th><th class="num">Params</th><th class="share">% of total</th></tr>
    </thead>
    <tbody>
      {#each s.rows as r (r.name)}
        <tr>
          <td class="mname" style="padding-left:{6 + (r.depth - 1) * 12}px" title={r.name}>{r.name.split('.').pop()}</td>
          <td class="mcls">{r.cls}</td>
          <td class="mshape">{r.outShape && r.outShape.length ? `[${fmtShape(r.outShape, s.dims, $abstract)}]` : '—'}</td>
          <td class="num">{fmtN(r.params)}</td>
          <td class="share">
            <div class="sharecell"><span class="pct">{r.pctParams}%</span><div class="hbar" style="width:{Math.min(100, r.pctParams)}%"></div></div>
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
  <div class="msum-foot">
    {fmtN(s.totalParams)} params · {fmtN(s.trainableParams)} trainable · {(s.paramBytes / 1e6).toFixed(2)} MB (params)
  </div>
{/if}
