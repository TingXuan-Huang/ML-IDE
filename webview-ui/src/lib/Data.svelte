<script lang="ts">
  import { data } from '../store';
  import { post } from '../vscode';

  $: d = $data;

  const fmt = (x: number) =>
    Math.abs(x) >= 1000 || (Math.abs(x) < 0.01 && x !== 0)
      ? x.toExponential(2)
      : String(Math.round(x * 1000) / 1000);

  function hist(values: unknown[] | undefined) {
    const nums = (values ?? []).filter((v): v is number => typeof v === 'number' && isFinite(v));
    if (nums.length < 2) return null;
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    const span = max - min || 1;
    const bins = 24;
    const counts = new Array(bins).fill(0);
    for (const v of nums) {
      const i = Math.max(0, Math.min(bins - 1, Math.floor(((v - min) / span) * bins)));
      counts[i]++;
    }
    const mx = Math.max(...counts) || 1;
    return { bars: counts.map((c) => (c / mx) * 100), n: nums.length, min, max };
  }

  const pick = () => post({ type: 'pickDataFile' });
  const cellsOf = (r: unknown): string =>
    (Array.isArray(r) ? r : []).map((x) => (x === null ? '·' : String(x))).join(' | ');

  $: name = d ? (d.path.split('/').pop() ?? '') : '';
  $: numIdx = d?.columns ? d.columns.findIndex((c) => /int|float|double/i.test(c.dtype)) : -1;
  $: ndHist = d && d.kind === 'ndarray' ? hist(d.sample) : null;
  $: colVals =
    d && d.kind === 'table' && numIdx >= 0
      ? (d.sample ?? []).map((r) => (r as unknown[])[numIdx])
      : [];
  $: colHist = colVals.length ? hist(colVals) : null;
</script>

<div class="hint">Mode 2 · Data Viz</div>

{#if !d}
  <div class="empty"><b>Pick a data file</b> (csv / npy / parquet) —
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <span class="trace" role="button" tabindex="0" on:click={pick}>browse…</span></div>
{:else if d.kind === 'ndarray'}
  <div class="hint">{name} · shape {JSON.stringify(d.shape)} · {d.dtype}</div>
  {#if ndHist}
    <div class="hist">{#each ndHist.bars as h}<div class="hbar" style={`height:${h}%`}></div>{/each}</div>
    <div class="hint">{ndHist.n} values · min {fmt(ndHist.min)} · max {fmt(ndHist.max)}</div>
  {/if}
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
  <div class="hint"><span class="trace" role="button" tabindex="0" on:click={pick}>pick another…</span></div>
{:else if d.kind === 'table'}
  <div class="hint">{name} · {(d.columns ?? []).map((c) => `${c.name}:${c.dtype}`).join('  ')}</div>
  {#each (d.sample ?? []).slice(0, 10) as r}
    <div class="ln"><span>{cellsOf(r)}</span></div>
  {/each}
  {#if colHist}
    <div class="hint">histogram · {(d.header ?? [])[numIdx] ?? 'col'}</div>
    <div class="hist">{#each colHist.bars as h}<div class="hbar" style={`height:${h}%`}></div>{/each}</div>
  {/if}
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
  <div class="hint"><span class="trace" role="button" tabindex="0" on:click={pick}>pick another…</span></div>
{:else}
  <div class="empty">{d.note || 'unsupported'}</div>
{/if}
