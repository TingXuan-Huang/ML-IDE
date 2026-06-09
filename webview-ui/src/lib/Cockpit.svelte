<script lang="ts">
  import { structure, trace, zone } from '../store';
  import { post } from '../vscode';
  import Blocks from './Blocks.svelte';
  import Graph from './Graph.svelte';
  import Data from './Data.svelte';
  import type { Zone } from '@fusion/shared';

  const tabs: Zone[] = ['blocks', 'graph', 'data'];
  const label = (z: Zone) => z[0].toUpperCase() + z.slice(1);

  function setZone(z: Zone): void {
    zone.set(z);
    post({ type: 'setPrimaryZone', zone: z });
  }

  $: s = $structure;
  $: problems = s ? s.functions.reduce((n, f) => n + f.lines.filter((l) => l.problem).length, 0) : 0;
  $: status = (() => {
    const base = s ? `${s.path.split('/').pop()} · ${s.functions.length} functions` : 'no file';
    const t =
      $trace.phase === 'tracing'
        ? ' · tracing…'
        : $trace.phase === 'error'
          ? ` · error: ${'message' in $trace ? $trace.message : ''}`
          : s && s.hasShapes
            ? problems
              ? ` · traced · ✕ ${problems} shape problem${problems > 1 ? 's' : ''}`
              : ' · traced ✓'
            : '';
    return base + t;
  })();
</script>

<div class="tabs">
  {#each tabs as t}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div class="tab" class:on={$zone === t} role="tab" tabindex="0" on:click={() => setZone(t)}>{label(t)}</div>
  {/each}
  <div class="spacer"></div>
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
  <div class="tab trace" role="button" tabindex="0" on:click={() => post({ type: 'requestTrace', path: '' })}>
    ▶ Trace this file
  </div>
</div>

<div class="hint">{status}</div>

<div class="body">
  {#if $zone === 'blocks'}
    <Blocks />
  {:else if $zone === 'graph'}
    <Graph />
  {:else}
    <Data />
  {/if}
</div>
