<script lang="ts">
  import { askTraceFile, folder, isDesktop, structure, trace, zone } from '../store';
  import { post } from '../vscode';
  import Blocks from './Blocks.svelte';
  import Graph from './Graph.svelte';
  import Data from './Data.svelte';
  import ProjectGraph from './ProjectGraph.svelte';
  import ModelSummary from './ModelSummary.svelte';
  import PaperView from './PaperView.svelte';
  import Compare from './Compare.svelte';
  import type { Zone } from '@fusion/shared';

  // Summary (#2) + Paper (#8) are per-file; Project (#3) needs an opened folder; Compare (B)
  // needs the Electron file dialog, so it's desktop-only.
  $: tabs = ['blocks', 'graph', 'summary', 'paper', 'data', ...($folder ? ['project'] : []), ...(isDesktop ? ['compare'] : [])] as Zone[];
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
          : $trace.phase === 'done'
            ? problems
              ? ` · traced · ✕ ${problems} shape problem${problems > 1 ? 's' : ''}`
              : s && s.hasShapes
                ? ' · traced ✓'
                : ' · traced · 0 shapes — ✦ ask to add inputs'
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
  {#if isDesktop}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div
      class="tab trace"
      role="button"
      tabindex="0"
      title="Ask the agent to make every un-traceable function traceable"
      on:click={askTraceFile}>
      ✦ ask
    </div>
  {/if}
</div>

<div class="hint">{status}</div>

<div class="body">
  {#if $zone === 'blocks'}
    <Blocks />
  {:else if $zone === 'graph'}
    <Graph />
  {:else if $zone === 'summary'}
    <ModelSummary />
  {:else if $zone === 'paper'}
    <PaperView />
  {:else if $zone === 'project'}
    <ProjectGraph />
  {:else if $zone === 'compare'}
    <Compare />
  {:else}
    <Data />
  {/if}
</div>
