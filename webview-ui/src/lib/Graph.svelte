<script lang="ts">
  import { graph, structure } from '../store';
  import { post } from '../vscode';

  $: g = $graph;
  $: nodes = g ? [...g.nodes].sort((a, b) => a.line - b.line) : [];
  $: pos = Object.fromEntries(nodes.map((n, i) => [n.id, { x: 46, y: i * 44 + 24 }]));
  $: height = nodes.length * 44 + 24;

  function reveal(line: number): void {
    const s = $structure;
    post({ type: 'revealSymbol', path: s ? s.path : '', line });
  }
  const width = (label: string) => Math.min(210, 24 + label.length * 8.5);
</script>

{#if !g || !g.nodes.length}
  <div class="empty">No functions to graph. Open a Python file.</div>
{:else}
  {#if g.sparse}
    <div class="gsparse">static intra-file calls only — dynamic edges may be missing</div>
  {/if}
  <svg width="100%" height={height} viewBox={`0 0 300 ${height}`} preserveAspectRatio="xMinYMin meet">
    {#each g.edges as e}
      {#if pos[e.from] && pos[e.to]}
        <path
          class="gedge"
          d={`M ${pos[e.from].x} ${pos[e.from].y} C 20 ${pos[e.from].y}, 20 ${pos[e.to].y}, ${pos[e.to].x} ${pos[e.to].y}`}
        />
      {/if}
    {/each}
    {#each nodes as n}
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <g class="gnode" class:focus={n.id === g.focus} role="button" tabindex="0" on:click={() => reveal(n.line)}>
        <rect x={pos[n.id].x} y={pos[n.id].y - 12} width={width(n.label)} height="22" rx="2" />
        <text x={pos[n.id].x + 8} y={pos[n.id].y + 4}>{n.label}()</text>
      </g>
    {/each}
  </svg>
{/if}
