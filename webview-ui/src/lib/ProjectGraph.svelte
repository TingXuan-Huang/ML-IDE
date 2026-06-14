<script lang="ts">
  // Cross-file import graph as a laid-out node-link diagram. The Python helper computes a
  // layered (Sugiyama) layout (importers on top, dependencies below); here we just draw it:
  // arrowed edges, click-to-open, and highlighting of the open file's neighborhood.
  import { onMount } from 'svelte';
  import { doc, folder, openTreeFile, projectGraph, requestProjectGraph } from '../store';
  import type { CallGraphEdge } from '@fusion/shared';

  const NODE_W = 168;
  const NODE_H = 28;
  const PAD = 30;

  onMount(() => requestProjectGraph());

  $: g = $projectGraph;
  $: cur = $doc && $folder && $doc.path.startsWith(`${$folder.root}/`) ? $doc.path.slice($folder.root.length + 1) : '';
  $: nodePos = g ? Object.fromEntries(g.nodes.map((n) => [n.id, n])) : {};
  $: neighbors = (() => {
    const s = new Set<string>();
    if (g && cur) for (const e of g.edges) { if (e.from === cur) s.add(e.to); if (e.to === cur) s.add(e.from); }
    return s;
  })();
  const label = (p: string): string => {
    const b = p.split('/').pop() ?? p;
    return b.length > 22 ? `${b.slice(0, 21)}…` : b;
  };

  let host: HTMLDivElement;
  let zoom = 1;
  let fittedFor: unknown = null;
  const clamp = (z: number): number => Math.max(0.2, Math.min(2.5, +z.toFixed(2)));
  const setZoom = (z: number): void => void (zoom = clamp(z));
  function fit(): void {
    if (g && host) zoom = clamp(Math.min(1, (host.clientWidth - 24) / (g.width + PAD * 2)));
  }
  // Fit width once per distinct graph (a 60-file project is much wider than the pane).
  $: if (g && host && fittedFor !== g) {
    fittedFor = g;
    fit();
  }

  const edgeCls = (e: CallGraphEdge): string => (cur && (e.from === cur || e.to === cur) ? 'pgedge on' : 'pgedge');
  function edgePath(e: CallGraphEdge): string {
    const a = nodePos[e.from];
    const b = nodePos[e.to];
    if (!a || !b) return '';
    const x1 = a.x + NODE_W / 2;
    const y1 = a.y + NODE_H;
    const x2 = b.x + NODE_W / 2;
    const y2 = b.y;
    const my = (y1 + y2) / 2;
    return `M ${x1} ${y1} C ${x1} ${my} ${x2} ${my} ${x2} ${y2}`;
  }
</script>

{#if !g}
  <div class="empty">Building the project graph…</div>
{:else if !g.nodes.length}
  <div class="empty">No Python files here. <b>File ▸ Open Folder…</b></div>
{:else}
  <div class="pgbar">
    <span>{g.nodes.length} files · {g.edges.length} imports{g.sparse ? ' · none in-project' : ''}</span>
    <span class="spacer"></span>
    <button title="Zoom out" on:click={() => setZoom(zoom - 0.15)}>−</button>
    <span class="pgzoom">{Math.round(zoom * 100)}%</span>
    <button title="Zoom in" on:click={() => setZoom(zoom + 0.15)}>+</button>
    <button title="Fit width" on:click={fit}>Fit</button>
  </div>
  <div class="pgcanvas" bind:this={host}>
    <svg
      width={(g.width + PAD * 2) * zoom}
      height={(g.height + PAD * 2) * zoom}
      viewBox={`${-PAD} ${-PAD} ${g.width + PAD * 2} ${g.height + PAD * 2}`}>
      <defs>
        <marker id="pgarrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="pgarrowhead" />
        </marker>
      </defs>
      {#each g.edges as e}
        <path class={edgeCls(e)} d={edgePath(e)} marker-end="url(#pgarrow)" />
      {/each}
      {#each g.nodes as n (n.id)}
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
        <g
          class="pgn"
          class:focus={n.id === cur}
          class:dim={cur && n.id !== cur && !neighbors.has(n.id)}
          role="button"
          tabindex="0"
          on:click={() => openTreeFile(n.id)}>
          <title>{n.id}</title>
          <rect x={n.x} y={n.y} width={NODE_W} height={NODE_H} rx="4" />
          <text x={n.x + NODE_W / 2} y={n.y + NODE_H / 2 + 4}>{label(n.id)}</text>
        </g>
      {/each}
    </svg>
  </div>
{/if}
