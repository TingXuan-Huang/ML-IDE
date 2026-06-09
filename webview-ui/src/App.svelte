<script lang="ts">
  // Shell selector. Desktop (Electron) host -> a two-pane IDE: Monaco editor | cockpit.
  // VS Code host -> the cockpit alone (VS Code provides the editor). Monaco is loaded
  // via a dynamic import so it's code-split and never ships in the VS Code webview.
  import { isDesktop } from './store';
  import Cockpit from './lib/Cockpit.svelte';
  import type { ComponentType } from 'svelte';

  let Editor: ComponentType | null = null;
  if (isDesktop) void import('./lib/Editor.svelte').then((m) => (Editor = m.default));
</script>

{#if isDesktop}
  <div class="ide">
    <div class="pane editor">
      {#if Editor}
        <svelte:component this={Editor} />
      {:else}
        <div class="loading">Loading editor…</div>
      {/if}
    </div>
    <div class="pane cockpit"><Cockpit /></div>
  </div>
{:else}
  <Cockpit />
{/if}

<style>
  .ide {
    display: flex;
    height: 100vh;
    width: 100vw;
  }
  .pane {
    height: 100%;
    overflow: hidden;
  }
  .pane.editor {
    flex: 1.5 1 0;
    min-width: 0;
    border-right: 1px solid var(--vscode-panel-border);
  }
  .pane.cockpit {
    flex: 1 1 0;
    min-width: 340px;
    overflow-y: auto;
  }
  .loading {
    padding: 20px;
    color: var(--vscode-descriptionForeground);
  }
</style>
