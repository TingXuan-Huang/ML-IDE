<script lang="ts">
  // Shell. Desktop -> [explorer? | editor | cockpit | chat] with draggable splitters; VS Code -> cockpit alone.
  import { folder, isDesktop, pickFolder, settingsOpen } from './store';
  import Cockpit from './lib/Cockpit.svelte';
  import Chat from './lib/Chat.svelte';
  import FileTree from './lib/FileTree.svelte';
  import Settings from './lib/Settings.svelte';
  import type { ComponentType } from 'svelte';

  let Editor: ComponentType | null = null;
  if (isDesktop) void import('./lib/Editor.svelte').then((m) => (Editor = m.default));

  // --- resizable panes ---
  let ideEl: HTMLDivElement;
  let editorFlex = 1.5; // editor : cockpit grow ratio
  let cockpitFlex = 1;
  let chatW = 340; // chat fixed width (px)

  function beginDrag(move: (ev: MouseEvent) => void): void {
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    const up = (): void => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  }
  function dragEditor(e: MouseEvent): void {
    e.preventDefault();
    const startX = e.clientX;
    const total = editorFlex + cockpitFlex;
    const startEditor = editorFlex;
    const areaW = Math.max(1, (ideEl?.clientWidth ?? 1200) - chatW - 10);
    beginDrag((ev) => {
      const d = ((ev.clientX - startX) / areaW) * total;
      editorFlex = Math.max(0.25, Math.min(total - 0.25, startEditor + d));
      cockpitFlex = total - editorFlex;
    });
  }
  function dragChat(e: MouseEvent): void {
    e.preventDefault();
    const startX = e.clientX;
    const startW = chatW;
    beginDrag((ev) => {
      chatW = Math.max(220, Math.min(720, startW - (ev.clientX - startX)));
    });
  }
</script>

{#if isDesktop}
  <div class="ide" bind:this={ideEl}>
    {#if $folder}
      <div class="pane explorer">
        <div class="exhd" title={$folder.root}>
          <span class="exname">{$folder.root.split('/').pop()}</span>
          <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
          <span class="exopen" role="button" tabindex="0" title="Open another folder" on:click={pickFolder}>⊞</span>
        </div>
        <div class="extree"><FileTree /></div>
      </div>
    {/if}
    <div class="pane editor" style="flex: {editorFlex} 1 0">
      {#if Editor}
        <svelte:component this={Editor} />
      {:else}
        <div class="loading">Loading editor…</div>
      {/if}
    </div>
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="splitter" role="separator" on:mousedown={dragEditor}></div>
    <div class="pane cockpit" style="flex: {cockpitFlex} 1 0"><Cockpit /></div>
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="splitter" role="separator" on:mousedown={dragChat}></div>
    <div class="pane chat" style="flex: 0 0 {chatW}px"><Chat /></div>
  </div>
{:else}
  <Cockpit />
{/if}

{#if isDesktop && $settingsOpen}
  <Settings />
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
    min-width: 0;
  }
  .pane.cockpit {
    overflow-y: auto;
  }
  .pane.explorer {
    flex: 0 0 220px;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--vscode-panel-border);
    background: var(--vscode-tab-inactiveBackground);
  }
  .exhd {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    font-weight: 600;
    border-bottom: 1px solid var(--vscode-panel-border);
  }
  .exhd .exname {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .exhd .exopen {
    cursor: pointer;
    opacity: 0.7;
  }
  .exhd .exopen:hover {
    opacity: 1;
  }
  .extree {
    flex: 1;
    overflow: auto;
    padding: 4px 0;
  }
  .splitter {
    flex: 0 0 5px;
    cursor: col-resize;
    background: var(--vscode-panel-border);
  }
  .splitter:hover {
    background: var(--vscode-focusBorder);
  }
  .loading {
    padding: 20px;
    color: var(--vscode-descriptionForeground);
  }
</style>
