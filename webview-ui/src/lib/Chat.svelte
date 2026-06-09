<script lang="ts">
  // Agent chat sidebar (desktop). Streams the configured CLI agent; also surfaces
  // trace-assist proposals (a `# fusion:` directive with an Insert button) and a gear
  // to open the settings page.
  import { afterUpdate, onMount } from 'svelte';
  import { agentBusy, agentConfig, applyDirective, chat, nextAgentId, settingsOpen } from '../store';
  import { post } from '../vscode';

  let text = '';
  let activeId: string | null = null;
  let logEl: HTMLDivElement | undefined;

  onMount(() => post({ type: 'getAgentConfig' }));
  // Autoscroll after each update (NOT a reactive + queueMicrotask — that starves the paint).
  afterUpdate(() => {
    if (logEl) logEl.scrollTop = logEl.scrollHeight;
  });

  function send(): void {
    const t = text.trim();
    if (!t || $agentBusy) return;
    const id = nextAgentId();
    chat.update((ms) => [...ms, { role: 'user', text: t }, { id, role: 'agent', text: '', streaming: true }]);
    agentBusy.set(true);
    activeId = id;
    post({ type: 'agentPrompt', id, text: t });
    text = '';
  }
  function stop(): void {
    if (activeId) post({ type: 'agentCancel', id: activeId });
  }
  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }
</script>

<div class="chat">
  <div class="chathd">
    <span>✦ Agent</span>
    <span class="agentkind">{$agentConfig?.kind ?? '…'}</span>
    <span class="spacer"></span>
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <span class="gear" role="button" tabindex="0" title="Agent settings" on:click={() => settingsOpen.set(true)}>⚙</span>
  </div>
  <div class="chatlog" bind:this={logEl}>
    {#if !$chat.length}
      <div class="empty">
        Ask about your code, or click <b>✦ ask</b> on a function to make it traceable. Runs
        <code>{$agentConfig?.command || 'claude'}</code>.
      </div>
    {/if}
    {#each $chat as m}
      <div class="msg {m.role}" class:err={m.error}>
        <div class="who">{m.role === 'user' ? 'you' : '✦'}</div>
        <div class="text">
          {m.text}{#if m.streaming}<span class="caret">▋</span>{/if}
          {#if m.directive}
            <pre class="dir">{m.directive.text}</pre>
            <button class="ins" on:click={() => m.directive && applyDirective(m.directive)}>Insert &amp; trace</button>
          {/if}
        </div>
      </div>
    {/each}
  </div>
  <div class="chatin">
    <textarea bind:value={text} on:keydown={onKey} placeholder="Ask the agent…  (Enter to send, Shift+Enter newline)" rows="2"></textarea>
    {#if $agentBusy}
      <button class="stop" on:click={stop}>Stop</button>
    {:else}
      <button on:click={send} disabled={!text.trim()}>Send</button>
    {/if}
  </div>
</div>
