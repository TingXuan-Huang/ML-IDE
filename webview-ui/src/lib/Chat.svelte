<script lang="ts">
  // Agent chat sidebar (desktop). Streams the user's configured CLI agent (Claude Code /
  // Codex / custom) via the host. v1 is stateless per prompt (the host attaches the open
  // file as context); multi-turn memory + trace-assist land in later steps.
  import { afterUpdate, onMount } from 'svelte';
  import { agentBusy, agentInfo, chat, nextAgentId } from '../store';
  import { post } from '../vscode';

  let text = '';
  let activeId: string | null = null;
  let logEl: HTMLDivElement | undefined;

  onMount(() => post({ type: 'getAgentConfig' }));

  // Autoscroll to the bottom after each update. NB: do NOT use a reactive that reads a
  // bind:this element + schedules a microtask ($: ... queueMicrotask) — in Svelte 4 that
  // can re-fire every flush and starve the microtask queue, freezing the first paint.
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
  <div class="chathd">✦ Agent <span class="agentkind">{$agentInfo ? $agentInfo.kind : '…'}</span></div>
  <div class="chatlog" bind:this={logEl}>
    {#if !$chat.length}
      <div class="empty">Ask the agent about your code — it runs your configured CLI ({$agentInfo?.command || 'claude'}).</div>
    {/if}
    {#each $chat as m}
      <div class="msg {m.role}" class:err={m.error}>
        <div class="who">{m.role === 'user' ? 'you' : '✦'}</div>
        <div class="text">{m.text}{#if m.streaming}<span class="caret">▋</span>{/if}</div>
      </div>
    {/each}
  </div>
  <div class="chatin">
    <!-- svelte-ignore a11y-autofocus -->
    <textarea bind:value={text} on:keydown={onKey} placeholder="Ask the agent…  (Enter to send, Shift+Enter for newline)" rows="2"></textarea>
    {#if $agentBusy}
      <button class="stop" on:click={stop}>Stop</button>
    {:else}
      <button on:click={send} disabled={!text.trim()}>Send</button>
    {/if}
  </div>
</div>
