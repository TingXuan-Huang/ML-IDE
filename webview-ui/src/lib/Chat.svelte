<script lang="ts">
  // Agent chat sidebar (desktop). Streams the configured CLI agent; also surfaces
  // trace-assist proposals (a `# fusion:` directive with an Insert button) and a gear
  // to open the settings page.
  import { afterUpdate, onMount } from 'svelte';
  import {
    agentBusy,
    agentConfig,
    applyDirective,
    cancelAllAgents,
    chat,
    chatHistory,
    conversations,
    deleteConversation,
    loadConversations,
    nextAgentId,
    newConversation,
    resumeConversation,
    settingsOpen,
  } from '../store';
  import { theme, toggleTheme } from '../theme';
  import { post } from '../vscode';

  let text = '';
  let logEl: HTMLDivElement | undefined;
  let showHistory = false;

  onMount(() => post({ type: 'getAgentConfig' }));
  // Autoscroll after each update (NOT a reactive + queueMicrotask — that starves the paint).
  afterUpdate(() => {
    if (logEl) logEl.scrollTop = logEl.scrollHeight;
  });

  function send(): void {
    const t = text.trim();
    if (!t || $agentBusy) return;
    const history = chatHistory(); // prior turns BEFORE this one -> the agent's context
    const id = nextAgentId();
    chat.update((ms) => [...ms, { role: 'user', text: t }, { id, role: 'agent', text: '', streaming: true }]);
    agentBusy.set(true);
    post({ type: 'agentPrompt', id, text: t, history });
    text = '';
  }
  const stop = (): void => cancelAllAgents();
  function toggleHistory(): void {
    showHistory = !showHistory;
    if (showHistory) loadConversations();
  }
  function resume(id: string): void {
    resumeConversation(id);
    showHistory = false;
  }
  function startNew(): void {
    newConversation();
    showHistory = false;
  }
  const rel = (ts: number): string => {
    const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
  };
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
    <span class="gear" role="button" tabindex="0" title="New chat" on:click={startNew}>＋</span>
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <span class="gear" class:on={showHistory} role="button" tabindex="0" title="Conversation history" on:click={toggleHistory}>🕘</span>
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <span class="gear" role="button" tabindex="0" title="Toggle light / dark" on:click={toggleTheme}>{$theme === 'dark' ? '☀' : '☾'}</span>
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <span class="gear" role="button" tabindex="0" title="Agent settings" on:click={() => settingsOpen.set(true)}>⚙</span>
  </div>
  {#if showHistory}
    <div class="history">
      {#if !$conversations.length}
        <div class="empty">No saved conversations yet.</div>
      {/if}
      {#each $conversations as c (c.id)}
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
        <div class="hrow" role="button" tabindex="0" on:click={() => resume(c.id)}>
          <span class="htitle">{c.title}</span>
          <span class="htime">{rel(c.updatedAt)} · {c.count}</span>
          <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
          <span class="hdel" role="button" tabindex="0" title="Delete" on:click|stopPropagation={() => deleteConversation(c.id)}>✕</span>
        </div>
      {/each}
    </div>
  {/if}
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
