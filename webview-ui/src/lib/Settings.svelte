<script lang="ts">
  // Settings modal with three tabs: Agent (CLI agent), Tracing (synth/density), Memory.
  import { onDestroy, onMount } from 'svelte';
  import {
    agentConfig,
    agentTest,
    memory,
    saveAgentConfig,
    saveMemory,
    saveTracingConfig,
    settingsOpen,
    testAgent,
    tracingConfig,
  } from '../store';
  import { post } from '../vscode';
  import type { AgentConfigWire, TracingConfigWire } from '@fusion/shared';

  let tab: 'agent' | 'tracing' | 'memory' = 'agent';

  // --- agent ---
  const PRESETS: Record<string, Pick<AgentConfigWire, 'command' | 'args' | 'promptVia'>> = {
    claude: { command: 'claude', args: ['-p', '{prompt}'], promptVia: 'arg' },
    codex: { command: 'codex', args: ['exec', '{prompt}'], promptVia: 'arg' },
    custom: { command: '', args: [], promptVia: 'stdin' },
    mock: { command: '', args: [], promptVia: 'stdin' },
  };
  const MODEL_HINT: Record<string, string> = {
    claude: 'e.g. opus, sonnet, haiku — blank = your default',
    codex: 'e.g. gpt-5-codex — blank = your default',
  };
  let cfg: AgentConfigWire = $agentConfig ? { ...$agentConfig } : { kind: 'claude', ...PRESETS.claude, trust: 'review' };
  let argsText = cfg.args.join(' ');
  // Inactivity timeout shown in SECONDS (friendlier); default 300, 0 = none.
  let timeoutSec = cfg.timeoutMs == null ? 300 : Math.round(cfg.timeoutMs / 1000);
  let advanced = cfg.kind === 'custom';
  $: isCustom = cfg.kind === 'custom';
  $: hasModel = cfg.kind === 'claude' || cfg.kind === 'codex';
  $: invalid = isCustom && !cfg.command.trim();
  function applyPreset(): void {
    const p = PRESETS[cfg.kind];
    if (p) {
      cfg = { ...cfg, command: p.command, args: [...p.args], promptVia: p.promptVia };
      argsText = cfg.args.join(' ');
    }
    advanced = cfg.kind === 'custom';
    agentTest.set(null);
  }
  const buildAgent = (): AgentConfigWire => ({
    ...cfg,
    args: argsText.trim() ? argsText.trim().split(/\s+/) : [],
    timeoutMs: Math.max(0, Math.round(Number(timeoutSec) || 0)) * 1000,
  });
  function saveAgent(): void {
    if (!invalid) saveAgentConfig(buildAgent());
  }

  // --- tracing ---
  const DEFAULT_TRACING: TracingConfigWire = { density: 'changed', abstract: false, autoTrace: false, retries: 3 };
  let tc: TracingConfigWire = $tracingConfig ? { ...$tracingConfig } : { ...DEFAULT_TRACING };
  function saveTracing(): void {
    saveTracingConfig({ ...tc, retries: Math.max(1, Number(tc.retries) || 1) });
  }

  // --- memory ---
  let mem = $memory;
  $: mem = $memory; // refresh the textarea if the host pushes an updated memory while open
  const saveMem = (): void => saveMemory(mem);

  const close = (): void => settingsOpen.set(false);
  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') close();
    else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ({ agent: saveAgent, tracing: saveTracing, memory: saveMem })[tab]();
  }
  onMount(() => {
    agentTest.set(null);
    post({ type: 'getTracingConfig' });
    post({ type: 'getMemory' });
    window.addEventListener('keydown', onKey);
  });
  onDestroy(() => window.removeEventListener('keydown', onKey));
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div class="modal-bg" on:click={close}>
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
  <div class="modal" role="dialog" on:click|stopPropagation>
    <div class="modal-tabs">
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <span class="mtab" class:on={tab === 'agent'} role="tab" tabindex="0" on:click={() => (tab = 'agent')}>Agent</span>
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <span class="mtab" class:on={tab === 'tracing'} role="tab" tabindex="0" on:click={() => (tab = 'tracing')}>Tracing</span>
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <span class="mtab" class:on={tab === 'memory'} role="tab" tabindex="0" on:click={() => (tab = 'memory')}>Memory</span>
    </div>

    {#if tab === 'agent'}
      <label>Agent
        <select bind:value={cfg.kind} on:change={applyPreset}>
          <option value="claude">Claude Code</option>
          <option value="codex">Codex</option>
          <option value="custom">Custom command</option>
          <option value="mock">Mock (test, no CLI)</option>
        </select>
      </label>
      {#if hasModel}
        <label>Model <span class="sub">(optional)</span><input bind:value={cfg.model} placeholder={MODEL_HINT[cfg.kind]} /></label>
      {/if}
      {#if cfg.kind !== 'mock'}
        <label>Reply timeout <span class="sub">(seconds)</span><input type="number" min="0" bind:value={timeoutSec} placeholder="300" /></label>
        <span class="help">Aborts only after this long with <b>no new output</b> (streaming resets it). <code>0</code> = no timeout — set this for slow local models like a 120B.</span>
      {/if}
      <label>Apply agent-suggested inputs
        <select bind:value={cfg.trust}>
          <option value="review">review — show &amp; insert on click</option>
          <option value="auto">auto — insert &amp; trace immediately</option>
        </select>
        <span class="help">
          {cfg.trust === 'auto'
            ? 'Inserts & runs the agent’s # fusion: directive as soon as it’s proposed.'
            : 'You review each proposed # fusion: directive (Insert button) before it touches your file.'}
        </span>
      </label>
      {#if !isCustom}
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
        <div class="adv-toggle" role="button" tabindex="0" on:click={() => (advanced = !advanced)}>{advanced ? '▾' : '▸'} Advanced</div>
      {/if}
      {#if advanced || isCustom}
        <label>Command <input bind:value={cfg.command} placeholder="claude" /></label>
        <label>Args <span class="sub">(space-separated; <code>{'{prompt}'}</code> = the prompt)</span><input bind:value={argsText} placeholder={'-p {prompt}'} /></label>
        <label>Prompt delivery
          <select bind:value={cfg.promptVia}>
            <option value="arg">as an argument</option>
            <option value="stdin">via stdin</option>
          </select>
          <span class="help">
            {cfg.promptVia === 'arg'
              ? 'Passed on the command line (claude -p "…"). Simplest; very large prompts can hit the OS arg limit.'
              : 'Piped to the process’s stdin. No size limit, but the CLI must read its prompt from stdin.'}
          </span>
        </label>
      {/if}
      <div class="row">
        <button on:click={() => testAgent(buildAgent())} disabled={invalid || $agentTest?.pending}>{$agentTest?.pending ? 'Testing…' : 'Test'}</button>
        <span class="spacer"></span>
        <button class="ghost" on:click={close}>Cancel</button>
        <button on:click={saveAgent} disabled={invalid}>Save</button>
      </div>
      {#if $agentTest && !$agentTest.pending}
        <p class="test {$agentTest.ok ? 'ok' : 'bad'}">{$agentTest.ok ? '✓ reachable' : '✗ failed'} — {$agentTest.message}</p>
      {/if}
    {:else if tab === 'tracing'}
      <label>Shape density
        <select bind:value={tc.density}>
          <option value="changed">changed only — show shapes when they change</option>
          <option value="all">every tensor — show every captured shape</option>
        </select>
      </label>
      <label class="check"><input type="checkbox" bind:checked={tc.abstract} /> <span>Abstract shapes — relabel dims as symbols (B, L, D)</span></label>
      <p class="help">Reads dimensions back as <code>B</code>(atch) / <code>L</code>(ength) / <code>D</code>(model) / <code>C,H,W</code> from the traced input, e.g. <code>qkv[B, L, 3, 4, 32]</code>. Off = concrete numbers.</p>
      <label class="check"><input type="checkbox" bind:checked={tc.autoTrace} /> <span>Auto-trace a file as soon as it’s opened</span></label>
      <label>✦ ask retries <span class="sub">— agent↔tracer rounds</span><input type="number" min="1" max="6" bind:value={tc.retries} /></label>
      <div class="row">
        <span class="spacer"></span>
        <button class="ghost" on:click={close}>Cancel</button>
        <button on:click={saveTracing}>Save</button>
      </div>
    {:else}
      <label>Agent memory <span class="sub">— durable notes injected into every agent prompt</span>
        <textarea class="memarea" bind:value={mem} placeholder={'- this project uses BodyConfig(d_model=128)\n- forward takes a dict {"x_dense": ...}\n- prefer seq length 16'}></textarea>
      </label>
      <p class="help">
        Markdown notes the agent sees on every chat &amp; ✦ ask. Fusion auto-remembers a directive once
        it traces clean, and the agent can add facts via <code>{'# fusion: remember <fact>'}</code>. Global for
        now — per-project memory arrives with open-a-folder.
      </p>
      <div class="row">
        <span class="spacer"></span>
        <button class="ghost" on:click={close}>Cancel</button>
        <button on:click={saveMem}>Save</button>
      </div>
    {/if}

    <p class="hint">Saved to <code>~/.fusion/</code>. Esc to close · ⌘↵ to save.</p>
  </div>
</div>
