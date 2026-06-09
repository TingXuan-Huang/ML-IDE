<script lang="ts">
  // Settings modal with two tabs: Agent (CLI agent) and Tracing (synth dims, density, etc.).
  import { onDestroy, onMount } from 'svelte';
  import {
    agentConfig,
    agentTest,
    saveAgentConfig,
    saveTracingConfig,
    settingsOpen,
    testAgent,
    tracingConfig,
  } from '../store';
  import { post } from '../vscode';
  import type { AgentConfigWire, TracingConfigWire } from '@fusion/shared';

  let tab: 'agent' | 'tracing' = 'agent';

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
  const buildAgent = (): AgentConfigWire => ({ ...cfg, args: argsText.trim() ? argsText.trim().split(/\s+/) : [] });
  function saveAgent(): void {
    if (!invalid) saveAgentConfig(buildAgent());
  }

  // --- tracing ---
  const DEFAULT_TRACING: TracingConfigWire = { batch: 2, seq: 16, density: 'changed', autoTrace: false, retries: 3 };
  let tc: TracingConfigWire = $tracingConfig ? { ...$tracingConfig } : { ...DEFAULT_TRACING };
  function saveTracing(): void {
    saveTracingConfig({
      ...tc,
      batch: Math.max(1, Number(tc.batch) || 1),
      seq: Math.max(1, Number(tc.seq) || 1),
      retries: Math.max(1, Number(tc.retries) || 1),
    });
  }

  const close = (): void => settingsOpen.set(false);
  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') close();
    else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) (tab === 'agent' ? saveAgent : saveTracing)();
  }
  onMount(() => {
    agentTest.set(null);
    post({ type: 'getTracingConfig' });
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
    {:else}
      <label>Batch size (B) <span class="sub">— the B in synthesized inputs</span><input type="number" min="1" bind:value={tc.batch} /></label>
      <label>Sequence length (S) <span class="sub">— for Embedding / RNN inputs</span><input type="number" min="1" bind:value={tc.seq} /></label>
      <label>Shape density
        <select bind:value={tc.density}>
          <option value="changed">changed only — show shapes when they change</option>
          <option value="all">every tensor — show every captured shape</option>
        </select>
      </label>
      <label class="check"><input type="checkbox" bind:checked={tc.autoTrace} /> <span>Auto-trace a file as soon as it’s opened</span></label>
      <label>✦ ask retries <span class="sub">— agent↔tracer rounds</span><input type="number" min="1" max="6" bind:value={tc.retries} /></label>
      <div class="row">
        <span class="spacer"></span>
        <button class="ghost" on:click={close}>Cancel</button>
        <button on:click={saveTracing}>Save</button>
      </div>
    {/if}

    <p class="hint">Saved to <code>~/.fusion/</code>. Esc to close · ⌘↵ to save.</p>
  </div>
</div>
