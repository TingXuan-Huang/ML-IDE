<script lang="ts">
  // Agent settings page (desktop). Pick the CLI agent, model, and trust mode; advanced
  // fields (command/args/delivery) are tucked away for presets. "Test" runs the unsaved
  // config once. Saves to ~/.fusion/agent.json via the host.
  import { onDestroy, onMount } from 'svelte';
  import { agentConfig, agentTest, saveAgentConfig, settingsOpen, testAgent } from '../store';
  import type { AgentConfigWire } from '@fusion/shared';

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
  const build = (): AgentConfigWire => ({ ...cfg, args: argsText.trim() ? argsText.trim().split(/\s+/) : [] });
  function save(): void {
    if (!invalid) saveAgentConfig(build());
  }
  const close = (): void => settingsOpen.set(false);
  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') close();
    else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) save();
  }
  onMount(() => {
    agentTest.set(null);
    window.addEventListener('keydown', onKey);
  });
  onDestroy(() => window.removeEventListener('keydown', onKey));
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div class="modal-bg" on:click={close}>
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
  <div class="modal" role="dialog" on:click|stopPropagation>
    <h2>Agent settings</h2>

    <label>Agent
      <select bind:value={cfg.kind} on:change={applyPreset}>
        <option value="claude">Claude Code</option>
        <option value="codex">Codex</option>
        <option value="custom">Custom command</option>
        <option value="mock">Mock (test, no CLI)</option>
      </select>
    </label>

    {#if hasModel}
      <label>Model <span class="sub">(optional)</span>
        <input bind:value={cfg.model} placeholder={MODEL_HINT[cfg.kind]} />
      </label>
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
      <div class="adv-toggle" role="button" tabindex="0" on:click={() => (advanced = !advanced)}>
        {advanced ? '▾' : '▸'} Advanced
      </div>
    {/if}
    {#if advanced || isCustom}
      <label>Command <input bind:value={cfg.command} placeholder="claude" /></label>
      <label>Args <span class="sub">(space-separated; <code>{'{prompt}'}</code> = the prompt)</span>
        <input bind:value={argsText} placeholder={'-p {prompt}'} />
      </label>
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
      <button on:click={() => testAgent(build())} disabled={invalid || $agentTest?.pending}>
        {$agentTest?.pending ? 'Testing…' : 'Test'}
      </button>
      <span class="spacer"></span>
      <button class="ghost" on:click={close}>Cancel</button>
      <button on:click={save} disabled={invalid}>Save</button>
    </div>

    {#if $agentTest && !$agentTest.pending}
      <p class="test {$agentTest.ok ? 'ok' : 'bad'}">
        {$agentTest.ok ? '✓ reachable' : '✗ failed'} — {$agentTest.message}
      </p>
    {/if}
    <p class="hint">Uses your CLI’s own auth. Saved to <code>~/.fusion/agent.json</code>. Esc to close · ⌘↵ to save.</p>
  </div>
</div>
