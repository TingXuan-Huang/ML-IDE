// Light/dark theming for the standalone (Electron) host. The UI renders against the VS
// Code `--vscode-*` design tokens; in VS Code those come from the real theme, but the
// desktop host has none — so here we inject a full palette and let the user flip it.
// VS Code webviews keep using VS Code's own theme (we never inject there).
import { get, writable } from 'svelte/store';

export type Theme = 'dark' | 'light';

// Full --vscode-* palette per theme (keys are the var name minus the `--vscode-` prefix).
// Dark mirrors VS Code "Dark+"; light mirrors "Light+" — the tokens this UI relies on.
const PALETTES: Record<Theme, Record<string, string>> = {
  dark: {
    foreground: '#cccccc',
    'editor-background': '#1e1e1e',
    'tab-inactiveBackground': '#2d2d2d',
    'tab-inactiveForeground': '#969696',
    'panel-border': '#3c3c3c',
    focusBorder: '#007fd4',
    'textLink-foreground': '#3794ff',
    'textLink-activeForeground': '#4daafc',
    descriptionForeground: '#9d9d9d',
    'editorWidget-background': '#252526',
    'symbolIcon-functionForeground': '#dcdcaa',
    'inputValidation-errorBackground': '#5a1d1d',
    errorForeground: '#f48771',
    'charts-blue': '#4daafc',
    'charts-yellow': '#d7ba7d',
    'button-background': '#0e639c',
    'button-foreground': '#ffffff',
    'input-background': '#3c3c3c',
    'input-foreground': '#cccccc',
  },
  light: {
    foreground: '#3b3b3b',
    'editor-background': '#ffffff',
    'tab-inactiveBackground': '#ececec',
    'tab-inactiveForeground': '#6f6f6f',
    'panel-border': '#e5e5e5',
    focusBorder: '#005fb8',
    'textLink-foreground': '#006ab1',
    'textLink-activeForeground': '#005299',
    descriptionForeground: '#6a6a6a',
    'editorWidget-background': '#f3f3f3',
    'symbolIcon-functionForeground': '#795e26',
    'inputValidation-errorBackground': '#f5d5d5',
    errorForeground: '#c72e2e',
    'charts-blue': '#1a7fd4',
    'charts-yellow': '#9a7700',
    'button-background': '#005fb8',
    'button-foreground': '#ffffff',
    'input-background': '#ffffff',
    'input-foreground': '#3b3b3b',
  },
};

const FONT: Record<string, string> = {
  'editor-font-family': 'ui-monospace,SFMono-Regular,Menlo,Monaco,monospace',
  'editor-font-size': '13px',
};

const KEY = 'fusion-theme';
const onDesktop = (): boolean => !!(window as unknown as { fusionHost?: unknown }).fusionHost;

function load(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    if (v === 'light' || v === 'dark') return v;
  } catch {
    /* localStorage unavailable */
  }
  return 'dark';
}

export const theme = writable<Theme>(load());

/** Apply a theme: inject the --vscode-* vars (desktop only — VS Code supplies its own),
 *  flip <html data-theme> (drives color-scheme + any theme-specific CSS), persist, and
 *  update the store so Monaco re-themes. Safe to call before the app mounts (no flash). */
export function applyTheme(t: Theme): void {
  theme.set(t);
  if (!onDesktop()) return; // VS Code owns its own theme — never touch it
  const root = document.documentElement;
  const vars = { ...PALETTES[t], ...FONT };
  for (const k in vars) root.style.setProperty(`--vscode-${k}`, vars[k]);
  root.setAttribute('data-theme', t);
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* ignore */
  }
}

export const toggleTheme = (): void => applyTheme(get(theme) === 'dark' ? 'light' : 'dark');

/** Monaco's built-in base theme for our light/dark choice. */
export const monacoTheme = (t: Theme): string => (t === 'light' ? 'vs' : 'vs-dark');
