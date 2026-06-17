<script lang="ts">
  // Monaco editor pane for the standalone (Electron) host — VS Code's editor as a
  // library. This component is DYNAMICALLY imported (App.svelte) only when isDesktop,
  // so Monaco is code-split into its own chunk and never enters the VS Code bundle.
  import { onDestroy, onMount } from 'svelte';
  import { get } from 'svelte/store';
  // Slim Monaco: the editor API only, plus the Python language contribution (the one language we
  // use). This drops ~80 unused basic-language chunks + the json/css/html/ts language modes from
  // the bundle (3.3 MB → ~1 MB). The Python Monarch tokenizer is synchronous, so the single
  // editor.worker below still covers every editor service we touch.
  import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
  import 'monaco-editor/esm/vs/basic-languages/python/python.contribution';
  import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
  import { abstract, caretLine, density, doc, fmtMeta, fmtOp, fmtShape, revealTarget, showMeta, structure } from '../store';
  import { monacoTheme, theme } from '../theme';
  import { post } from '../vscode';
  import type { FileStructure } from '@fusion/shared';

  // Monaco needs a worker factory. Python uses a synchronous Monarch tokenizer (no
  // language worker), so the basic editor worker covers it; if it ever fails to load,
  // the editor still renders/edits/scrolls (workers power only background services).
  (self as unknown as { MonacoEnvironment?: unknown }).MonacoEnvironment = {
    getWorker: () => new editorWorker(),
  };

  let host: HTMLDivElement;
  let editor: monaco.editor.IStandaloneCodeEditor | undefined;
  let loadedPath: string | null = null;
  let shapeDecos: monaco.editor.IEditorDecorationsCollection | undefined;

  // Render traced shapes INLINE in the editor: gray ghost-text at each line's end
  // (changed shapes only, like the Blocks zone) + red markers on crash lines. This lets
  // big models be read with full code + horizontal scroll, not the narrow Blocks pane.
  function applyShapes(s: FileStructure, dens: 'changed' | 'all', abst: boolean, meta: boolean): void {
    if (!editor) return;
    const model = editor.getModel();
    if (!model) return;
    const lastLine = model.getLineCount();
    const decos: monaco.editor.IModelDeltaDecoration[] = [];
    const markers: monaco.editor.IMarkerData[] = [];
    for (const fn of s.functions) {
      for (const l of fn.lines) {
        if (l.line < 1 || l.line > lastLine) continue;
        const shown = dens === 'all' ? l.shapes : l.shapes.filter((x) => x.changed);
        const parts = shown.map((x) => {
          const base = `${x.varName}[${fmtShape(x.shape, s.dimNames, abst)}]`;
          const m = meta ? fmtMeta(x) : '';
          return m ? `${base} ${m}` : base;
        });
        if (l.op) parts.push('∗ ' + fmtOp(l.op, s.dimNames, abst));
        if (parts.length) {
          const col = model.getLineMaxColumn(l.line);
          decos.push({
            range: new monaco.Range(l.line, col, l.line, col),
            options: { after: { content: '   ' + parts.join('   '), inlineClassName: 'fusion-shape' } },
          });
        }
        if (l.problem) {
          markers.push({
            severity: monaco.MarkerSeverity.Error,
            message: l.problem.message,
            startLineNumber: l.line,
            startColumn: 1,
            endLineNumber: l.line,
            endColumn: model.getLineMaxColumn(l.line),
          });
        }
      }
    }
    // NaN/Inf sentinel: a warning marker on the first non-finite line (the print(x.isnan()) killer).
    if (s.nonFinite && s.nonFinite.line >= 1 && s.nonFinite.line <= lastLine) {
      markers.push({
        severity: monaco.MarkerSeverity.Warning,
        message: `Fusion: first NaN here — ${s.nonFinite.var} contains NaN`,
        startLineNumber: s.nonFinite.line,
        startColumn: 1,
        endLineNumber: s.nonFinite.line,
        endColumn: model.getLineMaxColumn(s.nonFinite.line),
      });
    }
    if (!shapeDecos) shapeDecos = editor.createDecorationsCollection();
    shapeDecos.set(decos);
    monaco.editor.setModelMarkers(model, 'fusion', markers);
  }

  onMount(() => {
    editor = monaco.editor.create(host, {
      value: '',
      language: 'python',
      theme: monacoTheme(get(theme)),
      automaticLayout: true,
      minimap: { enabled: false },
      fontSize: 13,
      scrollBeyondLastLine: false,
      renderWhitespace: 'selection',
      tabSize: 4,
    });
    editor.onDidChangeCursorPosition((e) => caretLine.set(e.position.lineNumber));
    // Cmd/Ctrl+S -> ask the host to write to disk + re-structure.
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      const d = get(doc);
      if (d && editor) post({ type: 'saveDocument', path: d.path, text: editor.getValue() });
    });
  });

  // Load source when a NEW file opens (path change) — never on re-trace, so edits survive.
  $: if (editor && $doc && $doc.path !== loadedPath) {
    loadedPath = $doc.path;
    editor.setValue($doc.text);
    const model = editor.getModel();
    if (model) monaco.editor.setModelLanguage(model, $doc.language || 'python');
  }

  // Re-apply inline shapes whenever the trace (or density / abstract view) updates for the loaded file.
  $: if (editor && $structure && $doc && $structure.path === $doc.path) applyShapes($structure, $density, $abstract, $showMeta);
  // Follow the app's light/dark toggle live.
  $: if (editor) monaco.editor.setTheme(monacoTheme($theme));

  // Reveal: scroll to a line when the cockpit asks (revealTarget.seq bumps each click).
  $: if (editor && $revealTarget.seq) {
    const ln = $revealTarget.line;
    editor.revealLineInCenter(ln);
    editor.setPosition({ lineNumber: ln, column: 1 });
    editor.focus();
  }

  onDestroy(() => editor?.dispose());
</script>

<div class="editor" bind:this={host}></div>

<style>
  .editor {
    width: 100%;
    height: 100%;
  }
</style>
