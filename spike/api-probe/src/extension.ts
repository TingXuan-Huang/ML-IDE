// M0 SPIKE — VS Code API probe (THROWAWAY).
//
// Answers the outside-voice questions: does executeHoverProvider return a parseable
// TYPE for functions and intermediate variables? does CallHierarchy return real edges?
// Run via Run -> Start Debugging (F5 is the Dictation key on Mac), open a real torch .py
// in the [Extension Development Host] window, focus it, then Cmd+Shift+P ->
// "Fusion Spike: Probe APIs on active file". Read the "Fusion Spike" output channel.
import * as vscode from 'vscode';

declare function setTimeout(handler: (...args: unknown[]) => void, ms: number): unknown;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(() => r(), ms));

export function activate(context: vscode.ExtensionContext) {
  const out = vscode.window.createOutputChannel('Fusion Spike');
  context.subscriptions.push(
    vscode.commands.registerCommand('fusionSpike.probe', async () => {
      const ed = vscode.window.activeTextEditor;
      if (!ed) {
        vscode.window.showErrorMessage('Open a Python file and focus it first.');
        return;
      }
      const uri = ed.document.uri;
      out.clear();
      out.show(true);
      out.appendLine(`# Fusion API probe — ${uri.fsPath}`);
      out.appendLine(`# languageId: ${ed.document.languageId}\n`);

      // 0. Environment — is the Python language server even here? -------------
      out.appendLine('## Environment');
      const py = vscode.extensions.getExtension('ms-python.python');
      const pylance = vscode.extensions.getExtension('ms-python.vscode-pylance');
      out.appendLine(`   ms-python.python:          ${py ? (py.isActive ? 'ACTIVE' : 'installed (will activate)') : 'NOT INSTALLED'}`);
      out.appendLine(`   ms-python.vscode-pylance:  ${pylance ? (pylance.isActive ? 'ACTIVE' : 'installed') : 'NOT INSTALLED'}`);
      if (py && !py.isActive) {
        try {
          await py.activate();
          out.appendLine('   -> activated ms-python.python');
        } catch (e) {
          out.appendLine(`   -> activation failed: ${e}`);
        }
      }
      if (!pylance && !py) {
        out.appendLine('\n   STOP: no Python extension installed in this window.');
        out.appendLine('   Fix: Extensions panel (Cmd+Shift+X) -> search "Python" -> Install (it bundles Pylance).');
        out.appendLine('   Then re-run this command. Symbols/hovers/callhierarchy all come from Pylance.');
        return;
      }

      // 1. Structure via DocumentSymbols — RETRY until the LSP wakes up -------
      let symbols: vscode.DocumentSymbol[] = [];
      let tries = 0;
      for (; tries < 24; tries++) {
        symbols =
          (await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
            'vscode.executeDocumentSymbolProvider',
            uri,
          )) ?? [];
        if (symbols.length) break;
        await sleep(500); // give Pylance time to analyze the file (up to 12s)
      }
      const flat = flatten(symbols);
      out.appendLine(`\n## DocumentSymbols: ${flat.length} total  (ready after ${(tries * 0.5).toFixed(1)}s)`);
      if (flat.length === 0) {
        out.appendLine('   Still 0 after 12s. Either Pylance is disabled, the file is not type-checked,');
        out.appendLine('   or no interpreter is selected (bottom-left status bar). Select an interpreter and re-run.');
        return;
      }
      for (const s of flat) {
        out.appendLine(`   ${vscode.SymbolKind[s.kind]} ${s.name}  @L${s.selectionRange.start.line + 1}`);
      }

      const funcs = flat.filter(
        (s) => s.kind === vscode.SymbolKind.Function || s.kind === vscode.SymbolKind.Method,
      );

      // 2. Hover on function names — parseable type? -------------------------
      out.appendLine(`\n## Hover on function names (executeHoverProvider)`);
      for (const fn of funcs.slice(0, 10)) {
        out.appendLine(`   ${fn.name}: ${oneLine(await hoverAt(uri, fn.selectionRange.start))}`);
      }

      // 3. Hover on INTERIOR assignment targets — the hard case --------------
      out.appendLine(`\n## Hover on interior identifiers (do intermediate vars resolve a type?)`);
      let probed = 0;
      for (const fn of funcs) {
        if (probed >= 12) break;
        for (let line = fn.range.start.line + 1; line <= fn.range.end.line && probed < 12; line++) {
          const text = ed.document.lineAt(line).text;
          const m = /^\s*([A-Za-z_]\w*)\s*=/.exec(text);
          if (!m) continue;
          const col = text.indexOf(m[1]);
          out.appendLine(`   L${line + 1} ${m[1]} = ...  ->  ${oneLine(await hoverAt(uri, new vscode.Position(line, col)))}`);
          probed++;
        }
      }
      if (probed === 0) out.appendLine('   (no simple "name = ..." lines found to probe)');

      // 4. CallHierarchy — real edges? ---------------------------------------
      out.appendLine(`\n## CallHierarchy (prepare -> incoming/outgoing)`);
      for (const fn of funcs.slice(0, 8)) {
        const items =
          (await vscode.commands.executeCommand<vscode.CallHierarchyItem[]>(
            'vscode.prepareCallHierarchy',
            uri,
            fn.selectionRange.start,
          )) ?? [];
        if (items.length === 0) {
          out.appendLine(`   ${fn.name}: prepare -> none`);
          continue;
        }
        const inc =
          (await vscode.commands.executeCommand<vscode.CallHierarchyIncomingCall[]>(
            'vscode.provideIncomingCalls',
            items[0],
          )) ?? [];
        const og =
          (await vscode.commands.executeCommand<vscode.CallHierarchyOutgoingCall[]>(
            'vscode.provideOutgoingCalls',
            items[0],
          )) ?? [];
        out.appendLine(
          `   ${fn.name}: in=[${inc.map((c) => c.from.name).join(', ') || '-'}]  out=[${og.map((c) => c.to.name).join(', ') || '-'}]`,
        );
      }

      out.appendLine(`\n# VERDICT (read above):`);
      out.appendLine(`#  - function hovers showing "(function) f(x: Tensor) -> Tensor" => regex-able type`);
      out.appendLine(`#  - interior hovers mostly "(no type)" => static is thin => the TRACE carries Mode 1 (fine, proven)`);
      out.appendLine(`#  - CallHierarchy 'out=[...]' listing callees => the scoped graph (M4) is viable`);
    }),
  );
}

export function deactivate() {}

// --- helpers -------------------------------------------------------------------
function flatten(syms: vscode.DocumentSymbol[]): vscode.DocumentSymbol[] {
  const acc: vscode.DocumentSymbol[] = [];
  const walk = (list: vscode.DocumentSymbol[]) => {
    for (const s of list) {
      acc.push(s);
      if (s.children?.length) walk(s.children);
    }
  };
  walk(syms);
  return acc;
}

async function hoverAt(uri: vscode.Uri, pos: vscode.Position): Promise<string> {
  const hovers =
    (await vscode.commands.executeCommand<vscode.Hover[]>('vscode.executeHoverProvider', uri, pos)) ?? [];
  const parts: string[] = [];
  for (const h of hovers) {
    for (const c of h.contents) {
      if (typeof c === 'string') parts.push(c);
      else if ('value' in c) parts.push((c as vscode.MarkdownString).value);
    }
  }
  return parts.join(' ');
}

function oneLine(s: string): string {
  const t = s.replace(/```[a-z]*/g, '').replace(/\s+/g, ' ').trim();
  return t.length > 110 ? t.slice(0, 110) + '…' : t || '(no type)';
}
