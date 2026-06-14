<script lang="ts" context="module">
  export interface TreeNode {
    name: string;
    path: string; // posix relpath under the workspace root
    dir: boolean;
    children: TreeNode[];
  }

  /** Fold a flat list of relpaths into a nested tree, dirs-first then alphabetical. */
  export function buildTree(files: string[]): TreeNode[] {
    const root: TreeNode = { name: '', path: '', dir: true, children: [] };
    for (const f of files) {
      const parts = f.split('/');
      let cur = root;
      let acc = '';
      parts.forEach((part, i) => {
        acc = acc ? `${acc}/${part}` : part;
        const isFile = i === parts.length - 1;
        let child = cur.children.find((c) => c.name === part && c.dir === !isFile);
        if (!child) {
          child = { name: part, path: acc, dir: !isFile, children: [] };
          cur.children.push(child);
        }
        cur = child;
      });
    }
    const sort = (n: TreeNode): void => {
      n.children.sort((a, b) => (a.dir === b.dir ? a.name.localeCompare(b.name) : a.dir ? -1 : 1));
      n.children.forEach(sort);
    };
    sort(root);
    return root.children;
  }
</script>

<script lang="ts">
  import { doc, folder, openTreeFile } from '../store';

  // Root instance builds the tree from $folder; recursive child instances get `nodes`.
  export let nodes: TreeNode[] | undefined = undefined;
  $: tree = nodes ?? buildTree($folder?.files ?? []);

  let collapsed = new Set<string>();
  function toggle(p: string): void {
    if (collapsed.has(p)) collapsed.delete(p);
    else collapsed.add(p);
    collapsed = collapsed; // reassign -> reactive
  }
  // Highlight the open file (its path ends with the relpath under the root).
  $: openRel = $doc && $folder && $doc.path.startsWith(`${$folder.root}/`) ? $doc.path.slice($folder.root.length + 1) : '';
</script>

{#each tree as n (n.path)}
  {#if n.dir}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div class="tnode tdir" role="button" tabindex="0" on:click={() => toggle(n.path)}>
      <span class="tcaret">{collapsed.has(n.path) ? '▸' : '▾'}</span>{n.name}
    </div>
    {#if !collapsed.has(n.path)}
      <div class="tchildren"><svelte:self nodes={n.children} /></div>
    {/if}
  {:else}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div class="tnode tfile" class:on={n.path === openRel} role="button" tabindex="0" on:click={() => openTreeFile(n.path)}>
      {n.name}
    </div>
  {/if}
{/each}
