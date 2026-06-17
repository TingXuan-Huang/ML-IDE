// Smoke test for HelperClient — spawns the real lens-helper and exercises the
// cross-language bridge + crash recovery. Run: node dist/helperClient.smoke.js
import * as path from 'path';
import { HelperClient } from './helperClient';

// Paths are derived from this compiled file's location (core/dist/), so the smoke test runs on
// any clone, not just the author's machine: repo root is two levels up from core/dist/.
const REPO = path.resolve(__dirname, '..', '..');
const HELPER_CWD = path.join(REPO, 'lens-helper');
const NPY = path.join(REPO, 'spike', 'sampledata', 'embeddings.npy');
const CSV = path.join(REPO, 'spike', 'sampledata', 'metrics.csv');

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(() => r(), ms));

async function main(): Promise<void> {
  const c = new HelperClient({ cwd: HELPER_CWD });

  console.log('ping     ->', JSON.stringify(await c.ping()));
  console.log('version  ->', JSON.stringify(await c.version()));

  const npy = (await c.loadFile(NPY)) as { kind: string; shape: number[] };
  console.log('load npy ->', npy.kind, JSON.stringify(npy.shape));

  const csv = (await c.loadFile(CSV)) as { kind: string; header: string[] };
  console.log('load csv ->', csv.kind, JSON.stringify(csv.header));

  // crash recovery: kill the process, then make another request
  c.killForTest();
  await sleep(250);
  const after = await c.ping();
  console.log('after-crash ping ->', JSON.stringify(after), '| restarts =', c.restartCount);

  c.dispose();
  console.log('\nSMOKE OK: bridge works + auto-restarted after a crash');
}

main().catch((e) => {
  console.error('SMOKE FAILED:', e);
  process.exit(1);
});
