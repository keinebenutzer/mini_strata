/** CTk options (KV_WORDS and --kv argument choices) are read from terminal/setup.py, not hardcoded. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {join, dirname, resolve} from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const scriptPath = join(HERE, '..', 'server', 'ctk_options.py');
const repoRoot = resolve(HERE, '..', '..', '..');

test('ctk_options.py reads KV_WORDS and --kv choices from the real setup.py', () => {
  const out = execFileSync('python3', [scriptPath, repoRoot], {encoding: 'utf8'});
  const data = JSON.parse(out);
  assert.ok(data.kv_words && Object.keys(data.kv_words).length, 'kv_words should be non-empty');
  assert.ok(data.kv_choices && data.kv_choices.length, 'kv_choices should be non-empty');
  assert.equal(data.kv_words['int8'], '8-bit');
  assert.equal(data.kv_words['q4_0'], '4-bit (Hadamard-rotated)');
  assert.equal(data.kv_words['turbo2'], '2-bit (Hadamard-rotated)');
  assert.equal(data.kv_words['turbo3'], '3-bit (Hadamard-rotated)');
  assert.equal(data.kv_words['k3v8'], '3-bit keys, 8-bit values (Hadamard-rotated)');
  assert.equal(data.kv_words['k8v4'], '8-bit keys, 4-bit values');
  assert.equal(data.kv_words['fp16'], '16-bit');
  for (const kv of ['int8', 'q4_0', 'turbo2', 'turbo3', 'k3v8', 'k8v4', 'fp16', 'fp8', 'k8v8']) {
    assert.equal(data.kv_choices.includes(kv), true, `--kv choices should include ${kv}`);
  }
  // Every choice that has a KV_WORDS entry must carry a non-empty word (fp8/k8v8 may lack one; the script reports both faithfully)
  for (const choice of data.kv_choices) {
    if (Object.prototype.hasOwnProperty.call(data.kv_words, choice)) {
      assert.ok(typeof data.kv_words[choice] === 'string' && data.kv_words[choice].length > 0, `KV_WORDS entry for ${choice} should be a non-empty string`);
    }
  }
});

test('ctk_options.py returns error on missing setup.py', () => {
  const out = execFileSync('python3', [scriptPath, '/nonexistent/path'], {encoding: 'utf8'});
  const data = JSON.parse(out);
  assert.ok(data.error, 'should return an error for missing setup.py');
});
