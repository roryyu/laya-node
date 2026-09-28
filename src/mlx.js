import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const TOOLS = fileURLToPath(new URL('../tools/', import.meta.url));

function findPython() {
  if (process.env.LAYA_MLX_PYTHON) return process.env.LAYA_MLX_PYTHON;
  // 优先项目 .venv
  const venv = fileURLToPath(new URL('../.venv/bin/python', import.meta.url));
  return venv;
}

export async function createMLXSidecar(options = {}) {
  const python = findPython();
  const args = [path.join(TOOLS, 'mlx_runtime.py'), 'serve'];
  if (options.fp16) args.push('--fp16');
  if (options.modelDir) args.push('--model-dir', options.modelDir);

  const proc = spawn(python, args, { stdio: ['pipe', 'pipe', 'inherit'] });
  const rl = createInterface({ input: proc.stdout, crlfDelay: Infinity });

  // 等待 ready 信号
  const info = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('MLX sidecar 启动超时（30s）')), 30000);
    const onLine = (line) => {
      try {
        const obj = JSON.parse(line);
        if (obj.ready) {
          clearTimeout(timer);
          rl.off('line', onLine);
          resolve(obj);
        }
      } catch {}
    };
    rl.on('line', onLine);
    proc.on('error', (e) => { clearTimeout(timer); reject(e); });
    proc.on('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`MLX sidecar 进程退出 code=${code}`));
    });
  });

  // 请求队列：Python 端按 stdin 顺序处理，Node 端顺序写入并等待
  let id = 0;
  let pending = null;
  const queue = [];

  rl.on('line', (line) => {
    if (!pending) return;
    try {
      const obj = JSON.parse(line);
      if (obj.error) pending.reject(new Error(obj.error));
      else pending.resolve(obj.outputs);
    } catch (e) {
      pending.reject(e);
    } finally {
      pending = null;
      flush();
    }
  });

  function flush() {
    if (pending || !queue.length) return;
    const { inputs, resolve, reject } = queue.shift();
    pending = { resolve, reject };
    proc.stdin.write(JSON.stringify({ id: ++id, inputs }) + '\n');
  }

  function run(inputs) {
    return new Promise((resolve, reject) => {
      queue.push({ inputs, resolve, reject });
      flush();
    });
  }

  return {
    dtype: info.dtype,
    modelDir: info.model_dir,
    run,
    dispose() {
      rl.close();
      if (!proc.killed) proc.kill();
      return once(proc, 'exit').catch(() => {});
    },
  };
}
