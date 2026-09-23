#!/usr/bin/env node
import { parseArgs } from 'node:util';
import path from 'node:path';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { createMcpServer } from '../src/mcp.js';

const HELP = `laya-node-mcp：本地 stdio MCP / Streamable HTTP 服务（Node.js >=22）
用法：node bin/laya-mcp.js [--models-dir /absolute/models]
      node bin/laya-mcp.js --transport http --port 7777
  --transport stdio|http  默认 stdio，由客户端启动；http 为独立常驻服务
  --host ADDRESS          HTTP 监听地址，默认 127.0.0.1，仅支持回环地址
  --port N                HTTP 端口，默认 7777；MCP 路径 /mcp，健康检查 /health
  --english PATH          单独配置英语模型
  --multilingual PATH     单独配置中文及多语言模型
  --typed-decisions PATH  单独配置工作流模型
  --max-loaded N          最多保留的模型数，默认 1（切换时有瞬时峰值）
  --max-pending N         在途加排队请求上限，默认 8，最大 64
  --allow-tiny            仅协议测试允许使用随机 tiny 模型
环境变量：LAYA_MODELS_DIR、LAYA_MODEL_ENGLISH、LAYA_MODEL_MULTILINGUAL、
LAYA_MODEL_TYPED_DECISIONS；命令行覆盖环境变量。
未配置路径时使用 SDK 自身的 models/；不会自动下载模型。
stdio 模式 stdout 仅输出 MCP JSON-RPC；HTTP 模式通过网络通信。
HTTP 客户端只配置 URL，共享模型缓存与队列；日志与错误写入 stderr。
`;

async function main() {
  const { values: args } = parseArgs({ options: {
    help: { type: 'boolean', short: 'h' }, 'models-dir': { type: 'string' },
    transport: { type: 'string' }, host: { type: 'string' }, port: { type: 'string' },
    english: { type: 'string' }, multilingual: { type: 'string' }, 'typed-decisions': { type: 'string' },
    'max-loaded': { type: 'string' }, 'max-pending': { type: 'string' }, 'allow-tiny': { type: 'boolean' },
  } });
  if (args.help) { process.stdout.write(HELP); return; }
  const transport = args.transport ?? 'stdio';
  if (!['stdio', 'http'].includes(transport)) throw new Error('--transport 必须为 stdio 或 http');
  if (transport === 'stdio' && (args.host !== undefined || args.port !== undefined)) throw new Error('--host / --port 仅用于 --transport http');
  if (args.port !== undefined && !/^\d+$/.test(args.port)) throw new Error('--port 必须为整数');
  const root = args['models-dir'] ?? process.env.LAYA_MODELS_DIR;
  if (root !== undefined && (!root.trim() || /^[a-z][a-z0-9+.-]*:\/\//i.test(root))) throw new Error('--models-dir 必须是本地目录');
  const models = {};
  for (const name of ['english', 'multilingual', 'typed-decisions']) {
    const value = args[name] ?? process.env[`LAYA_MODEL_${name.replaceAll('-', '_').toUpperCase()}`] ?? (root ? path.join(root, name) : undefined);
    if (value !== undefined) models[name] = value;
  }
  // 防止依赖的普通日志破坏 stdio JSON-RPC；不记录用户输入。
  console.log = console.info = console.debug = (...values) => console.error(...values);
  const options = { models, allowTiny: args['allow-tiny'] === true,
    maxLoaded: args['max-loaded'] === undefined ? 1 : Number(args['max-loaded']),
    maxPending: args['max-pending'] === undefined ? 8 : Number(args['max-pending']),
  };
  const app = transport === 'http'
    ? await (await import('../src/mcp-http.js')).startMcpHttpServer({ ...options,
      host: args.host, port: args.port === undefined ? 7777 : Number(args.port) })
    : createMcpServer(options);
  let shuttingDown = false;
  const shutdown = () => {
    if (shuttingDown) return;
    shuttingDown = true;
    void app.close().catch((error) => {
      process.stderr.write(`laya-node-mcp: ${error.message}\n`);
      process.exitCode = 1;
    }).finally(() => {
      process.stdin.pause();
      process.removeListener('SIGINT', shutdown);
      process.removeListener('SIGTERM', shutdown);
    });
  };
  process.once('SIGINT', shutdown);
  process.once('SIGTERM', shutdown);
  if (transport === 'http') {
    process.stderr.write(`laya-node-mcp: Streamable HTTP 已启动 ${app.url}\n`);
  } else {
    process.stdin.once('end', shutdown);
    app.server.server.onerror = () => { process.stderr.write('laya-node-mcp: MCP 协议错误，请检查客户端消息格式\n'); };
    await app.server.connect(new StdioServerTransport());
  }
}

try { await main(); }
catch (error) {
  process.stderr.write(`laya-node-mcp: ${error.message}\n`);
  process.exitCode = 1;
}
