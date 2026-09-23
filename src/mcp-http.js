import { createServer } from 'node:http';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { createMcpService } from './mcp.js';

const MAX_BODY_BYTES = 1024 * 1024;
const fail = (status, message) => Object.assign(new Error(message), { status });

function readJSON(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let size = 0;
    const timer = setTimeout(() => finish(fail(408, '请求正文接收超时')), 10000);
    timer.unref();
    const finish = (error, body) => {
      clearTimeout(timer);
      req.off('data', onData); req.off('end', onEnd); req.off('error', onError); req.off('aborted', onAbort);
      if (error) { req.resume(); reject(error); } else resolve(body);
    };
    const onData = (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) finish(fail(413, 'HTTP 请求正文超过 1 MiB'));
      else chunks.push(chunk);
    };
    const onEnd = () => {
      try { finish(null, JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch { finish(fail(400, '请求正文不是有效 JSON')); }
    };
    const onError = () => finish(fail(400, '请求正文读取失败'));
    const onAbort = () => finish(fail(400, '请求已断开'));
    req.on('data', onData); req.once('end', onEnd); req.once('error', onError); req.once('aborted', onAbort);
  });
}

/** 无状态 MCP HTTP 连接，共享有状态的模型缓存；不对外网提供未鉴权服务。 */
export async function startMcpHttpServer({ host = '127.0.0.1', port = 7777, ...options } = {}) {
  if (!['127.0.0.1', 'localhost', '::1'].includes(host)) throw new Error('HTTP 仅支持本机回环地址：127.0.0.1、localhost、::1');
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('port 必须是 0–65535 的整数');
  const address = host === 'localhost' ? '127.0.0.1' : host;
  const service = createMcpService(options), connections = new Set(), tasks = new Set();
  let closing = false, disposal, actualPort;

  function reply(res, status, message) {
    if (res.destroyed || res.writableEnded) return;
    if (res.headersSent) { res.destroy(); return; }
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
      ...(status === 405 ? { Allow: 'POST' } : {}), Connection: 'close' });
    res.end(JSON.stringify({ jsonrpc: '2.0', id: null, error: { code: status === 400 ? -32700 : -32000, message } }));
  }

  async function handle(req, res) {
    // 严格校验 Host / Origin，防止恶意网页借 DNS rebinding 调用本地推理。
    const authorities = new Set([`127.0.0.1:${actualPort}`, `localhost:${actualPort}`, `[::1]:${actualPort}`]);
    if (!authorities.has(req.headers.host?.toLowerCase())) return reply(res, 403, '不允许的 Host');
    const origin = req.headers.origin;
    if (origin !== undefined && ![...authorities].some((value) => origin === `http://${value}`)) return reply(res, 403, '不允许的 Origin');
    if (closing) return reply(res, 503, '服务正在关闭');
    if (req.url === '/health' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify({ status: 'ok', transport: 'streamable-http' }));
      return;
    }
    if (req.url !== '/mcp') return reply(res, 404, 'MCP 入口为 /mcp，健康检查为 /health');
    if (req.method !== 'POST') return reply(res, 405, '无状态 MCP 不提供 GET SSE 或 DELETE 会话操作，请使用 POST');
    if (connections.size >= 64) return reply(res, 503, 'HTTP 并发请求已满，请稍后重试');
    if (req.headers['content-type']?.split(';')[0].trim().toLowerCase() !== 'application/json') return reply(res, 415, 'Content-Type 必须为 application/json');
    if (Number(req.headers['content-length']) > MAX_BODY_BYTES) return reply(res, 413, 'HTTP 请求正文超过 1 MiB');
    const marker = {};
    connections.add(marker);
    let server;
    try {
      const body = await readJSON(req);
      if (closing || res.destroyed) return reply(res, 503, '服务正在关闭或连接已断开');
      // 每个请求独立的协议实例，避免多个客户端相同 JSON-RPC id 相互干扰。
      server = service.createServer('streamable-http');
      const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
      const disconnected = new Promise((resolve) => {
        res.once('close', () => { void server.close().catch(() => {}); resolve(); });
      });
      await server.connect(transport);
      if (res.destroyed) return;
      // SDK 的 JSON 响应 Promise 在断开时可能不结算，不能让它阻塞清理。
      await Promise.race([transport.handleRequest(req, res, body), disconnected]);
    } catch (error) {
      reply(res, error.status ?? 500, error.status ? error.message : 'MCP HTTP 请求处理失败');
    } finally {
      connections.delete(marker);
      await server?.close();
    }
  }

  const httpServer = createServer({ requestTimeout: 30000, headersTimeout: 10000, keepAliveTimeout: 5000 }, (req, res) => {
    const task = handle(req, res);
    tasks.add(task);
    void task.catch(() => reply(res, 500, 'MCP HTTP 请求处理失败')).finally(() => tasks.delete(task));
  });
  httpServer.maxConnections = 128;

  function close() {
    closing = true;
    return disposal ??= (async () => {
      const stopped = new Promise((resolve) => httpServer.close(resolve));
      try { await service.close(); }
      finally {
        // 模型在途调用已结束，此时才关闭仍在读取正文的连接。
        httpServer.closeAllConnections();
        await Promise.allSettled([...tasks]);
        await stopped;
      }
    })();
  }
  try {
    await new Promise((resolve, reject) => {
      httpServer.once('error', reject);
      httpServer.listen(port, address, () => {
        httpServer.off('error', reject);
        actualPort = httpServer.address().port;
        resolve();
      });
    });
  } catch (error) { await close(); throw error; }
  return { url: `http://${address === '::1' ? '[::1]' : address}:${actualPort}/mcp`, close };
}
