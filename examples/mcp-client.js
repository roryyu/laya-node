import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

// 默认启动 stdio 子进程；--url 连接已有 HTTP 服务，不改动客户端配置。
const { values } = parseArgs({ options: {
  predict: { type: 'boolean' }, 'models-dir': { type: 'string' }, url: { type: 'string' },
} });
if (values.url !== undefined && values['models-dir'] !== undefined) {
  throw new Error('--url 与 --models-dir 不能同时使用；HTTP 模型目录由服务启动端配置');
}
const args = [fileURLToPath(new URL('../bin/laya-mcp.js', import.meta.url))];
if (values['models-dir']) args.push('--models-dir', values['models-dir']);
const client = new Client({ name: 'laya-example', version: '1.0.0' });
try {
  const transport = values.url !== undefined
    ? new StreamableHTTPClientTransport(new URL(values.url))
    : new StdioClientTransport({ command: process.execPath, args });
  await client.connect(transport);
  console.log((await client.listTools()).tools.map((tool) => tool.name));
  const status = await client.callTool({ name: 'laya_status', arguments: {} });
  console.log(status.structuredContent);
  if (values.predict) {
    const prediction = await client.callTool({ name: 'laya_predict', arguments: {
      state: { message: '今天重复扣款了，请帮我退款。' }, lang: 'zh',
      questions: { refund: { type: 'noul', instructions: 'Does the customer request a refund?' } },
    } }, undefined, { timeout: 180000 });
    if (prediction.isError) throw new Error(prediction.content[0].text);
    console.log(prediction.structuredContent);
  }
} finally { await client.close(); }
