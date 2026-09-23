// 仅供 tsc --noEmit 检查，不执行模型下载。
import {
  load, Agent, RLAgent, Router, RouteDecision, DEFAULT_MODELS,
  predictShortlist, embedFnFromAgent, shortlist_choice, triage_questions,
  type Questions, type ChoiceAnswer, type NoulAnswer, type ScoreAnswer,
} from '../src/index.js';
import { createMcpServer, createMcpService, type McpOptions, type McpService } from 'laya-node/mcp';
import { startMcpHttpServer, type McpHttpOptions } from 'laya-node/mcp/http';

const mcpOptions: McpOptions = { models: { multilingual: './models/multilingual' }, maxPending: 8 };
const mcp = createMcpServer(mcpOptions);
await mcp.close();
// @ts-expect-error MCP 不接受远程服务 URL 参数
createMcpServer({ url: 'https://example.com' });
const shared: McpService = createMcpService(mcpOptions);
const protocol = shared.createServer('streamable-http');
await protocol.close();
await shared.close();
// @ts-expect-error 不支持旧 SSE 传输标记
shared.createServer('sse');
const httpOptions: McpHttpOptions = { ...mcpOptions, host: '127.0.0.1', port: 0 };
const http = await startMcpHttpServer(httpOptions);
const url: string = http.url;
await http.close();
// @ts-expect-error HTTP 只允许本机回环地址
startMcpHttpServer({ host: '0.0.0.0' });
// @ts-expect-error HTTP 端口必须为数字
startMcpHttpServer({ port: '7777' });
void url;

const questions = {
  team: { type: 'choice', instructions: 'Choose', criteria: ['billing', 'tech'] },
  priority: { type: 'score', instructions: 'Urgency', criteria: ['low', 'high'] },
  refund: { type: 'noul', instructions: 'Refund?', criteria: { true: 0, false: false } },
} satisfies Questions;
const agent: Agent = await load(DEFAULT_MODELS.english, { localFilesOnly: true, device: 'cpu' });
const alias: typeof Agent = RLAgent;
const result = await agent.predict({ body: 'refund', count: 0, flag: false }, questions);
const choice: ChoiceAnswer = result.answers.team;
const score: ScoreAnswer = result.answers.priority;
const noul: NoulAnswer = result.answers.refund;
const zero: 0 = result.usage.output_tokens;
// @ts-expect-error choice 答案不包含 noul
result.answers.team.noul;
// @ts-expect-error 必须 await
const synchronous: ChoiceAnswer = agent.predict('x', questions).answers.team;
// @ts-expect-error 不支持的问题类型
await agent.predict('x', { bad: { type: 'unknown', instructions: 'x' } });
const router = new Router({ models: { english: './models/english' }, maxLoaded: 1 });
const routed = await router.predict('中文', questions, { lang: 'zh' });
const routing: RouteDecision = routed.routing;
await router.attach('english', agent);
await router.preload(['english']);
await predictShortlist(agent, 'x', questions, { k: 1, embedFn: embedFnFromAgent(agent) });
await shortlist_choice('x', ['a', 'b'], async () => [new Float32Array([1, 0]), [1, 0], [0, 1]], 1);
await agent.system_one('x', triage_questions());
await router.dispose();
void [alias, choice, score, noul, zero, synchronous, routing];
