import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { ModelName, RouterOptions } from './index.js';

export interface McpOptions {
  models?: Partial<Record<ModelName, string>>;
  maxLoaded?: number;
  maxPending?: number;
  allowTiny?: boolean;
  /** 用于测试或嵌入场景的模型加载器；stdio 入口不允许客户端替换。 */
  loader?: RouterOptions['loader'];
}
export interface McpService {
  createServer(transport?: 'stdio' | 'streamable-http'): McpServer;
  close(): Promise<void>;
}
export function createMcpService(options?: McpOptions): McpService;
export function createMcpServer(options?: McpOptions): { server: McpServer; close(): Promise<void> };
