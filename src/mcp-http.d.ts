import type { McpOptions } from './mcp.js';

export interface McpHttpOptions extends McpOptions {
  host?: '127.0.0.1' | 'localhost' | '::1';
  /** 默认 7777，0 由系统分配空闲端口。 */
  port?: number;
}
export function startMcpHttpServer(options?: McpHttpOptions): Promise<{ url: string; close(): Promise<void> }>;
