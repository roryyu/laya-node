import { Router, triageQuestions } from '../src/index.js';

// 先分别导出 english 和 multilingual。仅观察路由时无需下载模型。
const router = new Router({ maxLoaded: 1 });
const questions = triageQuestions();
const states = [{ message: 'Please refund the duplicate charge.' }, { message: '发票重复扣款，请退款。' }];
try {
  for (const state of states) {
    console.log(router.route(state, questions));
    if (process.argv.includes('--predict')) console.log(await router.predict(state, questions));
  }
  // 服务端可 await router.preload(['english', 'multilingual'])，但会增加常驻内存。
} finally { await router.dispose(); }
