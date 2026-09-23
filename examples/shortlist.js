import { load, embedFnFromAgent, predictShortlist, DEFAULT_MODELS } from '../src/index.js';

const agent = await load(process.argv[2] ?? DEFAULT_MODELS.english);
try {
  const questions = { intent: { type: 'choice', instructions: 'Which intent best describes the request?', criteria: {
    refund: 'request money back', outage: 'service not available', sales: 'new purchase',
    cancel: 'cancel a subscription', password: 'reset password', invoice: 'get a receipt',
  } } };
  const result = await predictShortlist(agent, 'Please refund my duplicate payment.', questions, {
    k: 3, embedFn: embedFnFromAgent(agent),
  });
  // probabilities 只在入围标签之间归一化，不是完整标签集的概率。
  console.log(JSON.stringify(result, null, 2));
} finally { await agent.dispose(); }
