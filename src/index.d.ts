export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export type State = string | JsonValue[] | { [key: string]: JsonValue };
export type Criteria = Record<string, JsonValue> | string[];
export type Question =
  | { type: 'choice'; instructions: JsonValue; criteria: Criteria }
  | { type: 'score'; instructions: JsonValue; criteria: JsonValue[] }
  | { type: 'noul'; instructions: JsonValue; criteria?: { true?: JsonValue; false?: JsonValue } };
export type Questions = Record<string, Question>;
export interface ChoiceAnswer { type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number; action: { act_probability: number } }
export interface ScoreAnswer { type: 'score'; score: number; legend: Record<string, JsonValue>; probabilities: Record<string, number>; confidence: number; action: { act_probability: number } }
export interface NoulAnswer { type: 'noul'; noul: number; confidence: number; action: { act_probability: number } }
export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;
export type AnswerFor<Q> = Q extends { type: 'choice' } ? ChoiceAnswer : Q extends { type: 'score' } ? ScoreAnswer : Q extends { type: 'noul' } ? NoulAnswer : Answer;
export interface Prediction<Q extends Questions = Questions> {
  model: string;
  answers: { [K in keyof Q]: AnswerFor<Q[K]> };
  usage: { input_tokens: number; output_tokens: 0 };
  routing?: RouteDecision;
}
export interface LoadOptions {
  cacheDir?: string; revision?: string; localFilesOnly?: boolean; device?: 'cpu';
  progressCallback?: (event: Record<string, unknown>) => void;
  sessionOptions?: Record<string, unknown>;
}
export interface PredictOptions { maxLength?: number; headMaxLength?: number; truncateLeft?: boolean }
export interface EmbedOptions { maxLength?: number; batchSize?: number }
export interface AgentConfig {
  encoder: string; head_layers: number; max_len: number; head_max_len: number;
  temperature: number[]; temperature_by_options?: Record<string, number>; [key: string]: unknown;
}
export interface ModelMetadata {
  format_version: 1; checkpoint: ModelName; tiny: boolean; source: string; revision: string;
  hidden_size: number; max_position_embeddings: number; n_act: number; agent_config: AgentConfig;
}
export interface InternalQuestion { t: 'choice' | 'score' | 'noul'; ins: string; crit: Criteria | JsonValue[] | { true?: JsonValue; false?: JsonValue } }
export interface Batch { input_ids: number[][]; attention_mask: number[][]; marker_pos: number[][]; marker_mask: boolean[][]; qtype: number[] }
export interface Prepared {
  entries: [string, InternalQuestion][];
  items: { ids: number[]; markers: number[]; qtype: number }[];
  batch: Batch | null;
}
export interface RawOutput { logits: number[][]; act_logits: number[][]; embeddings: number[][] }
export class Agent {
  private constructor();
  static load(modelPath?: string, options?: LoadOptions): Promise<Agent>;
  cfg: AgentConfig;
  readonly metadata: ModelMetadata;
  readonly source: string;
  readonly tok: { encode(text: string, options?: { add_special_tokens?: boolean }): number[] };
  readonly temperature_raw: number[];
  readonly temperature_by_options_raw: Record<string, number>;
  readonly temperature: number[];
  readonly temperature_by_options: Record<string, number>;
  prepare(state: State, questions: Questions, options?: PredictOptions): Prepared;
  predict<Q extends Questions>(state: State, questions: Q, options?: PredictOptions): Promise<Prediction<Q>>;
  systemOne<Q extends Questions>(state: State, questions: Q, options?: PredictOptions): Promise<Prediction<Q>>;
  system_one<Q extends Questions>(state: State, questions: Q, options?: PredictOptions): Promise<Prediction<Q>>;
  predictRaw(state: State, questions: Questions, options?: PredictOptions): Promise<Prepared & { output: RawOutput | null }>;
  embed(texts: string[], options?: EmbedOptions): Promise<number[][]>;
  dispose(): Promise<unknown>;
}
export function load(modelPath?: string, options?: LoadOptions): Promise<Agent>;
export { Agent as RLAgent };
export type ModelName = 'english' | 'multilingual' | 'typed-decisions';
export interface RoutingOptions extends PredictOptions { model?: string; task?: string; lang?: string }
export interface LanguageDetection {
  script: string; script_profile: Record<string, number>; language: string | null; is_english: boolean;
  language_undecided: boolean; diacritic_rate: number; non_latin_fraction: number;
}
export class RouteDecision {
  constructor(model: ModelName, repo: string, reason: string, detection?: LanguageDetection | null, workflow?: string | null);
  model: ModelName; repo: string; reason: string; detection: LanguageDetection | null; workflow: string | null;
}
export interface RouterOptions extends LoadOptions {
  models?: Partial<Record<ModelName, string>>; maxLoaded?: number; default?: ModelName; autoTaskDetection?: boolean;
  loader?: (modelPath: string, options?: LoadOptions) => Promise<Agent>;
}
export class Router {
  constructor(options?: RouterOptions);
  models: Record<ModelName, string>; maxLoaded: number; default: ModelName; autoTaskDetection: boolean;
  readonly loaded: ModelName[];
  route(state: State | null, questions?: Questions, options?: RoutingOptions): RouteDecision;
  load(name: string): Promise<Agent>;
  predict<Q extends Questions>(state: State, questions: Q, options?: RoutingOptions): Promise<Prediction<Q> & { routing: RouteDecision }>;
  systemOne<Q extends Questions>(state: State, questions: Q, options?: RoutingOptions): Promise<Prediction<Q>>;
  system_one<Q extends Questions>(state: State, questions: Q, options?: RoutingOptions): Promise<Prediction<Q>>;
  preload(names?: string[]): Promise<this>;
  attach(name: string, agent: Agent): Promise<Agent>;
  unload(name?: string): Promise<void>;
  dispose(): Promise<unknown>;
}
export const DEFAULT_MODELS: Readonly<Record<ModelName, string>>;
export function detectLanguage(state: State | null): LanguageDetection;
export function detectScript(text: string): string;
export function isEnglish(state: State | null): boolean;
export function cleanEmailBody(body: string | null, maxChars?: number): string;
export function emailState(subject: string | null, body: string | null, options?: { sender?: string; clean?: boolean; [key: string]: JsonValue | undefined }): Record<string, JsonValue>;
export function triageQuestions(): Questions;
export function emailQuestions(categories?: Record<string, JsonValue>): Questions;
export function guardQuestions(): Questions;
export function moderationQuestions(): Questions;
export function routerQuestions(): Questions;
export type EmbeddingFunction = (texts: string[]) => Promise<(number[] | Float32Array | Float64Array)[]> | (number[] | Float32Array | Float64Array)[];
export interface ShortlistInfo { labels: string[]; scores: number[] | null; passthrough: boolean; original_count: number; probability_scope: 'all' | 'shortlisted' }
export function shortlistChoice(state: State, criteria: Criteria, embedFn: EmbeddingFunction, k?: number, options?: { instructions?: JsonValue }): Promise<string[]>;
export function predictShortlist<Q extends Questions>(agent: Agent | Router, state: State, questions: Q, options?: RoutingOptions & { embedFn?: EmbeddingFunction; k?: number }): Promise<Prediction<Q> & { shortlist: Record<string, ShortlistInfo> }>;
export function embedFnFromAgent(agent: Agent, options?: EmbedOptions): EmbeddingFunction;
export const QTYPES: Readonly<{ choice: 0; score: 1; noul: 2 }>;
export const QTYPE_NAMES: Readonly<{ 0: 'choice'; 1: 'score'; 2: 'noul' }>;
export const TEMP_MIN: 0.5;
export const TEMP_MAX: 5;
export const __version__: string;
export function serializeState(state: State): string;
export function renderOptions(question: Question | InternalQuestion): string[];
export function clampTemperature(value: unknown, lo?: number, hi?: number): number;
export function tempBucket(qtype: 0 | 1 | 2 | 'choice' | 'score' | 'noul', k: number): string;
export function softmax(logits: ArrayLike<number>, temperature?: number): number[];
export function confidenceFromProbs(p: ArrayLike<number>, k?: number): number;
export function eceScore(confidence: number[], correct: (boolean | number)[], bins?: number): number;
export function properReward(q: number[], target: number[], qtype: 0 | 1 | 2 | 'choice' | 'score' | 'noul', mask?: boolean[], options?: { wSph?: number; wRps?: number; logFloor?: number }): number;
export function tdLambdaTargets(pTrue: number[], batch: { target: number[][]; ep_group?: number[]; ep_step?: number[] }, lam?: number): number[][];
export {
  detectLanguage as detect_language, detectScript as detect_script, isEnglish as is_english,
  cleanEmailBody as clean_email_body, emailState as email_state, renderOptions as render_options,
  triageQuestions as triage_questions, emailQuestions as email_questions, guardQuestions as guard_questions,
  moderationQuestions as moderation_questions, routerQuestions as router_questions,
  shortlistChoice as shortlist_choice, predictShortlist as predict_shortlist, embedFnFromAgent as embed_fn_from_agent,
  properReward as proper_reward, tdLambdaTargets as td_lambda_targets, eceScore as ece_score,
  confidenceFromProbs as confidence_from_probs,
};
