import { round4 } from './math.js';

const RANGES = [
  ['greek', [[0x370, 0x3ff], [0x1f00, 0x1fff]]],
  ['cyrillic', [[0x400, 0x52f], [0x2de0, 0x2dff], [0xa640, 0xa69f]]],
  ['armenian', [[0x530, 0x58f]]], ['hebrew', [[0x590, 0x5ff]]],
  ['arabic', [[0x600, 0x6ff], [0x750, 0x77f], [0x8a0, 0x8ff], [0xfb50, 0xfdff], [0xfe70, 0xfeff]]],
  ['devanagari', [[0x900, 0x97f], [0xa8e0, 0xa8ff]]], ['bengali', [[0x980, 0x9ff]]],
  ['gurmukhi', [[0xa00, 0xa7f]]], ['gujarati', [[0xa80, 0xaff]]], ['oriya', [[0xb00, 0xb7f]]],
  ['tamil', [[0xb80, 0xbff]]], ['telugu', [[0xc00, 0xc7f]]], ['kannada', [[0xc80, 0xcff]]],
  ['malayalam', [[0xd00, 0xd7f]]], ['sinhala', [[0xd80, 0xdff]]], ['thai', [[0xe00, 0xe7f]]],
  ['lao', [[0xe80, 0xeff]]], ['tibetan', [[0xf00, 0xfff]]], ['myanmar', [[0x1000, 0x109f]]],
  ['georgian', [[0x10a0, 0x10ff]]], ['ethiopic', [[0x1200, 0x137f]]], ['khmer', [[0x1780, 0x17ff]]],
  ['hangul', [[0x1100, 0x11ff], [0x3130, 0x318f], [0xac00, 0xd7af]]],
  ['kana', [[0x3040, 0x309f], [0x30a0, 0x30ff], [0x31f0, 0x31ff]]],
  ['han', [[0x3400, 0x4dbf], [0x4e00, 0x9fff], [0xf900, 0xfaff]]],
];
const STOP = Object.fromEntries(Object.entries({
  en: 'the and is are was were to of in for with that this it you have has not but on at be as from will can would there their what which please we i',
  fr: 'le la les des une est pour dans que qui avec sur pas plus nous vous être cette mais sont ont aux ce',
  de: 'der die das und ist ein eine den dem nicht mit für auf von zu sich auch werden wurde haben sind oder aber',
  es: 'el los las que por con para una es se del como pero son está este esta todo más muy hay sus',
  pt: 'os as que em um uma para com não é se do da dos das mas são está este esta muito pelo pela',
  it: 'il lo gli che di per con non è si del della sono questo questa anche come più nella alla',
  nl: 'het een van is op te dat niet met voor zijn aan door maar ook worden deze naar wordt',
  ro: 'și să este sunt care pentru din dar după până fără ale lui în fost acum vreau trebuie foarte acest această acesta aceasta mi ți vă nu',
}).map(([lg, words]) => [lg, new Set(words.split(' '))]));
const DIACRITICS = new Set('àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœăâîșțşţąćęłńśźżčďěňřšťůžőűğıāēģīķļņūžđ');

export function stateText(state, maxChars = 4000) {
  const leaves = [];
  function visit(v, depth = 0) {
    if (depth > 6 || v == null) return;
    if (typeof v === 'string') leaves.push(v);
    else if (typeof v === 'object') for (const child of Object.values(v)) visit(child, depth + 1);
  }
  visit(state);
  return Array.from(leaves.join(' ')).slice(0, maxChars).join('');
}
function counts(text) {
  const result = new Map(); let latin = 0;
  for (const ch of text) {
    if (!/\p{L}/u.test(ch)) continue;
    const cp = ch.codePointAt(0);
    if (cp < 0x250 || (cp >= 0x1e00 && cp <= 0x1eff)) { latin++; continue; }
    const found = RANGES.find(([, ranges]) => ranges.some(([lo, hi]) => cp >= lo && cp <= hi));
    if (found) result.set(found[0], (result.get(found[0]) ?? 0) + 1);
  }
  result.set('latin', latin);
  return result;
}
export function detectScript(text) {
  const list = [...counts(text)];
  list.sort((a, b) => b[1] - a[1]);
  return list[0]?.[1] ? list[0][0] : 'unknown';
}
export function scriptProfile(text) {
  const c = counts(text), total = [...c.values()].reduce((a, b) => a + b, 0);
  return Object.fromEntries([...c].filter(([, n]) => n).map(([k, n]) => [k, n / total]));
}

/** 拉丁语言仅为启发式；已知语言时优先由调用方提供 lang。 */
export function detectLanguage(state) {
  const text = stateText(state), script = detectScript(text), profile = scriptProfile(text);
  const base = { script, script_profile: profile, language: null, is_english: script === 'unknown',
    language_undecided: true, diacritic_rate: 0,
    non_latin_fraction: Object.keys(profile).length ? round4(1 - (profile.latin ?? 0)) : 0 };
  if (script !== 'latin') return base;
  const lower = text.toLowerCase(), chars = Array.from(lower);
  const rate = chars.filter((c) => DIACRITICS.has(c)).length / Math.max(1, chars.length);
  const nonEnglish = rate >= 0.02;
  const words = lower.match(/\p{L}+/gu) ?? [];
  let language = null;
  if (words.length >= 4) {
    const scores = Object.fromEntries(Object.entries(STOP).map(([lg, stop]) => [lg, words.filter((w) => stop.has(w)).length]));
    const [best, hits] = Object.entries(scores).filter(([lg]) => lg !== 'en').sort((a, b) => b[1] - a[1])[0];
    if (hits >= Math.max(2, scores.en + 2) || (nonEnglish && hits >= Math.max(2, scores.en))) language = best;
    else if (scores.en && !nonEnglish) language = 'en';
  }
  return { ...base, language, language_undecided: language === null, diacritic_rate: round4(rate),
    is_english: language === 'en' || (language === null && !nonEnglish) };
}
export const isEnglish = (state) => detectLanguage(state).is_english;
