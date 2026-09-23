import { positiveInteger } from './math.js';
const QUOTE_HEADERS = [/^\s*On .{0,300}wrote:\s*$/i, /^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}/i, /^\s*_{8,}\s*$/, /^\s*From:\s.+$/i];
const SIGNATURES = [/^\s*--\s*$/, /^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\p{L}\p{N}_ ,!.]*$/iu, /^\s*sent from my (iphone|android|mobile|ipad)/i];
const DISCLAIMER = /(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|if you (have )?received this (e-?mail|message) in error)/i;
export function cleanEmailBody(body, maxChars = 3000) {
  positiveInteger(maxChars, 'maxChars');
  if (body != null && typeof body !== 'string') throw new TypeError('邮件正文必须为字符串');
  const newline = String.fromCharCode(10);
  const text = (body ?? '').replace(/\r\n?/g, newline).split(String.fromCharCode(92) + 'n').join(newline);
  let lines = [];
  for (const line of text.split('\n')) {
    if (lines.length && QUOTE_HEADERS.some((p) => p.test(line))) break;
    if (!line.trimStart().startsWith('>')) lines.push(line.trimEnd());
  }
  for (let i = Math.max(1, Math.min(Math.floor(lines.length * 0.6), lines.length - 8)); i < lines.length; i++) {
    if (lines[i].trim().length <= 40 && SIGNATURES.some((p) => p.test(lines[i]))) { lines = lines.slice(0, i); break; }
  }
  const paragraphs = lines.join('\n').split(/\n\s*\n/).map((p) => {
    if (!DISCLAIMER.test(p)) return p;
    return p.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter((s) => s && !DISCLAIMER.test(s)).join(' ');
  });
  return Array.from(paragraphs.map((p) => p.trim()).filter(Boolean).join('\n\n').replace(/[ \t]+/g, ' ')).slice(0, maxChars).join('');
}
export function emailState(subject, body, { sender, clean = true, ...extra } = {}) {
  return { subject: (subject ?? '').trim(), body: clean ? cleanEmailBody(body) : body ?? '',
    ...(sender ? { from: sender } : {}), ...Object.fromEntries(Object.entries(extra).filter(([, v]) => v != null)) };
}
