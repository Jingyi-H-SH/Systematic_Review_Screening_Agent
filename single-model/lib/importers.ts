import { unzipSync, strFromU8 } from 'fflate';
import type { Block } from './analysis';
import { pdfTextRows } from './pdf-layout';
export function readCSV(text: string): Record<string, string>[] {
  const rows: string[][] = []; let row: string[] = [], cell = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') { if (quoted && text[i + 1] === '"') { cell += '"'; i++; } else quoted = !quoted; }
    else if (c === ',' && !quoted) { row.push(cell); cell = ''; }
    else if ((c === '\n' || c === '\r') && !quoted) { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cell); if (row.some(s => s.trim())) rows.push(row); row = []; cell = ''; }
    else cell += c;
  }
  if (quoted) throw new Error('CSV has an unclosed quoted field.');
  row.push(cell); if (row.some(s => s.trim())) rows.push(row);
  const headers = rows.shift()?.map(s => s.trim().toLowerCase()) || [];
  if (!headers.includes('title')) throw new Error('CSV needs title, abstract and optional doi columns.');
  return rows.map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] || ''])));
}
export function readRIS(text: string) {
  const records: { title: string; abstract: string; doi: string }[] = []; let record = { title: '', abstract: '', doi: '' }; let last: keyof typeof record | null = null;
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/^([A-Z0-9]{2})\s{2}-\s?(.*)$/);
    if (!match) { if (last) record[last] += ' ' + line.trim(); continue; }
    const [, tag, value] = match;
    if (tag === 'TY') record = { title: '', abstract: '', doi: '' };
    last = ['TI', 'T1'].includes(tag) ? 'title' : ['AB', 'N2'].includes(tag) ? 'abstract' : tag === 'DO' ? 'doi' : null;
    if (last) record[last] = value;
    if (tag === 'ER' && record.title) records.push({ ...record });
  }
  if (!records.length) throw new Error('No complete RIS records found.'); return records;
}
function split(text: string, prefix: string, page: number | null, location: string): Block[] {
  const blocks: Block[] = []; let buffer = '', start = 1, line = 0;
  const push = () => { if (buffer.trim()) blocks.push({ id: `${prefix}-${blocks.length + 1}`, text: buffer.trim(), page, location: `${location}, lines ${start}–${line}` }); buffer = ''; start = line + 1; };
  for (const p of text.split(/\n/)) { if (buffer.length + p.length > 12000) push(); line++; if (p.length > 12000) { push(); for (let n = 0; n < p.length; n += 12000) blocks.push({ id: `${prefix}-${blocks.length + 1}`, text: p.slice(n, n + 12000), page, location: `${location}, line ${line}, part ${n / 12000 + 1}` }); } else buffer += p + '\n'; }
  push(); return blocks;
}
export async function parseDocument(file: File): Promise<{ blocks: Block[]; parser: string; warnings: string[] }> {
  if (file.size > 12 * 1024 * 1024) throw new Error('Maximum file size is 12 MB.');
  const ext = file.name.split('.').pop()?.toLowerCase(); const warnings: string[] = [];
  if (ext === 'txt') return { blocks: split(await file.text(), 'txt', null, 'Text document'), parser: 'UTF-8 text / line blocks v1', warnings: ['Plain text has no physical page numbers.'] };
  if (ext === 'docx') {
    const archive = unzipSync(new Uint8Array(await file.arrayBuffer())); if (!archive['word/document.xml']) throw new Error('Invalid DOCX.');
    const dom = new DOMParser().parseFromString(strFromU8(archive['word/document.xml']), 'application/xml');
    const body = dom.getElementsByTagName('w:body')[0]; const blocks: Block[] = []; let p = 0, t = 0;
    const text = (node: Element) => Array.from(node.getElementsByTagName('w:t')).map(n => n.textContent).join('');
    for (const node of Array.from(body.children)) {
      if (node.localName === 'p') { p++; blocks.push(...split(text(node), `p${p}`, null, `Paragraph ${p}`)); }
      if (node.localName === 'tbl') { t++; let r = 0; for (const tr of Array.from(node.getElementsByTagName('w:tr'))) { r++; const cells = Array.from(tr.getElementsByTagName('w:tc')).map((tc, i) => `[column ${i + 1}] ${text(tc)}`); blocks.push(...split(cells.join(' | '), `table${t}-row${r}`, null, `Table ${t}, row ${r}`)); } }
    }
    return { blocks, parser: 'DOCX XML paragraphs and table cells v1', warnings: ['DOCX locations use paragraphs and table rows; physical pages depend on Word layout.'] };
  }
  if (ext !== 'pdf') throw new Error('Choose PDF, DOCX or TXT.');
  const pdfjs = await import('pdfjs-dist'); pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';
  const pdf = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()), isEvalSupported: false }).promise;
  const blocks: Block[] = []; if (pdf.numPages > 150) { await pdf.destroy(); throw new Error('Maximum document length is 150 pages.'); }
  try { for (let n = 1; n <= pdf.numPages; n++) {
    const page = await pdf.getPage(n); const content = await page.getTextContent(); const output = pdfTextRows(content.items);
    if (output.trim().length < 20) warnings.push(`Page ${n} has little or no readable text; check the original image.`);
    blocks.push(...split(output, `pdf-p${n}`, n, `PDF page ${n}`)); page.cleanup();
  } } finally { await pdf.destroy(); }
  if (!blocks.length) throw new Error('This PDF has no extractable text. OCR it before analysis.');
  return { blocks, parser: 'PDF.js 5.4.624 / spatial row blocks v2', warnings };
}
