import {OPTIMIZED_ROUTING,STAGE_POLICY,runOptimizedScreening} from '../../../lib/optimized-screening';
import {MODEL_OPTIONS,SCREENING_MODEL,EXTRACTION_MODEL,selectedModel} from '../../../lib/model-options';
import {SCREENING_ROUTING,ADJUDICATION,consensusResult,checkedScreeningResult,screeningInput} from '../../../lib/screening';
import { env } from 'cloudflare:workers';
import { chunks, combineGeneral, extractionContext, screenSchema, extractionSchema, correlationSchema, generalSchema, GENERAL_EXTRACTION, validateAgainstSchema, validateExtraction, evidenceIssues, ROLE_INSTRUCTIONS, EXTRACTION, VERSION, MODEL, csv } from '../../../lib/analysis';
import type { Block, DocumentRecord, Extraction, Screening } from '../../../lib/analysis';
import { modelCall } from '../../../lib/model';
import { sourceUnits, linkedSchema, LINKED_INSTRUCTIONS, GENERAL_LINKED_INSTRUCTIONS, resolveLinked, statisticIssues } from '../../../lib/evidence-extraction';
import {matrixCandidates,extractionBatches,matrixValueIssues} from '../../../lib/matrices';
import {curateExtraction} from '../../../lib/masem-curation';
import {matrixMappingSchema,MATRIX_MAPPING_INSTRUCTIONS,extractMappedMatrices,statisticEvidenceUnits} from '../../../lib/matrix-extraction';

export const dynamic = 'force-dynamic';
const now = () => new Date().toISOString();
const json = (data: unknown, status = 200) => Response.json(data, { status, headers: { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' } });
function db() { if (!env.DB) throw new Error('Database unavailable.'); return env.DB; }
function bucket() { if (!env.BUCKET) throw new Error('File storage unavailable.'); return env.BUCKET; }
function required(value: unknown, name: string, max = 20000): string { if (typeof value !== 'string' || !value.trim() || value.length > max) throw new Error(`Invalid ${name}.`); return value.trim(); }
async function rawOne(table: 'projects' | 'studies' | 'jobs', id: string) { const row = await db().prepare(`SELECT * FROM ${table} WHERE id = ?`).bind(id).first<any>(); if (!row) throw new Error('Record not found.'); return row; }
class AccessError extends Error { constructor(message: string, public status: number) { super(message); } }
function userId(request: Request): string | null {
  const identity = request.headers.get('oai-authenticated-user-id');
  if (identity) return identity;
  const host = new URL(request.url).hostname;
  return ['127.0.0.1','localhost','[::1]'].includes(host) ? 'local-development' : null;
}
function requireUser(request: Request): string {
  const id = userId(request); if (!id) throw new AccessError('Sign in to access your review workspace.', 401); return id;
}
function scopedOne(owner: string) {
  return async (table: 'projects' | 'studies' | 'jobs', id: string) => {
    const row = await rawOne(table, id);
    const study = table === 'jobs' ? await rawOne('studies', row.study_id) : row;
    const project = table === 'projects' ? row : await rawOne('projects', study.project_id);
    if (project.owner_id !== owner) throw new AccessError('Record not found in your workspace.', 403);
    return row;
  };
}
function parsed<T>(raw: string | null, fallback: T): T { return raw ? JSON.parse(raw) : fallback; }
function sameOrigin(request: Request) { const origin = request.headers.get('origin'); if (origin && origin !== new URL(request.url).origin) throw new Error('Cross-origin request rejected.'); }
async function documentData(study: any): Promise<DocumentRecord | null> {
  if (!study.document) return null;
  const object = await bucket().get(study.document + '.json');
  if (!object) throw new Error('Source metadata missing.');
  return await object.json<DocumentRecord>();
}
function sourceBlocks(study: any): Block[] { return [{ id: 'title', text: study.title, page: null, location: 'Title' }, { id: 'abstract', text: study.abstract, page: null, location: 'Abstract' }]; }
function publicJob(row: any) { return { ...row, data: parsed(row.data, {}), lease: undefined }; }

// Sites authenticates visitors. Every project, study, file, task and export is scoped to its owner.
export async function GET(request: Request) {
  try {
    const url = new URL(request.url); const action = url.searchParams.get('action');
    if (action === 'health') return json({ version: VERSION, model: SCREENING_MODEL, models: MODEL_OPTIONS, screening_routing: OPTIMIZED_ROUTING, storage: !!env.DB && !!env.BUCKET, key_mode: 'bring-your-own', visibility: 'public landing page; authenticated per-user workspaces', server_key_fallback: false });
    if (action === 'identity') return json({ signed_in: !!userId(request), local: userId(request) === 'local-development' });
    const owner = requireUser(request), one = scopedOne(owner);
    if (action === 'projects') return json((await db().prepare('SELECT * FROM projects WHERE owner_id=? ORDER BY created DESC').bind(owner).all()).results);
    if (action === 'project') {
      const id = required(url.searchParams.get('id'), 'project ID', 100); const project = await one('projects', id);
      const studies = (await db().prepare('SELECT * FROM studies WHERE project_id = ? ORDER BY created').bind(id).all()).results;
      const jobs = (await db().prepare('SELECT jobs.* FROM jobs JOIN studies ON studies.id = jobs.study_id WHERE studies.project_id = ? ORDER BY jobs.created').bind(id).all<any>()).results.map(publicJob);
      const decisions = (await db().prepare('SELECT decisions.* FROM decisions JOIN studies ON studies.id = decisions.study_id WHERE studies.project_id = ? ORDER BY decisions.created').bind(id).all<any>()).results.map(r => ({ ...r, data: parsed(r.data, {}) }));
      const searches = (await db().prepare('SELECT * FROM searches WHERE project_id=? ORDER BY created').bind(id).all()).results;
      return json({ project, studies, jobs, decisions, searches });
    }
    if (action === 'source') { const study = await one('studies', required(url.searchParams.get('id'), 'study ID', 100)); return json(await documentData(study)); }
    if (action === 'file') {
      const study = await one('studies', required(url.searchParams.get('id'), 'study ID', 100));
      const key = url.searchParams.get('key') || study.document;
      // Historical originals can only be accessed if referenced by this study's immutable job input.
      if (key !== study.document) {
        const old = (await db().prepare('SELECT data FROM jobs WHERE study_id = ?').bind(study.id).all<any>()).results;
        const reviewed = (await db().prepare('SELECT data FROM decisions WHERE study_id = ?').bind(study.id).all<any>()).results;
        if (!old.some(r => parsed<any>(r.data, {}).input?.document?.key === key) && !reviewed.some(r => parsed<any>(r.data, {}).source_key === key)) throw new Error('Source does not belong to this study.');
      }
      const file = key && await bucket().get(key); if (!file) throw new Error('Original file unavailable.');
      return new Response(file.body, { headers: { 'Content-Type': file.httpMetadata?.contentType || 'application/octet-stream', 'Content-Disposition': 'inline; filename="source-document"', 'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff' } });
    }
    if (action === 'export') {
      const id = required(url.searchParams.get('id'), 'project ID', 100); const project = await one('projects', id);
      const studies = (await db().prepare('SELECT * FROM studies WHERE project_id = ?').bind(id).all<any>()).results;
      const jobs = (await db().prepare('SELECT jobs.* FROM jobs JOIN studies ON studies.id = jobs.study_id WHERE studies.project_id = ?').bind(id).all<any>()).results.map(publicJob);
      const decisions = (await db().prepare('SELECT decisions.* FROM decisions JOIN studies ON studies.id = decisions.study_id WHERE studies.project_id = ? ORDER BY decisions.created').bind(id).all<any>()).results.map(r => ({ ...r, data: parsed<any>(r.data, {}) }));
      const searches = (await db().prepare('SELECT * FROM searches WHERE project_id=? ORDER BY created').bind(id).all()).results;
      if (url.searchParams.get('format') === 'decisions') {
        const latest = new Map<string, any>();
        for (const d of decisions) if (['screen', 'fulltext'].includes(d.kind)) latest.set(d.study_id + ':' + d.kind, d);
        const rows = studies.map((s:any) => {
          const screen = latest.get(s.id + ':screen'), full = latest.get(s.id + ':fulltext');
          return { report_id:s.id, title:s.title, doi:s.doi, title_abstract_decision:screen?.data.decision || '', title_abstract_reason:screen?.data.rationale || '', title_abstract_reviewer:screen?.actor || '', fulltext_decision:full?.data.decision || '', fulltext_reason:full?.data.rationale || '', fulltext_reviewer:full?.actor || '', source_available:!!s.document };
        });
        const columns=['report_id','title','doi','title_abstract_decision','title_abstract_reason','title_abstract_reviewer','fulltext_decision','fulltext_reason','fulltext_reviewer','source_available'];
        return new Response(csv(rows,columns), {headers:{'Content-Type':'text/csv;charset=utf-8','Content-Disposition':'attachment; filename="screening-decisions.csv"','Cache-Control':'no-store'}});
      }
      if (url.searchParams.get('format') === 'csv') {
        const kind = url.searchParams.get('kind') === 'data' ? 'data' : 'extract';
        const latest = new Map<string, any>(); for (const d of decisions) if (d.kind === kind) latest.set(d.study_id, d);
        const rows = [...latest.values()].filter(d => d.data.status === 'approved').flatMap(d => (d.data.extraction?.records || []).map((r: any, i: number) => ({ report_id: d.study_id, ...r, validation_issues: d.data.issues?.[i] || [], reviewer: d.actor, reviewed_at: d.created, source_job_id: d.job_id })));
        const columns = kind === 'data' ? ['field','value','missing_status','evidence','notes'] : Object.keys(correlationSchema.properties);
        return new Response(csv(rows, ['report_id', ...columns, 'validation_issues', 'reviewer', 'reviewed_at', 'source_job_id']), { headers: { 'Content-Type': 'text/csv;charset=utf-8', 'Content-Disposition': `attachment; filename="reviewed-${kind}.csv"`, 'Cache-Control': 'no-store' } });
      }
      return new Response(JSON.stringify({ version: VERSION, project, studies, jobs, decisions, searches }, null, 2), { headers: { 'Content-Type': 'application/json', 'Content-Disposition': 'attachment; filename="sr-mas-audit.json"', 'Cache-Control': 'no-store' } });
    }
    return json({ error: 'Unknown action.' }, 404);
  } catch (e) { return json({ error: e instanceof Error ? e.message : 'Request failed.' }, e instanceof AccessError ? e.status : 400); }
}

export async function POST(request: Request) {
  try {
    sameOrigin(request);
    const owner = requireUser(request), one = scopedOne(owner);
    const length = Number(request.headers.get('content-length') || 0); if (length > 20 * 1024 * 1024) return json({ error: 'Upload exceeds 20 MB.' }, 413);
    if (request.headers.get('content-type')?.includes('multipart/form-data')) {
      const form = await request.formData(); const study = await one('studies', required(form.get('study_id'), 'study ID', 100));
      const file = form.get('file'); if (!(file instanceof File) || !file.size || file.size > 12 * 1024 * 1024) throw new Error('Choose a PDF, DOCX or TXT file up to 12 MB.');
      const ext = file.name.split('.').pop()?.toLowerCase(); if (!['pdf', 'docx', 'txt'].includes(ext || '')) throw new Error('Unsupported document type.');
      const raw = await file.arrayBuffer(); const prefix = new Uint8Array(raw).slice(0, 5);
      if (ext === 'pdf' && new TextDecoder().decode(prefix) !== '%PDF-') throw new Error('File is not a PDF.');
      if (ext === 'docx' && !(prefix[0] === 80 && prefix[1] === 75)) throw new Error('File is not a DOCX.');
      const blocks = JSON.parse(required(form.get('blocks'), 'source blocks', 2000000)) as Block[];
      if (!Array.isArray(blocks) || !blocks.length || blocks.length > 3000) throw new Error('No readable source blocks. Use a searchable document or OCR it first.');
      const ids = new Set<string>(); let total = 0;
      for (const b of blocks) {
        required(b.id, 'block ID', 100); required(b.text, 'source text', 24000); required(b.location, 'source location', 300);
        if (ids.has(b.id) || (b.page !== null && (!Number.isInteger(b.page) || b.page < 1))) throw new Error('Invalid source locations.');
        ids.add(b.id); total += b.text.length;
      }
      if (total < 30 || total > 500000) throw new Error('Document must contain 30–500,000 readable characters.');
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', raw))).map(x => x.toString(16).padStart(2, '0')).join('');
      const key = `sources/${study.id}/${crypto.randomUUID()}.${ext}`;
      const metadata: DocumentRecord = { key, name: file.name, sha256: hash, blocks, parser: required(form.get('parser'), 'parser', 200), warnings: JSON.parse(String(form.get('warnings') || '[]')) };
      await bucket().put(key, raw, { httpMetadata: { contentType: ext === 'pdf' ? 'application/pdf' : ext === 'txt' ? 'text/plain' : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' } });
      await bucket().put(key + '.json', JSON.stringify(metadata));
      await db().prepare('UPDATE studies SET document = ? WHERE id = ?').bind(key, study.id).run();
      return json(metadata);
    }
    const body = await request.json() as any;
    if (body.action === 'create_project') {
      const id = crypto.randomUUID();
      await db().prepare('INSERT INTO projects (id,title,criteria,constructs,created,review_type,extraction_template,owner_id) VALUES (?,?,?,?,?,?,?,?)').bind(id, required(body.title, 'title', 300), required(body.criteria, 'criteria'), String(body.constructs || 'Not specified'), now(), body.review_type === 'masem' ? 'masem' : 'general', String(body.extraction_template || 'Study design; country and setting; population; sample size; intervention or exposure; comparator; outcomes; follow-up; effect estimates with uncertainty; funding; limitations'), owner).run();
      return json({ id });
    }
    if (body.action === 'search_log') {
      await one('projects', required(body.project_id, 'project ID', 100));
      if (!Number.isInteger(body.count) || body.count < 0) throw new Error('Record count must be a nonnegative integer.');
      if (!/^\d{4}-\d{2}-\d{2}$/.test(body.searched)) throw new Error('Enter the search date.');
      await db().prepare('INSERT INTO searches (id,project_id,database,query,searched,count,created) VALUES (?,?,?,?,?,?,?)').bind(crypto.randomUUID(),body.project_id,required(body.database,'database',300),required(body.query,'search query'),body.searched,body.count,now()).run();
      return json({ saved: true });
    }
    if (body.action === 'quality') {
      const study = await one('studies',required(body.study_id,'study ID',100));
      const data = { tool: required(body.tool,'assessment tool',200), domains: required(body.domains,'domain judgments and supporting evidence',20000), overall: required(body.overall,'overall judgment',300), source_key: study.document };
      await db().prepare('INSERT INTO decisions (id,study_id,job_id,kind,actor,data,created) VALUES (?,?,NULL,?,?,?,?)').bind(crypto.randomUUID(),study.id,'quality',required(body.reviewer,'reviewer name',200),JSON.stringify(data),now()).run();
      return json({ saved: true });
    }
    if (body.action === 'manual_review') {
      const study = await one('studies', required(body.study_id, 'study ID', 100));
      if (!['screen', 'fulltext'].includes(body.kind)) throw new Error('Choose title/abstract or full-text screening.');
      if (body.kind === 'fulltext' && !study.document) throw new Error('Upload the source full text before recording its eligibility.');
      if (!['include', 'exclude', 'uncertain'].includes(body.decision)) throw new Error('Choose an eligibility decision.');
      const project = await one('projects', study.project_id);
      const data = { decision: body.decision, rationale: required(body.rationale, 'review rationale', 10000), method: 'manual', source_key: body.kind === 'fulltext' ? study.document : null, input: { title: study.title, abstract: study.abstract, criteria: project.criteria } };
      await db().prepare('INSERT INTO decisions (id,study_id,job_id,kind,actor,data,created) VALUES (?,?,NULL,?,?,?,?)').bind(crypto.randomUUID(), study.id, body.kind, required(body.reviewer, 'reviewer name', 200), JSON.stringify(data), now()).run();
      return json({ saved: true });
    }
    if (body.action === 'import') {
      await one('projects', required(body.project_id, 'project ID', 100));
      if (!Array.isArray(body.records) || !body.records.length || body.records.length > 500) throw new Error('Import 1–500 records at a time.');
      const existing = (await db().prepare('SELECT title, doi FROM studies WHERE project_id = ?').bind(body.project_id).all<any>()).results;
      const normalizedTitle = (title: string) => title.toLowerCase().replace(/\s+/g, ' ').trim();
      const seenDois = new Set(existing.filter(r => r.doi).map(r => r.doi.toLowerCase()));
      const seenTitles = new Set(existing.map(r => normalizedTitle(r.title))); let skipped = 0;
      const rows = body.records.flatMap((r: any) => {
        const title = required(r.title, 'article title', 1500); const abstract = String(r.abstract || '').slice(0, 40000); const doi = String(r.doi || '').trim().replace(/^https?:\/\/(?:dx\.)?doi.org\//i, '').slice(0, 300);
        const titleKey = normalizedTitle(title);
        if ((doi && seenDois.has(doi.toLowerCase())) || seenTitles.has(titleKey)) { skipped++; return []; }
        if (doi) seenDois.add(doi.toLowerCase()); seenTitles.add(titleKey);
        return [{ id: crypto.randomUUID(), title, abstract, doi }];
      });
      for (let start = 0; start < rows.length; start += 50) await db().batch(rows.slice(start, start + 50).map((r: { id: string; title: string; abstract: string; doi: string }) => db().prepare('INSERT INTO studies (id,project_id,title,abstract,doi,created) VALUES (?,?,?,?,?,?)').bind(r.id, body.project_id, r.title, r.abstract, r.doi, now())));
      return json({ imported: rows.length, skipped });
    }
    if (body.action === 'close_job') {
      const job = await one('jobs', required(body.job_id, 'task ID', 100));
      if (!['queued','error'].includes(job.status)) throw new Error('Pause the run and wait for its current call before closing it.');
      await db().prepare("UPDATE jobs SET status='closed',lease=NULL,updated=? WHERE id=? AND status IN ('queued','error')").bind(now(),job.id).run();
      return json({closed:true});
    }
    if (body.action === 'create_job') {
      const study = await one('studies', required(body.study_id, 'study ID', 100)); const project = await one('projects', study.project_id);
      if (!['screen', 'fulltext', 'extract', 'data'].includes(body.kind)) throw new Error('Invalid analysis stage.');
      const doc = body.kind === 'screen' ? null : await documentData(study);
      if (body.kind !== 'screen' && !doc) throw new Error('Upload a readable full text first.');
      const blocks = doc?.blocks || sourceBlocks(study);
      const model=selectedModel(body.model,body.kind==='screen'?SCREENING_MODEL:EXTRACTION_MODEL);
      const input = { report_id: study.id, title: study.title, abstract: study.abstract, doi: study.doi, criteria: project.criteria, constructs: project.constructs, extraction_template: project.extraction_template, blocks, document: doc, stage: body.kind };
      const id = crypto.randomUUID(); const timestamp = now();
      const insert = await db().prepare("INSERT INTO jobs (id,study_id,kind,status,phase,data,created,updated) SELECT ?,?,?, 'queued',0,?,?,? WHERE NOT EXISTS (SELECT 1 FROM jobs WHERE study_id=? AND kind=? AND status IN ('queued','running'))").bind(id, study.id, body.kind, JSON.stringify({ version: VERSION, input, model, screening_policy:body.kind==='screen'?STAGE_POLICY:undefined, evidence_mode:'source-units-v1', screening_routing:body.kind==='screen'?OPTIMIZED_ROUTING:undefined, pipeline_revision:'matrix-v13-typed-cells', extraction_plan:body.kind==='extract'?'matrix-first-v1':undefined, instructions: { extraction: body.kind === 'data' ? GENERAL_LINKED_INSTRUCTIONS : EXTRACTION+'\n'+LINKED_INSTRUCTIONS, reviewers: ROLE_INSTRUCTIONS }, extraction_batch_limit: body.kind === 'extract' ? 16000 : 12000, reasoning_effort: body.kind==='screen'?'medium':'low', calls: [], outputs: [],raw_outputs:[],provenance:[] }), timestamp, timestamp, study.id, body.kind).run();
      if (!insert.meta.changes) throw new Error('This stage already has an active task. Resume it instead.');
      return json({ id });
    }
    if (body.action === 'step') {
      const job = await one('jobs', required(body.job_id, 'task ID', 100));
      if (job.status === 'completed') return json(publicJob(job));
      const key = String(body.api_key || '').trim(); if (!key.startsWith('sk-')) throw new Error('Enter your OpenAI API key in Settings.');
      const lease = crypto.randomUUID(); const stale = new Date(Date.now() - 240000).toISOString();
      const claim = await db().prepare("UPDATE jobs SET lease=?, status='running', error=NULL, updated=? WHERE id=? AND (status IN ('queued','error') OR (status='running' AND updated<?))").bind(lease, now(), job.id, stale).run();
      if (!claim.meta.changes) return json({ error: 'This task is already processing. Refresh in a moment.' }, 409);
      const data = parsed<any>(job.data, {}); const input = data.input; let phase = job.phase; let done = false;
      try {
        let result;
         if (job.kind === 'extract' || job.kind === 'data') {
          const batches = extractionBatches(input.blocks, data.extraction_batch_limit || 24000,data.extraction_plan);
          const linked=data.evidence_mode==='source-units-v1';
          const focus=linked?sourceUnits(batches[phase]):batches[phase];
          const context=job.kind==='extract'?extractionContext(input.blocks,batches[phase]):[];
          const contextUnits=linked?sourceUnits(context):context;
          const matrices=job.kind==='extract'&&data.extraction_plan==='matrix-first-v1'?matrixCandidates(batches[phase]):[];
          const tableTask=matrices.length?'\nThis part contains mechanically reconstructed matrices. Inspect EACH requested construct pair in their off-diagonal cells first. The cells are source-derived, not gold labels. Resolve aliases from the original text. For each extracted matrix coefficient, set row and column to the exact supplied header tokens and cite its source unit. Do not replace a matrix correlation with a path coefficient.':'';
          const mappingMode=matrices.length>0&&['matrix-v3-mapping-and-cells','matrix-v4-explicit-aliases','matrix-v5-named-scope','matrix-v6-source-aliases','matrix-v12-cell-curated','matrix-v13-typed-cells'].includes(data.pipeline_revision);
          if(mappingMode){
            const units=sourceUnits([...batches[phase],...context]);
            result=await modelCall(key,MATRIX_MAPPING_INSTRUCTIONS,{report_id:input.report_id,constructs:input.constructs,blocks:focus,context_blocks:contextUnits,statistic_evidence_candidates:statisticEvidenceUnits(units),matrices},matrixMappingSchema(matrices,units,data.pipeline_revision==='matrix-v13-typed-cells'),data.reasoning_effort||'low',data.model||MODEL);
            data.raw_outputs.push(result.result);const mapped=extractMappedMatrices(result.result,matrices,units,input.blocks,input.report_id,['matrix-v4-explicit-aliases','matrix-v5-named-scope','matrix-v6-source-aliases','matrix-v12-cell-curated','matrix-v13-typed-cells'].includes(data.pipeline_revision)?input.constructs:'');data.provenance.push(mapped.provenance);result.result=mapped.result;
          } else result = await modelCall(key, (data.instructions?.extraction || (job.kind === 'data' ? GENERAL_EXTRACTION : EXTRACTION))+tableTask, { stage:job.kind, report_id: input.report_id, title: input.title, constructs: input.constructs, extraction_template: input.extraction_template, blocks:focus, context_blocks:contextUnits, reconstructed_matrices:matrices, part: phase + 1, total_parts: batches.length }, linked?linkedSchema(job.kind,sourceUnits([...batches[phase],...context]).map(u=>u.id)):job.kind === 'data' ? generalSchema : extractionSchema, data.reasoning_effort || 'low',data.model||MODEL);
          if(linked&&!mappingMode){data.raw_outputs.push(result.result);const resolved=resolveLinked(result.result,sourceUnits([...batches[phase],...context]));data.provenance.push(resolved.provenance);result.result=resolved.result;}
          data.outputs.push(result.result); phase++; done = phase >= batches.length;
          if (done) { const curated=job.kind==='extract'&&['matrix-v12-cell-curated','matrix-v13-typed-cells'].includes(data.pipeline_revision)?curateExtraction(data.outputs,input.blocks,input.constructs,input.report_id):null; if(curated)data.curation=curated.curation; data.result = job.kind === 'data' ? combineGeneral(data.outputs) : curated?.result || { records: data.outputs.flatMap((r: Extraction) => r.records), notes: data.outputs.map((r: Extraction) => r.notes).filter(Boolean).join('\n') }; data.issues = job.kind === 'data' ? data.result.records.map((r:any) => r.missing_status==='not_reported' ? [] : evidenceIssues(r.evidence,input.blocks)) : validateExtraction(data.result, input.blocks).map((issues,i)=>[...issues,...statisticIssues(data.result.records[i]),...matrixValueIssues(data.result.records[i],input.blocks)]); }
        } else if(job.kind==='screen'&&data.screening_routing===OPTIMIZED_ROUTING){
          const single=await runOptimizedScreening(key,input,data.model||SCREENING_MODEL,data.screening_policy||STAGE_POLICY);
          result={result:single.result,metadata:single.metadata};
          data.outputs.push(single.raw);data.raw_outputs.push(single.raw);phase=1;done=true;
          data.result=single.result;
          data.routing={policy:OPTIMIZED_ROUTING,reason:'single model with stage-specific screening policy',calls:1,review_required:single.review_required};
          data.issues=(single.result as Screening).criteria_checks.flatMap(c=>c.evidence.length?evidenceIssues(c.evidence,input.blocks):c.finding!=='unclear'?['A resolved criterion has no cited passage.']:[]);
          if(single.review_required)data.issues.push('The exclusion had no source-matched failed criterion. Retained as uncertain for researcher review.');
        } else {
          const prompt = phase < 3 ? (data.instructions?.reviewers || ROLE_INSTRUCTIONS)[phase] : ADJUDICATION;
          const eligibilityInput=job.kind==='screen'&&data.screening_routing===SCREENING_ROUTING?screeningInput(input):{...input,document:undefined};
          result = await modelCall(key, prompt, { ...eligibilityInput, reviewers: phase === 3 ? data.outputs : undefined }, screenSchema,data.reasoning_effort||'low',data.model||MODEL);
          data.outputs.push(result.result); phase++;
          const consensus=job.kind==='screen'&&data.screening_routing===SCREENING_ROUTING?consensusResult(data.outputs,input.blocks):null;
          done=!!consensus||phase===4;
          if(done){const selected=checkedScreeningResult(consensus||result.result,input.blocks);result.result=selected.result;data.routing={policy:data.screening_routing||'fixed-four',reason:consensus?'three exact agreements':'adjudicated',calls:phase,review_required:selected.review_required};}
          if (done) { data.result = result.result; data.issues = (result.result as Screening).criteria_checks.flatMap(c => c.evidence.length ? evidenceIssues(c.evidence, input.blocks) : c.finding !== 'unclear' ? ['A resolved criterion has no cited passage.'] : []); if(data.routing?.review_required)data.issues.push('No source-matched failed criterion supports this exclusion suggestion. Verify eligibility before saving a final decision.'); }
        }
        data.calls.push(result.metadata);
        await db().prepare('UPDATE jobs SET data=?,phase=?,status=?,lease=NULL,updated=? WHERE id=? AND lease=?').bind(JSON.stringify(data), phase, done ? 'completed' : 'queued', now(), job.id, lease).run();
        return json(publicJob(await one('jobs', job.id)));
      } catch (e) {
        const message = e instanceof Error ? e.message : 'Analysis failed. Retry this task.';
        await db().prepare("UPDATE jobs SET status='error',error=?,lease=NULL,updated=? WHERE id=? AND lease=?").bind(message, now(), job.id, lease).run();
        return json({ error: message, job_id: job.id }, 502);
      }
    }
    if (body.action === 'review') {
      const job = await one('jobs', required(body.job_id, 'task ID', 100)); if (job.status !== 'completed') throw new Error('Wait for this analysis to finish.');
      const original = parsed<any>(job.data, {}); const rationale = required(body.rationale, 'review rationale', 10000); const actor = required(body.reviewer, 'reviewer name', 200);
      let data;
      if (job.kind === 'extract' || job.kind === 'data') {
        validateAgainstSchema(body.extraction, job.kind === 'data' ? generalSchema : extractionSchema);
        const issues = job.kind === 'data' ? body.extraction.records.map((r:any) => r.missing_status==='not_reported' ? [] : evidenceIssues(r.evidence,original.input.blocks)) : validateExtraction(body.extraction, original.input.blocks).map((issues,i)=>[...issues,...statisticIssues(body.extraction.records[i]),...matrixValueIssues(body.extraction.records[i],original.input.blocks)]);
        if (!['approved', 'needs_revision'].includes(body.status)) throw new Error('Invalid review status.');
        if (body.status === 'approved' && issues.some((row:string[]) => row.some((s:string) => /Unknown source|Quotation does not match|No source evidence|Pair-specific n must|Reported r|missing coefficient|Reported coefficient|Sample size is not present|Duplicate construct|Statistic type conflicts|Pearson type is not established|Matrix cell value|HTMT is not/.test(s)))) throw new Error('Resolve source, value, statistic-type and duplicate errors before approving.');
        data = { status: body.status, extraction: body.extraction, issues, rationale };
      } else {
        if (!['include', 'exclude', 'uncertain'].includes(body.decision)) throw new Error('Choose an eligibility decision.');
        data = { decision: body.decision, rationale };
      }
      const id = crypto.randomUUID();
      await db().prepare('INSERT INTO decisions (id,study_id,job_id,kind,actor,data,created) VALUES (?,?,?,?,?,?,?)').bind(id, job.study_id, job.id, job.kind, actor, JSON.stringify(data), now()).run();
      return json({ id });
    }
    return json({ error: 'Unknown action.' }, 404);
  } catch (e) { return json({ error: e instanceof Error ? e.message : 'Request failed.' }, e instanceof AccessError ? e.status : 400); }
}
