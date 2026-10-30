import type {Block,Correlation,Extraction} from './analysis';
import {explicitConstructAliases} from './matrix-extraction';
import {matrixCandidates} from './matrices';
const norm=(s:unknown)=>String(s??'').normalize('NFKC').toLowerCase().replace(/[()_:]/g,' ').replace(/\s+/g,' ').trim();
const escape=(s:string)=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
export function constructScope(dictionary:string,blocks:Block[]){
 if(/^(?:not specified|none|unspecified)$/i.test(dictionary.trim()))dictionary='';
 const map=new Map<string,string>();
 for(const [alias,name] of explicitConstructAliases(dictionary)){map.set(norm(alias),name);map.set(norm(name),name);}
 for(const m of dictionary.matchAll(/([A-Za-z][A-Za-z /-]*?)\s*\(([^()]*)\)/g)){
  const name=m[1].trim().replace(/^and\s+/i,'');map.set(norm(name),name);
  for(const a of m[2].split(';').map(x=>x.trim()))if(a&&!a.includes('/'))map.set(norm(a),name);
 }
 // Plain semicolon/newline lists are also accepted; prose instructions are not targets.
 if(!map.size)for(const name of dictionary.split(/[;\n]/).map(x=>x.trim()))if(name&&name.length<100&&!/[.!?]/.test(name))map.set(norm(name),name);
 for(const [name,target] of [...map]){
  if(name.length<4)continue;
  const p=escape(name).replace(/\s+/g,'\\s+');
  for(const block of blocks)for(const re of [new RegExp(`\\b([A-Z][A-Z0-9_-]{1,14})\\s*=\\s*${p}\\b`,'gi'),new RegExp(`\\b${p}\\s*\\(([A-Z][A-Z0-9_-]{1,14})\\)`,'gi')])for(const m of block.text.matchAll(re))map.set(norm(m[1]),target);
 }
 const resolve=(...labels:string[])=>{
  for(const label of labels){const n=norm(label);if(map.has(n))return map.get(n)!;
   for(const a of label?.matchAll(/\(([A-Z][A-Z0-9_-]{1,14})\)/g)||[])if(map.has(norm(a[1])))return map.get(norm(a[1]))!;
   const qualified=norm(label.replace(/\([^)]*\)/g,'')).replace(/^(?:negative|cognitive|affective|conative|overall)\s+/,'').replace(/\s+(?:items|scale|score)$/,'');
   if(map.has(qualified))return map.get(qualified)!;
  }return null;
 };
 return {resolve,configured:map.size>0};
}
function tableId(s:string){return String(s||'').match(/\btable\s*(\d+[a-z]?)\b/i)?.[1]?.toLowerCase()||(/^\d+[a-z]?$/i.test(String(s||'').trim())?String(s).trim().toLowerCase():'');}
function interaction(r:Correlation,blocks:Block[]){
 let text=[r.table,r.row,r.measure_x,r.measure_y].join(' ');
 const id=tableId(r.table);if(id&&/^\d/.test(id))for(const b of blocks)text+=' '+(b.text.match(new RegExp(`Table\\s+${escape(id)}\\.[^\\n]*`,'i'))?.[0]||'');
 return /moderat|interaction|indirect effect|mediated effect|[×*]/i.test(text);
}
const unknown=(s:string)=>/^(?:|null|none|not specified|unspecified|cross-sectional|cross sectional|timepoint unspecified)$/i.test(norm(s));
function sameContext(a:Correlation,b:Correlation){
 if(tableId(a.table)&&tableId(b.table)&&tableId(a.table)!==tableId(b.table))return false;
 const model=(s:string)=>s.match(/\bmodel\s*[-_:]?\s*([a-z1-9])\b/i)?.[0]?.toLowerCase();
 if(a.correlation_type==='standardized_beta'&&model(a.row)&&model(b.row)&&model(a.row)!==model(b.row))return false;
 if(a.n!==null&&b.n!==null&&a.n!==b.n)return false;
 if(!unknown(a.timepoint)&&!unknown(b.timepoint)&&norm(a.timepoint)!==norm(b.timepoint))return false;
 if(!unknown(a.subgroup)&&!unknown(b.subgroup)&&norm(a.subgroup)!==norm(b.subgroup))return false;
 const explicit=(s:string)=>s.match(/\b(?:sample|study|cohort|group)\s*[-_:]?\s*([a-d1-9])\b/i)?.[0]?.toLowerCase();
 if(explicit(a.sample_id)&&explicit(b.sample_id)&&explicit(a.sample_id)!==explicit(b.sample_id))return false;
 return true;
}
export function curateExtraction(outputs:Extraction[],blocks:Block[],dictionary:string,reportId?:string){
 const scope=constructScope(dictionary,blocks),all=outputs.flatMap(x=>x.records);
 const matrices=matrixCandidates(blocks);
 const records:Correlation[]=[],keys:string[]=[],sourceIndices:number[][]=[],excluded:any[]=[];
 for(let i=0;i<all.length;i++){
  const r=structuredClone(all[i]),reasons:string[]=[];
  let verifiedMatrixCell=false;
  // Article identity comes from the uploaded report, never a model-generated page/author alias.
  if(reportId)r.study_id=reportId;
  const x=scope.resolve(r.construct_x,r.measure_x),y=scope.resolve(r.construct_y,r.measure_y);
  if(scope.configured&&(!x||!y))reasons.push('Outside requested constructs or unresolved construct mapping.');
  if(r.r===null||r.missing_status!=='reported')reasons.push('No reported numeric association in this candidate.');
  if(interaction(r,blocks))reasons.push('Interaction or indirect effect, not a direct association between the requested pair.');
  if(tableId(r.table)&&r.correlation_type!=='standardized_beta'&&/basis: correlation_matrix/.test(r.notes||'')){
   const expected=matrices.filter(m=>m.kind!=='htmt'&&tableId(m.caption)===tableId(r.table)).flatMap(m=>m.cells.filter((c:any)=>{
    const a=scope.resolve(c.row),b=scope.resolve(c.column);return a&&b&&((a===x&&b===y)||(a===y&&b===x));
   }));
   if(expected.length===1){if(r.r!==expected[0].value)reasons.push('Value does not match the requested pair in the cited correlation table.');else verifiedMatrixCell=true;}
  }
  if(reasons.length){excluded.push({source_index:i,reasons,record:r});continue;}
  const dimension=(s:string)=>(s.match(/\b(?:negative|cognitive|affective|conative)\b/gi)||[]).map(s=>s.toLowerCase()).sort().join(':');
  const pair=[norm(x||r.construct_x)+':'+dimension(r.construct_x+' '+r.measure_x),norm(y||r.construct_y)+':'+dimension(r.construct_y+' '+r.measure_y)].sort();
  // A rounded prose report can repeat a unique, more precise source matrix cell.
  // Require a known shared sample size and no explicit competing table/context.
  if(!tableId(r.table)&&r.correlation_type==='pearson_zero_order'&&r.r!==null&&r.n!==null&&/basis: correlation_matrix/.test(r.notes||'')){
   const decimals=String(r.r).split('.')[1]?.length||0;
   const unique=matrices.flatMap(m=>m.cells.filter((c:any)=>{const a=scope.resolve(c.row),b=scope.resolve(c.column);return a&&b&&((a===x&&b===y)||(a===y&&b===x));}));
   const matches=records.map((a,j)=>({a,j})).filter(({a})=>a.correlation_type==='pearson_zero_order'&&a.n===r.n&&sameContext(a,r)&&norm(a.timepoint)===norm(r.timepoint)&&norm(a.subgroup)===norm(r.subgroup)&&a.study_id===r.study_id&&/Coefficient copied from the original/.test(a.notes)&&a.r!==null&&[scope.resolve(a.construct_x,a.measure_x),scope.resolve(a.construct_y,a.measure_y)].sort().join('|')===[x,y].sort().join('|')&&decimals>=2&&String(a.r).split('.')[1]?.length>decimals&&Number(a.r.toFixed(decimals))===r.r);
   if(unique.length===1&&matches.length===1){const {a,j}=matches[0];for(const e of r.evidence)if(!a.evidence.some(f=>f.block_id===e.block_id&&f.quotation===e.quotation))a.evidence.push(e);a.notes+=' A less precise prose repetition is linked in the audit; the original matrix value is retained.';sourceIndices[j].push(i);continue;}
  }
  const typeKey=verifiedMatrixCell&&['other','unclear','pearson_zero_order'].includes(r.correlation_type)?'same-source-correlation-cell':r.correlation_type;
  const key=JSON.stringify([r.study_id,typeKey,r.r,pair]);
  const index=records.findIndex((a,j)=>keys[j]===key&&sameContext(a,r)&&((tableId(a.table)&&tableId(a.table)===tableId(r.table))||a.evidence.some(e=>r.evidence.some(f=>e.block_id===f.block_id))));
  if(index>=0){const a=records[index],seen=new Set(a.evidence.map(e=>JSON.stringify(e)));for(const e of r.evidence)if(!seen.has(JSON.stringify(e))){a.evidence.push(e);seen.add(JSON.stringify(e));}if(a.correlation_type!==r.correlation_type){a.correlation_type='other';const note='Candidate labels disagree for the same verified matrix cell; correlation subtype requires review. Original labels remain in the audit.';if(!a.notes.includes(note))a.notes+=' '+note;}sourceIndices[index].push(i);}
  else{records.push(r);keys.push(key);sourceIndices.push([i]);}
 }
 return {result:{records,notes:outputs.map(o=>o.notes).filter(Boolean).join('\n')},curation:{input_count:all.length,retained_count:records.length,excluded,merged:sourceIndices.filter(x=>x.length>1).map(x=>({source_indices:x,reason:'Same source association: exact compatible cell, or a uniquely matched rounded prose repetition with known sample size. Conflicting subtype labels remain unresolved.'}))}};
}
