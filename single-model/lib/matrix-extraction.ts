import type {Block,Correlation} from './analysis';
import {sourceUnits,type Unit} from './evidence-extraction';
const obj=(properties:any)=>({type:'object',properties,required:Object.keys(properties),additionalProperties:false});
const str={type:'string'};
export const statisticEvidenceUnits=(units:Unit[])=>units.filter(u=>/Pearson|Spearman|partial correlation|bivariate|zero.order|unadjusted correlation|adjusted correlation/i.test(u.text));
export function matrixMappingSchema(matrices:any[],units:Unit[],typed=false){
  return obj({matrices:obj(Object.fromEntries(matrices.map((m,i)=>[`matrix_${i}`,obj({
    columns:obj(Object.fromEntries(m.headers.map((h:string)=>[h,obj({requested:{type:'boolean'},construct:str,direction:str})]))),
    interpretation:{type:'string',enum:['inter_construct_correlations','squared_correlations','htmt','unresolved']},
    ...(typed?{statistic:obj({method:{type:'string',enum:['pearson','spearman','unspecified']},adjustment:{type:'string',enum:['unadjusted','adjusted','unresolved']},conflicting_context:{type:'boolean'},source_ids:{type:'array',items:{type:'string',enum:(statisticEvidenceUnits(units).length?statisticEvidenceUnits(units):units).map(u=>u.id)}},rationale:str})}:{}),
    sample_label:str,n:{type:['number','null']},n_source_ids:{type:'array',items:{type:'string',enum:units.map(u=>u.id)}},notes:str,
  })]))),notes:str});
}
export const MATRIX_MAPPING_INSTRUCTIONS=`Interpret the supplied matrices and the requested construct dictionary. Return a decision for EVERY column, but do not transcribe any coefficients. Set requested=true ONLY if the column is one of the requested constructs or an explicitly allowed dimension. Unrelated constructs are false even if associated with a requested construct. Keep original measure direction; a negative image measure stays negative image. Identify whether the off-diagonal cells are correlations, squared correlations, HTMT or unresolved, using table notes. Fornell-Larcker off-diagonal cells are inter-construct correlations, while its diagonal is square-root AVE. The program will read matching cells directly. Sample size n must be supported as applying to these associations; otherwise null. Provide n_source_ids only when n is known. Do not generate missing or hypothetical associations. Use the study's own results, not cited prior research. When a statistic object is requested, link methods and results to THIS table and its constructs. Cite the methods plus any explicit results-to-table statement in statistic.source_ids. A generic Pearson statement alone does not resolve a competing partial-correlation description. Use adjustment=unresolved and conflicting_context=true when the same constructs have competing descriptions and this table does not settle it. Separate adjusted regressions in other analyses from this table's bivariate correlations. Prefer local table-footnote abbreviation definitions over reused abbreviations elsewhere. Never call a regression coefficient a Pearson correlation.`;
export function explicitConstructAliases(dictionary:string){
  const aliases=new Map<string,string>();
  for(const match of dictionary.matchAll(/([A-Za-z][A-Za-z /-]*?)\s*\(([^()]*)\)/g)){
    const name=match[1].trim().replace(/^and\s+/i,'');
    for(const token of match[2].split(/[;,]/).map(x=>x.trim()))if(/^[A-Z][A-Z0-9_-]{1,14}$/.test(token))aliases.set(token,name);
  }
  return aliases;
}
export function mappedStatistic(mapping:any,units:Unit[],matrix:any){
 const stat=mapping.statistic;
 const sources=(stat?.source_ids||[]).map((id:string)=>units.find(u=>u.id===id)).filter(Boolean) as Unit[];
 const text=sources.map(u=>u.text).join(' ');
 if(!stat)return {type:'other',sources,description:'Correlation subtype is not established by the saved mapping.'};
 const explicitTable=String(matrix.caption).match(/table\s+(\d+)/i)?.[1];
 const directUnadjusted=!!explicitTable&&sources.some(u=>new RegExp(`(?:bivariate|zero.order|unadjusted)[^.!?]{0,200}Table\\s+${explicitTable}\\b|Table\\s+${explicitTable}[^.!?]{0,200}(?:bivariate|zero.order|unadjusted)`,'i').test(u.text));
 const partialInContext=units.some(u=>/partial correlation/i.test(u.text));
 let type='unclear';
 // A model's conflict flag needs source support; an unsupported boolean cannot erase explicit type evidence.
 const methodConflict=(stat.method==='pearson'&&/Spearman/i.test(text))||(stat.method==='spearman'&&/Pearson/i.test(text));
 if(!methodConflict&&sources.length){
  if(stat.adjustment==='adjusted'&&/partial correlation/i.test(text))type='partial';
  else if(stat.adjustment==='unadjusted'&&(!partialInContext||directUnadjusted)){
   if(stat.method==='pearson'&&/Pearson/i.test(text))type='pearson_zero_order';
   else if(stat.method==='spearman'&&/Spearman/i.test(text))type='spearman';
  }
 }
 const unresolved=type==='unclear';
 const linkedSources=unresolved?[...new Map([...sources,...units.filter(u=>/Pearson|Spearman|partial correlation/i.test(u.text))].map(u=>[u.id,u])).values()]:sources;
 return {type,sources:linkedSources,description:`Statistic classification: ${type}; adjustment: ${unresolved?'unresolved':stat.adjustment}. ${unresolved?(partialInContext?'The supplied methods also describe partial correlations; the table-specific evidence does not resolve which adjustment applies.':'The cited evidence does not establish a table-specific correlation subtype.'):'The classification is linked to the cited methods and results.'}`};
}
export function extractMappedMatrices(raw:any,matrices:any[],units:Unit[],blocks:Block[],reportId:string,dictionary=''){
  const aliases=explicitConstructAliases(dictionary);
  const userAliases=Object.fromEntries(aliases);
  const names=[...new Set(aliases.values())].map(s=>s.toLowerCase());
  for(const group of dictionary.matchAll(/\(([^()]*)\)/g))for(const token of group[1].split(';').map(s=>s.trim()))if(token.includes(' ')&&!token.includes('/'))names.push(token.toLowerCase());
  const sourceAliases:any[]=[];
  for(const block of blocks)for(const name of names){
    const escaped=name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&').replace(/\s+/g,'\\s+');
    for(const pattern of [new RegExp(`\\b([A-Z][A-Z0-9_-]{1,14})\\s*=\\s*${escaped}\\b`,'gi'),new RegExp(`\\b${escaped}\\s*\\(([A-Z][A-Z0-9_-]{1,14})\\)`,'gi')]){
      for(const match of block.text.matchAll(pattern))if(!aliases.has(match[1])){aliases.set(match[1],name);sourceAliases.push({alias:match[1],construct:name,block_id:block.id,text:match[0]});}
    }
  }
  const records:Correlation[]=[];const notes:string[]=[];const provenance:any[]=[];
  for(let i=0;i<matrices.length;i++){
    const m=matrices[i],mapping=raw.matrices[`matrix_${i}`];
    // Known non-correlation matrices cannot be relabelled by the model.
    if(m.kind==='htmt'||(!String(m.layout).startsWith('mixed_')&&mapping.interpretation!=='inter_construct_correlations')){
      notes.push(`${m.caption||m.source_block}: ${mapping.interpretation}; no correlation records generated.`);continue;
    }
    const block=blocks.find(b=>b.id===m.source_block)!;
    const statistic=mappedStatistic(mapping,units,m);
    for(const cell of m.cells){
      // Explicit user-defined aliases take precedence over a model's scope classification.
      const mapped=(label:string)=>{
        const known=aliases.get(label)||names.find(name=>name===label.toLowerCase());
        if(known)return {...mapping.columns[label],requested:true,construct:known};
        const value=mapping.columns[label];
        // When the user supplies a named dictionary, require a named target for nonliteral mappings too.
        const inScope=!names.length||names.some(name=>String(value.construct).toLowerCase().includes(name));
        return {...value,requested:value.requested&&inScope};
      };
      const a=mapped(cell.row),b=mapped(cell.column);if(!a.requested||!b.requested)continue;
      const lines=[m.rows.find((r:any)=>r.label===cell.row)?.source_line,m.rows.find((r:any)=>r.label===cell.column)?.source_line,m.source_header||block.text.split('\n').find(line=>m.headers.every((h:string)=>line.includes(h))),m.legend_text].filter(Boolean);
      const ranges=lines.map(line=>({start:block.text.indexOf(line),end:block.text.indexOf(line)+line.length}));
      const selected=sourceUnits([block]).filter(u=>ranges.some(range=>u.start<range.end&&u.end>range.start));
      // Rows can cross unit boundaries. A full original block is an explicit fallback, never generated prose.
      const evidence=selected.length?selected.map(u=>({block_id:block.id,quotation:u.text})):[{block_id:block.id,quotation:block.text}];
      for(const id of [...mapping.n_source_ids,...statistic.sources.map(u=>u.id)]){const u=units.find(u=>u.id===id);if(u&&!evidence.some(e=>e.quotation===u.text))evidence.push({block_id:u.parent_id,quotation:u.text});}
      records.push({study_id:reportId,sample_id:`${reportId}:${mapping.sample_label||'sample-unspecified'}`,overlap_group:'',construct_x:a.construct||cell.row,construct_y:b.construct||cell.column,measure_x:cell.row,measure_y:cell.column,direction_x:a.direction,direction_y:b.direction,timepoint:'',subgroup:'',correlation_type:statistic.type as Correlation['correlation_type'],r:cell.value,n:mapping.n,missing_status:'reported',table:m.caption,row:cell.row,column:cell.column,evidence,notes:`Coefficient copied from the original ${m.layout} matrix cell; source basis: correlation_matrix. ${statistic.description} ${mapping.notes}`});
      provenance.push({method:'deterministic-matrix-cell-v1',source_block:block.id,row:cell.row,column:cell.column,value:cell.value,mapping,explicit_user_aliases:userAliases,source_defined_aliases:sourceAliases});
    }
  }
  return {result:{records,notes:[raw.notes,...notes].filter(Boolean).join('\n')},provenance};
}
