import { type Block, type Evidence, generalSchema, extractionSchema } from './analysis';
export type Unit = Block & {parent_id:string; start:number; end:number};
export const GENERAL_LINKED_INSTRUCTIONS = `Extract the current study's information for the supplied systematic-review extraction template. For each requested field, read all supplied units and return the actual value when stated. Keep the template field names exactly. Distinguish the study design from the statistical analysis method. Preserve sample sizes, setting, funding organizations and grant numbers, and stated study limitations. Return source_ids from the supplied list; the program retrieves their exact text, so do not write quotations. A source_id must directly support the extracted field value. Use not_reported with an empty source_ids array only when the information is absent from this part. A later part may supply it. Do not turn a stated value into not_reported. These are general study fields, not a restriction to correlation pairs.`;

/** Exact source slices, not generated quotations. The same map resolves model references. */
export function sourceUnits(blocks:Block[]):Unit[]{
  return blocks.flatMap(b=>{
    const units:Unit[]=[];let start=0;
    while(start<b.text.length){
      let end=Math.min(start+600,b.text.length);
      if(end<b.text.length){const newline=b.text.lastIndexOf('\n',end);if(newline>start+100)end=newline+1;}
      if(b.text.slice(start,end).trim())units.push({...b,id:`${b.id}@${start}`,parent_id:b.id,start,end,text:b.text.slice(start,end)});
      start=end;
    }return units;
  });
}
export function linkedSchema(kind:'data'|'extract',sourceIds?:string[]){
  const schema=structuredClone(kind==='data'?generalSchema:extractionSchema) as any;
  const record=schema.properties.records.items;
  delete record.properties.evidence;
  record.properties.source_ids={type:'array',items:{type:'string',...(sourceIds?.length?{enum:[...new Set(sourceIds)]}:{})}};
  if(kind==='extract'){
    record.properties.statistic_basis={type:'string',enum:['correlation_matrix','regression_or_path','narrative','unresolved']};
    record.properties.statistic_label={type:'string'};
  }
  record.required=Object.keys(record.properties);return schema;
}
export const LINKED_INSTRUCTIONS=`For evidence, return source_ids from the supplied units instead of writing quotations. Choose only the units that support that specific field or statistic, including row/column headers and variable definitions. The program will retrieve their exact original text. Do not cite unrelated literature-review statements as findings of the current study. Create records from the focus blocks only; context blocks support sample size, definitions and statistic type but must not generate repeated records from another batch. For MASEM, first establish the table's statistic type and column order. An off-diagonal Fornell-Larcker entry is an inter-construct correlation; its diagonal is square-root AVE. HTMT, AVE and loadings are not correlations. A regression or path coefficient is never Pearson r. Use standardized_beta ONLY when the selected coefficient is explicitly standardized. An unstandardized B, indirect effect or an ambiguously labeled coefficient uses other or unclear, preserving its original statistic label. Populate statistic_basis and statistic_label from the actual table or prose. Do not claim Pearson unless the source establishes Pearson; use other for an unspecified inter-construct correlation. Retain the original sign. Extract only requested construct pairs with actual numeric results; do not create rows for hypotheses or general significance statements. Use the table caption, complete header, row and variable-definition units together. Missing pair-specific n is null, not a reason to drop a correlation. Identify sample size separately from coefficient extraction. For general extraction, keep the exact field names from extraction_template, and do not treat a field missing in one part as absent from the whole study.`;
export function resolveLinked(raw:any,units:Unit[]){
  const map=new Map(units.map(u=>[u.id,u]));
  const provenance:any[]=[];
  const records=raw.records.map((r:any)=>{
    const {source_ids,statistic_basis,statistic_label,...value}=r;
    const ids=[...new Set<string>(source_ids)];
    const evidence:Evidence[]=ids.map(id=>{
      const u=map.get(id);
      return u?{block_id:u.parent_id,quotation:u.text}:{block_id:`unresolved:${id}`,quotation:''};
    });
    provenance.push({source_ids:ids,statistic_basis,statistic_label});
    return {...value,evidence,notes:[value.notes,statistic_basis?`Source statistic: ${statistic_label||'not specified'}; basis: ${statistic_basis}.`:''].filter(Boolean).join(' ')};
  });
  return {result:{records,notes:raw.notes},provenance};
}

/** Check declared statistic type against the supplied evidence; never silently change it. */
export function statisticIssues(r:any):string[]{
  const notes=String(r.notes||'');const errors:string[]=[];
  if(/basis: regression_or_path/.test(notes)&&!['standardized_beta','other','unclear'].includes(r.correlation_type))errors.push('Statistic type conflicts with a regression/path source.');
  if(/basis: correlation_matrix/.test(notes)&&r.correlation_type==='standardized_beta')errors.push('Statistic type conflicts with a correlation matrix source.');
  const text=(r.evidence||[]).map((e:Evidence)=>e.quotation).join(' ');
  if(r.correlation_type==='pearson_zero_order'&&/basis:/.test(notes)&&!/pearson/i.test(text))errors.push('Pearson type is not established by the cited source.');
  return errors;
}
