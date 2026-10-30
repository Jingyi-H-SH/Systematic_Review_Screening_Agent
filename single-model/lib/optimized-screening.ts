import {ROLE_INSTRUCTIONS,screenSchema,type Screening,type Block} from './analysis';
import {screeningInput} from './screening';
import {modelCall} from './model';
export const OPTIMIZED_ROUTING='strong-single-stage-policy-v2';
export const STAGE_POLICY = "TITLE/ABSTRACT SCREENING POLICY\nYour decision concerns obtaining full text, not final inclusion. Preserve the final eligibility criteria below.\nFirst identify whether the actual research addresses the review's core population and intervention, exposure, test or phenomenon. Background mentions do not count. A paper wholly about another population, intervention or question is excluded. Missing details alone are not an exclusion.\nFor a genuinely close primary study, retain for full-text checking when the remaining decision depends on a detailed comparator, treatment-period definition, extractable subgroup, measurement or operational definition. Flag the specific potential conflict; do not pretend that the criterion is met or change the final protocol. This policy can retain a close study that will later fail full-text eligibility.\nFor diagnostic studies, distinguish completely unrelated tests from eligible tests assessing disease extent or surgical findings in the target clinical pathway. Check whether a target subgroup could be reported separately. For qualitative reviews, distinguish firsthand accounts of the target phenomenon from quantitative-only surveys or professionals merely describing other people's views. Do not require the abstract to name an analysis method when qualitative results are evident.\nDo not retain an unrelated paper merely because its abstract is absent. Use positive information in its title to identify a clear mismatch. If a title is plausibly in scope but key details are absent, mark uncertain.\nReturn concise original-source quotations and the exact eligibility issue. Keep evidence of compliance separate from the decision to obtain full text. Do not use the result's direction or statistical significance as eligibility criteria.";
export function optimizedScreeningInput(input:any,policy=STAGE_POLICY){
 const base=screeningInput(input);
 return {...base,criteria:policy+'\n\nORIGINAL FINAL ELIGIBILITY PROTOCOL (unchanged):\n'+base.criteria};
}
// Exact V2 evaluation guard: at least one original-source quote must support
// a failed criterion. Keep the model output separately from the guarded result.
export function optimizedGuard(result:Screening,blocks:Block[]){
 const normalize=(s:string)=>s.split(/\s+/).filter(Boolean).join(' ');
 const supported=result.criteria_checks.filter(c=>c.finding==='not_met').some(c=>c.evidence.some(e=>{
  const source=blocks.find(b=>b.id===e.block_id),quote=normalize(e.quotation);
  return !!source && !!quote && normalize(source.text).includes(quote);
 }));
 const review_required=result.decision==='exclude'&&!supported;
 return {result:review_required?{...structuredClone(result),decision:'uncertain' as const}:structuredClone(result),review_required};
}
export async function runOptimizedScreening(key:string,input:any,model:string,policy=STAGE_POLICY){
 const call=await modelCall(key,ROLE_INSTRUCTIONS[2],optimizedScreeningInput(input,policy),screenSchema,'medium',model);
 const raw=structuredClone(call.result) as Screening;
 const checked=optimizedGuard(raw,input.blocks);
 return {result:checked.result,raw,metadata:call.metadata,review_required:checked.review_required};
}
