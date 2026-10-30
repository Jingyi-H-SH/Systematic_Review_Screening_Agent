import {evidenceIssues,type Screening,type Block} from './analysis';
export const SCREENING_ROUTING='three-reviewers-selective-adjudication-v1';
export const ADJUDICATION='Adjudicate the three reviewer judgments using the ORIGINAL criteria and source blocks supplied here. Resolve disagreements by checking source evidence. Explicit exclusion evidence supports exclude; unresolved eligibility is uncertain. Include only when the available evidence supports eligibility for this stage. Return your own concise judgment with criterion-level evidence.';
export function supportedExclusion(result:Screening,blocks:Block[]){
 return result.decision==='exclude'&&result.criteria_checks.some(c=>c.finding==='not_met'&&c.evidence.length>0&&evidenceIssues(c.evidence,blocks).length===0);
}
export function consensusResult(outputs:Screening[],blocks:Block[]):Screening|null{
 if(outputs.length!==3||new Set(outputs.map(r=>r.decision)).size!==1)return null;
 return structuredClone(outputs[2]);
}
export function checkedScreeningResult(result:Screening,blocks:Block[]){
 if(result.decision!=='exclude'||supportedExclusion(result,blocks))return {result,review_required:false};
 return {result:structuredClone(result),review_required:true};
}
// Screening uses eligibility evidence only; extraction settings are not screening requirements.
export function screeningInput(input:any){
 const {report_id,title,abstract,doi,criteria,stage,blocks}=input;
 return {report_id,title,abstract,doi,criteria,constructs:'',extraction_template:'',stage,blocks};
}
