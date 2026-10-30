import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {STAGE_POLICY,optimizedScreeningInput,optimizedGuard,runOptimizedScreening} from '../.test-build/optimized-screening.js';
import {COMMON,ROLE_INSTRUCTIONS} from '../.test-build/analysis.js';
import {MODEL_OPTIONS,SCREENING_MODEL,EXTRACTION_MODEL,selectedModel} from '../.test-build/model-options.js';
const input={report_id:'software-qa',title:'An intervention study',abstract:'Adults were interviewed.',doi:'',criteria:'Randomized adult studies.',constructs:'Do not use extraction rules',extraction_template:'Do not use extraction rules',stage:'screen',blocks:[{id:'title',text:'An intervention study',page:null,location:'Title'},{id:'abstract',text:'Adults were interviewed.',page:null,location:'Abstract'}]};
const result={decision:'exclude',explanation:'Design mismatch.',criteria_checks:[{criterion:'Randomized design',finding:'not_met',evidence:[{block_id:'abstract',quotation:'Adults were interviewed.'}]}]};
test('Published policy exactly matches the independently tested frozen policy',()=>{
 assert.equal(STAGE_POLICY,fs.readFileSync(new URL('./fixtures/stage-policy-v2.txt',import.meta.url),'utf8'));
 const prepared=optimizedScreeningInput(input);
 assert.equal(prepared.criteria,STAGE_POLICY+'\n\nORIGINAL FINAL ELIGIBILITY PROTOCOL (unchanged):\n'+input.criteria);
 assert.equal(prepared.constructs,'');assert.equal(prepared.extraction_template,'');assert.deepEqual(prepared.blocks,input.blocks);
 assert.equal(input.criteria,'Randomized adult studies.');
});
test('Optimized screening performs one call using the chosen model and benchmark instructions',async()=>{
 const fetchOriginal=globalThis.fetch;let calls=[];
 globalThis.fetch=async(url,init)=>{calls.push({url,headers:init.headers,payload:JSON.parse(init.body)});return Response.json({id:'qa-response',model:'gpt-6.1-sol',choices:[{finish_reason:'stop',message:{content:JSON.stringify(result)}}],usage:{prompt_tokens:10,completion_tokens:10}});};
 try{
  const value=await runOptimizedScreening('sk-software-test',input,SCREENING_MODEL);
  assert.equal(calls.length,1);const sent=calls[0].payload;
  assert.equal(sent.model,SCREENING_MODEL);assert.equal(sent.reasoning_effort,'medium');assert.equal(sent.store,false);
  assert.equal(sent.messages[0].content,COMMON+'\n'+ROLE_INSTRUCTIONS[2]);assert.deepEqual(JSON.parse(sent.messages[1].content),optimizedScreeningInput(input));
  assert.equal(value.metadata.requested_model,SCREENING_MODEL);assert.equal(value.result.decision,'exclude');assert.equal(value.review_required,false);
  assert.ok(!JSON.stringify(value).includes('sk-software-test'));
  await runOptimizedScreening('sk-software-test',input,EXTRACTION_MODEL);
  assert.equal(calls[1].payload.model,EXTRACTION_MODEL);
 }finally{globalThis.fetch=fetchOriginal;}
});
test('Unsupported exclusions become uncertain without mutating original model output',()=>{
 const raw=structuredClone(result);raw.criteria_checks[0].evidence[0].quotation='invented text';
 const checked=optimizedGuard(raw,input.blocks);
 assert.equal(checked.result.decision,'uncertain');assert.equal(checked.review_required,true);assert.equal(raw.decision,'exclude');
 assert.equal(optimizedGuard({...raw,criteria_checks:[]},input.blocks).result.decision,'uncertain');
 assert.equal(optimizedGuard({...raw,decision:'include'},input.blocks).result.decision,'include');
});
test('Model choices accept tested IDs, retain separate defaults and reject arbitrary providers',()=>{
 assert.equal(selectedModel(undefined),SCREENING_MODEL);assert.equal(selectedModel(undefined,EXTRACTION_MODEL),EXTRACTION_MODEL);
 for(const m of MODEL_OPTIONS)assert.equal(selectedModel(m.id),m.id);
 for(const m of ['https://untrusted.example/model',{},'unavailable-model'])assert.throws(()=>selectedModel(m),/supported model/);
});
