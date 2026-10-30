import test from 'node:test';
import assert from 'node:assert/strict';
import {modelCall} from '../.test-build/model.js';
import {screenSchema} from '../.test-build/analysis.js';
test('Model request sends source evidence with strict schema and never stores the key',async()=>{
 const original=globalThis.fetch;let sent;
 globalThis.fetch=async(url,init)=>{sent={url,init,body:JSON.parse(init.body)};return Response.json({id:'test-response',model:'test-model',choices:[{finish_reason:'stop',message:{content:JSON.stringify({decision:'uncertain',explanation:'Missing design detail',criteria_checks:[]})}}],usage:{total_tokens:10}});};
 try {
  const r=await modelCall('sk-test-not-a-real-key','Apply criteria',{blocks:[{id:'abstract',text:'Original abstract'}],criteria:'Original criteria'},screenSchema);
  assert.equal(sent.url,'https://api.openai.com/v1/chat/completions');assert.equal(sent.body.store,false);assert.equal(sent.body.response_format.json_schema.strict,true);
  assert.ok(sent.body.messages[1].content.includes('Original abstract'));assert.ok(!JSON.stringify(r).includes('sk-test'));assert.equal(r.result.decision,'uncertain');
 } finally {globalThis.fetch=original;}
});
test('Model HTTP errors and truncated output produce actionable failures',async()=>{
 const original=globalThis.fetch;
 try {
  for(const code of [401,403,429,500]){globalThis.fetch=async()=>new Response('sensitive upstream body',{status:code});await assert.rejects(()=>modelCall('sk-test','',{},screenSchema),e=>!e.message.includes('sensitive'));}
  globalThis.fetch=async()=>Response.json({choices:[{finish_reason:'length',message:{content:'{}'}}]});
  await assert.rejects(()=>modelCall('sk-test','',{},screenSchema),/complete analysis/);
 }finally{globalThis.fetch=original;}
});
