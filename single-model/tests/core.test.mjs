import test from 'node:test';
import assert from 'node:assert/strict';
import {readCSV,readRIS,parseDocument} from '../.test-build/importers.js';
import {evidenceIssues,validateExtraction,validateAgainstSchema,screenSchema,extractionSchema,chunks,extractionContext,combineGeneral,csv} from '../.test-build/analysis.js';

test('CSV preserves quoted multiline abstracts and escaped quotation marks',()=>{
 const rows=readCSV('\uFEFFtitle,abstract,doi\r\n"A, B","First line\nA ""quote""",10.1/a');
 assert.deepEqual(rows,[{title:'A, B',abstract:'First line\nA "quote"',doi:'10.1/a'}]);
 assert.throws(()=>readCSV('title\n"incomplete'));
});
test('RIS retains continuation lines and multiple records',()=>{
 assert.deepEqual(readRIS('TY  - JOUR\nTI  - First\nAB  - One\n  two\nDO  - 10.1/a\nER  -\nTY  - JOUR\nTI  - Second\nER  -'),[{title:'First',abstract:'One two',doi:'10.1/a'},{title:'Second',abstract:'',doi:''}]);
});
test('Evidence rejects unknown or fabricated quotations',()=>{
 const blocks=[{id:'p1',text:'A reported value is 0.24.',page:1,location:'Page 1'}];
 assert.deepEqual(evidenceIssues([{block_id:'p1',quotation:'value is 0.24.'}],blocks),[]);
 assert.match(evidenceIssues([{block_id:'p1',quotation:'value is 0.42.'}],blocks)[0],/does not match/);
 assert.match(evidenceIssues([{block_id:'p2',quotation:'0.24'}],blocks)[0],/Unknown/);
});
const blocks=[{id:'p1',text:'N = 100; r = -0.233.',page:1,location:'Table 4'}];
const row={study_id:'s1',sample_id:'main',overlap_group:'checked s1',construct_x:'negative image',construct_y:'satisfaction',measure_x:'DI',measure_y:'TS',direction_x:'higher = negative',direction_y:'higher = satisfied',timepoint:'baseline',subgroup:'all',correlation_type:'pearson_zero_order',r:-.233,n:100,missing_status:'reported',table:'4',row:'DI',column:'TS',evidence:[{block_id:'p1',quotation:'N = 100; r = -0.233.'}],notes:''};
test('MASEM preserves negative r and flags null N, beta, reversed duplicate pairs',()=>{
 assert.deepEqual(validateExtraction({records:[row],notes:''},blocks),[[]]);
 const issues=validateExtraction({records:[row,{...row,construct_x:row.construct_y,construct_y:row.construct_x,n:null,correlation_type:'standardized_beta'}],notes:''},blocks)[1];
 assert.ok(!issues.some(x=>x.includes('Duplicate')));assert.ok(issues.some(x=>x.includes('sample size')));assert.ok(issues.some(x=>x.includes('Pearson')));
 assert.ok(validateExtraction({records:[row,{...row,construct_x:row.construct_y,construct_y:row.construct_x}],notes:''},blocks)[1].some(x=>x.includes('Duplicate')));
 assert.ok(validateExtraction({records:[{...row,r:0,missing_status:'not_reported'}],notes:''},blocks)[0].some(x=>x.includes('null')));
});
test('Strict response schema rejects missing, extra and mistyped fields',()=>{
 validateAgainstSchema({records:[row],notes:''},extractionSchema);
 assert.throws(()=>validateAgainstSchema({decision:'include',criteria_checks:[]},screenSchema));
 assert.throws(()=>validateAgainstSchema({decision:'include',criteria_checks:[],explanation:'ok',invented:true},screenSchema));
 assert.throws(()=>validateAgainstSchema({records:[{...row,n:'100'}],notes:''},extractionSchema));
});
test('A real quotation does not validate a different extracted number',()=>{
 const issues=validateExtraction({records:[{...row,r:-.199,n:150}],notes:''},blocks)[0];
 assert.ok(issues.some(x=>x.includes('coefficient is not present')));assert.ok(issues.some(x=>x.includes('Sample size is not present')));
});
test('Supporting context preserves original IDs, excludes focus and stays bounded',()=>{
 const b=[{id:'methods',text:'Final sample: 100 participants.',page:2,location:'Methods'},...blocks];
 const context=extractionContext(b,blocks,100);
 assert.deepEqual(context,[b[0]]);assert.deepEqual(extractionContext(b,blocks,5),[]);
});
test('Chunking keeps source identifiers and respects the context limit',()=>{
 const b=Array.from({length:5},(_,i)=>({id:String(i),text:'x'.repeat(6000),page:i+1,location:'page'}));
 assert.equal(chunks(b).length,2);assert.deepEqual(chunks(b).flat(),b);
 assert.throws(()=>chunks([{...b[0],text:'x'.repeat(24001)}]));
});
test('CSV export defuses formulas while retaining negative coefficients',()=>{
 const output=csv([{title:'=HYPERLINK("bad")',r:-.233}],['title','r']);
 assert.ok(output.includes("'=HYPERLINK"));assert.ok(output.includes('"-0.233"'));
});
test('Text parser preserves original words and never invents physical pages',async()=>{
 const output=await parseDocument(new File(['A study reports N=100.\nIts result is r=0.24.'],'study.txt'));
 assert.equal(output.blocks[0].page,null);assert.match(output.blocks[0].text,/N=100/);assert.ok(output.warnings.length);
});
test('Chunk reconciliation removes false missingness while preserving conflicting reported values',()=>{
 const r=(value,status='reported',id='p1')=>({field:'Sample size',value,missing_status:status,evidence:[{block_id:id,quotation:value}],notes:''});
 const original=[{records:[r('100'),r('Not reported','not_reported')],notes:''},{records:[r('100','reported','p2'),r('98')],notes:''}];
 const result=combineGeneral(original);
 assert.deepEqual(result.records.map(r=>r.value),['100','98']);assert.equal(result.records[0].evidence.length,2);
 assert.equal(original[0].records[0].evidence.length,1);
});
