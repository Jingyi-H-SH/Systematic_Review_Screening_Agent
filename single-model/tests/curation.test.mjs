import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {matrixCandidates,triangleMeaning} from '../.test-build/matrices.js';
import {curateExtraction} from '../.test-build/masem-curation.js';
import {sourceUnits} from '../.test-build/evidence-extraction.js';
import {extractMappedMatrices} from '../.test-build/matrix-extraction.js';
const blocks=JSON.parse(fs.readFileSync(new URL('./fixtures/mixed-published-table.json',import.meta.url)));
test('Published mixed r/r2 table: wrapped footnotes and descriptive columns preserve original upper r',()=>{
 const m=matrixCandidates(blocks);assert.equal(m.length,1);assert.equal(m[0].layout,'mixed_upper_r');assert.equal(m[0].cells.length,21);
 assert.equal(m[0].cells.find(c=>c.row==='Escapism'&&c.column==='Tour satisfaction').value,.730);
 assert.equal(m[0].cells.some(c=>c.row===c.column),false);
});
test('Mixed table uses its explicit legend even when model calls it squared; evidence retains that legend',()=>{
 const m=matrixCandidates(blocks),units=sourceUnits(blocks);
 const raw={matrices:{matrix_0:{columns:Object.fromEntries(m[0].headers.map(h=>[h,{requested:['Escapism','Tour satisfaction'].includes(h),construct:h,direction:'higher'}])),interpretation:'squared_correlations',n:null,n_source_ids:[],sample_label:'main',notes:''}},notes:''};
 const out=extractMappedMatrices(raw,m,units,blocks,'s','Escapism (ESC), tourist satisfaction (TS; tour satisfaction).');
 assert.equal(out.result.records.length,1);assert.equal(out.result.records[0].r,.73);
 assert.match(out.result.records[0].evidence.map(e=>e.quotation).join(' '),/Squared correlations are below/);
});
test('Reverse orientation selects lower r and absence of a mixed legend does not infer square roots',()=>{
 const text='Table 2. Correlations\nA | B | C\nA | 1 | .25 | .36\nB | -.5 | 1 | .49\nC | .6 | .7 | 1\nCorrelations are below the diagonal; squared correlations are above the diagonal.';
 const b=[{id:'p',page:1,location:'1',text}];assert.deepEqual(matrixCandidates(b)[0].cells.map(c=>c.value),[-.5,.6,.7]);
 assert.equal(matrixCandidates([{...b[0],text:text.split('\nCorrelations')[0]}]).length,0);
 assert.equal(triangleMeaning('Correlations are above the diagonal. Correlations are below the diagonal.'),null);
});
const b=[{id:'p',page:1,location:'1',text:'Table 1. Correlations\nA B .4\nTable 2. Direct effects\nTable 3. Moderation effects'}];
const base={study_id:'s',sample_id:'s:main',overlap_group:'',construct_x:'Exposure',construct_y:'Outcome',measure_x:'A',measure_y:'B',direction_x:'higher',direction_y:'higher',timepoint:'',subgroup:'',correlation_type:'pearson_zero_order',r:.4,n:100,missing_status:'reported',table:'Table 1',row:'A',column:'B',evidence:[{block_id:'p',quotation:'A B .4'}],notes:''};
const run=(rows,dict='Exposure (A); Outcome (B)')=>curateExtraction([{records:rows,notes:''}],b,dict);
test('Repeated same-cell candidates merge and retain their original input indices',()=>{
 const out=run([base,{...base,construct_x:'Outcome',construct_y:'Exposure',measure_x:'B',measure_y:'A'}]);
 assert.equal(out.result.records.length,1);assert.deepEqual(out.curation.merged[0].source_indices,[0,1]);
});
test('Different samples, timepoints, subgroups, tables and statistic types are not collapsed',()=>{
 for(const patch of [{n:101},{sample_id:'s:cohort 2'},{timepoint:'follow-up'},{subgroup:'women'},{table:'Table 2'},{correlation_type:'standardized_beta'}]){
  const a={...base,sample_id:'s:cohort 1',timepoint:'baseline',subgroup:'men'};
  assert.equal(run([a,{...a,...patch}]).result.records.length,2,JSON.stringify(patch));
 }
});
test('Out-of-scope, interaction and absent-value candidates are set aside without changing raw input',()=>{
 const rows=[base,{...base,construct_y:'Enjoyment',measure_y:'Enjoyment'},{...base,table:'Table 3',correlation_type:'standardized_beta',r:-.112},{...base,r:null,missing_status:'not_reported'}];
 const original=structuredClone(rows);const out=run(rows);
 assert.equal(out.result.records.length,1);assert.equal(out.curation.excluded.length,3);assert.deepEqual(rows,original);
});
test('Source-defined and explicit long aliases resolve, while opposite measure direction remains separate',()=>{
 const out=curateExtraction([{records:[{...base,construct_x:'Negative destination image',measure_x:'NDI'},{...base,construct_x:'Destination image',measure_x:'DI'}],notes:''}],b,'Destination image (DI; negative destination image); Outcome (B)');
 assert.equal(out.result.records.length,2);
});
test('Empty construct dictionary does not exclude valid associations; distinct beta rows are retained',()=>{
 assert.equal(run([base],'').result.records.length,1);
 assert.equal(run([{...base,correlation_type:'standardized_beta',row:'Model 1'},{...base,correlation_type:'standardized_beta',row:'Model 2'}]).result.records.length,2);
});
test('Uploaded report identity merges invented page/author IDs but preserves raw suggestions',()=>{
 const rows=[{...base,study_id:'p1'},{...base,study_id:'Author2022'}];
 const out=curateExtraction([{records:rows,notes:''}],b,'Exposure (A); Outcome (B)','uploaded-report');
 assert.equal(out.result.records.length,1);assert.equal(out.result.records[0].study_id,'uploaded-report');assert.equal(rows[0].study_id,'p1');
});
test('Narrative repetitions sharing source evidence do not become separate tables',()=>{
 const out=run([base,{...base,table:'text',row:'A to B'}]);assert.equal(out.result.records.length,1);
});
test('Numbered triangular matrix omits leading means/SD and rejects a value from a different pair',()=>{
 const blocks=[{id:'p',page:1,location:'1',text:'Table 2. Correlation coefficients\nM | SD | 1 | 2 | 3\n1. PE | 19 | 8 | 1\n2. PESE | 2 | .5 | .248** | 1\n3. PSS | 5 | 1 | .242** | .501** | 1\nPE = physical exercise.'}];
 const m=matrixCandidates(blocks);assert.equal(m.length,1);assert.equal(m[0].cells.length,3);assert.equal(m[0].cells[0].value,.248);
 const r={...base,construct_x:'Physical exercise',construct_y:'Exercise self-efficacy',measure_x:'PE',measure_y:'PESE',table:'Table 2',r:.501,notes:'basis: correlation_matrix.'};
 const out=curateExtraction([{records:[r],notes:''}],blocks,'Physical exercise (PE); Exercise self-efficacy (PESE); Social support (PSS)');
 assert.equal(out.result.records.length,0);assert.match(out.curation.excluded[0].reasons.join(),/does not match/);
});
test('Named triangular matrix excludes leading descriptive columns',()=>{
 const blocks=[{id:'p',page:1,location:'1',text:'Table 1. Correlations\nVariable | M | SD | A | B | C\nA | 20 | 8 | 1\nB | 3 | .5 | .4 | 1\nC | 5 | 1 | -.2 | .3 | 1\nN = 100.'}];
 assert.deepEqual(matrixCandidates(blocks)[0].cells.map(c=>c.value),[.4,-.2,.3]);
});
test('A longer unrelated construct is not accepted solely because it contains a requested name',()=>{
 const r={...base,construct_x:'Physical exercise atmosphere',measure_x:'PEA',construct_y:'Outcome'};
 assert.equal(run([r],'Physical exercise (PE); Outcome (B)').result.records.length,0);
 assert.equal(run([base],'Not specified').result.records.length,1);
});
test('Conflicting generic/Pearson labels of one verified matrix cell consolidate as unresolved; partial remains separate',()=>{
 const blocks=[{id:'p',page:1,location:'1',text:'Table 1. Correlations\nA | B | C\nA | 1\nB | .4 | 1\nC | .2 | .3 | 1'}];
 const rows=['other','unclear','pearson_zero_order','partial'].map(t=>({...base,correlation_type:t,notes:'basis: correlation_matrix.'}));
 const out=curateExtraction([{records:rows,notes:''}],blocks,'Exposure (A); Outcome (B)');
 assert.equal(out.result.records.length,2);assert.equal(out.result.records[0].correlation_type,'other');assert.match(out.result.records[0].notes,/labels disagree/);assert.equal(out.result.records[1].correlation_type,'partial');
});
test('Rounded prose duplicates merge only with one known-sample matrix match',()=>{
 const blocks=[{id:'table',page:1,location:'1',text:'Table 1. Correlations\nA | B | C\nA | 1\nB | .362 | 1\nC | .5 | .6 | 1\nN = 100.'},{id:'prose',page:2,location:'2',text:'The correlation was r = .36 in 100 participants.'}];
 const matrix={...base,table:'Table 1',row:'B',column:'A',r:.362,notes:'Coefficient copied from the original lower_triangle matrix cell; source basis: correlation_matrix.',evidence:[{block_id:'table',quotation:blocks[0].text}]};
 const prose={...matrix,table:'',row:'',column:'',r:.36,notes:'Source statistic: r; basis: correlation_matrix.',evidence:[{block_id:'prose',quotation:blocks[1].text}]};
 const out=curateExtraction([{records:[matrix,prose],notes:''}],blocks,'Exposure (A); Outcome (B)');
 assert.equal(out.result.records.length,1);assert.equal(out.result.records[0].r,.362);assert.equal(out.result.records[0].evidence.length,2);
 for(const patch of [{n:101},{n:null},{correlation_type:'standardized_beta'},{table:'Table 2'},{timepoint:'follow-up'}])assert.equal(curateExtraction([{records:[matrix,{...prose,...patch}],notes:''}],blocks,'Exposure (A); Outcome (B)').result.records.length,2,JSON.stringify(patch));
});
