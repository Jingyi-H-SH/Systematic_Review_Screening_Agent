import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {matrixCandidates} from '../.test-build/matrices.js';
import {mappedStatistic,extractMappedMatrices,matrixMappingSchema} from '../.test-build/matrix-extraction.js';
import {sourceUnits} from '../.test-build/evidence-extraction.js';
import {extractionContext,validateAgainstSchema} from '../.test-build/analysis.js';
const blocks=JSON.parse(fs.readFileSync(new URL('./fixtures/zeng-numbered-table.json',import.meta.url)));
test('Numbered table with Variable/M/SD header preserves all 36 off-diagonal cells',()=>{
 const m=matrixCandidates(blocks);assert.equal(m.length,1);assert.equal(m[0].cells.length,36);
 for(const [row,column,value] of [['MPA','PE',-.14],['SC','PE',.15],['SC','MPA',-.45]])assert.equal(m[0].cells.find(c=>c.row===row&&c.column===column).value,value);
 const omitted=blocks.map(b=>({...b,text:b.text.replace('Variable | M | SD','M | SD')}));
 assert.deepEqual(matrixCandidates(omitted)[0].cells,m[0].cells);
});
test('All target Zeng cells have header, numeric row, local definitions and N evidence',()=>{
 const matrices=matrixCandidates(blocks),units=sourceUnits(blocks);
 const mapping={columns:Object.fromEntries(matrices[0].headers.map(h=>[h,{requested:['PE','MPA','SC'].includes(h),construct:h,direction:'higher'}])),interpretation:'inter_construct_correlations',sample_label:'main',n:1843,n_source_ids:units.filter(u=>u.text.includes('1,843')).map(u=>u.id),notes:'',statistic:{method:'pearson',adjustment:'unresolved',conflicting_context:true,source_ids:[],rationale:'Adjustment unresolved.'}};
 validateAgainstSchema({matrices:{matrix_0:mapping},notes:''},matrixMappingSchema(matrices,units,true));
 const out=extractMappedMatrices({matrices:{matrix_0:mapping},notes:''},matrices,units,blocks,'s','Physical exercise (PE); mobile phone addiction (MPA); self-control (SC)');
 assert.deepEqual(out.result.records.map(r=>r.r),[-.14,.15,-.45]);
 for(const r of out.result.records){assert.equal(r.correlation_type,'unclear');assert.equal(r.n,1843);const t=r.evidence.map(e=>e.quotation).join(' ');assert.match(t,/Variable.*M.*SD/);assert.match(t,/1,843/);assert.ok(t.includes(String(r.r).replace('0.','0.'))||t.includes(String(r.r).replace('-','−')));}
});
const unit=(text,id='m')=>({id,parent_id:id,start:0,end:text.length,text,page:1,location:'1'});
const classify=(stat,units)=>mappedStatistic({statistic:stat},units,{caption:'Table 2. Correlations'}).type;
const stat={method:'pearson',adjustment:'unadjusted',conflicting_context:false,source_ids:['m'],rationale:'Linked to this table.'};
test('Method evidence is required and competing partial correlation context blocks a generic Pearson claim',()=>{
 assert.equal(classify(stat,[unit('Pearson correlations were calculated.')]),'pearson_zero_order');
 assert.equal(classify(stat,[unit('A coefficient of .8 was found.')]),'unclear');
 assert.equal(classify(stat,[unit('Pearson correlations were calculated. Partial correlations were estimated among these variables.')]),'unclear');
 assert.equal(classify({...stat,conflicting_context:true},[unit('Pearson bivariate correlations are shown in Table 2.')]),'pearson_zero_order');
 assert.equal(classify({...stat,conflicting_context:true},[unit('Pearson and Spearman correlations are reported without identifying this table.')]),'unclear');
 assert.equal(classify(stat,[unit('Pearson bivariate correlation results are shown in Table 2.'),unit('Partial correlations in Table 3.','other')]),'pearson_zero_order');
 assert.equal(classify({...stat,adjustment:'adjusted'},[unit('Partial correlation coefficients adjusting for age.')]),'partial');
});
test('Method context is selected even without sample-size keywords',()=>{
 const b=[{id:'focus',text:'Table 2.',page:1,location:'1'},{id:'method',text:'Statistical analysis used Pearson correlations.',page:2,location:'2'}];
 assert.deepEqual(extractionContext(b,[b[0]]).map(x=>x.id),['method']);
});
test('Literal full headers stay in scope when the model returns their abbreviations',()=>{
 const b=[{id:'p',page:1,location:'1',text:'Table 1. Correlations\nPhysical activity | Social support | Outcome\nPhysical activity | 1\nSocial support | .337 | 1\nOutcome | .36 | .4 | 1\nN = 989.'}];
 const matrices=matrixCandidates(b),units=sourceUnits(b);
 const raw={matrices:{matrix_0:{columns:{'Physical activity':{requested:true,construct:'PA',direction:'higher'},'Social support':{requested:true,construct:'SS',direction:'higher'},Outcome:{requested:false,construct:'Outcome',direction:''}},interpretation:'inter_construct_correlations',sample_label:'main',n:989,n_source_ids:[units[0].id],notes:''}},notes:''};
 const out=extractMappedMatrices(raw,matrices,units,b,'s','Physical activity (PA); social support (SS)');
 assert.equal(out.result.records.length,1);assert.equal(out.result.records[0].r,.337);assert.equal(out.result.records[0].n,989);
});
