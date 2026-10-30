import test from 'node:test';
import assert from 'node:assert/strict';
import {sourceUnits,resolveLinked,linkedSchema,statisticIssues} from '../.test-build/evidence-extraction.js';
import {pdfTextRows} from '../.test-build/pdf-layout.js';
import {matrixCandidates,extractionBatches,matrixValueIssues} from '../.test-build/matrices.js';
import {matrixMappingSchema,extractMappedMatrices} from '../.test-build/matrix-extraction.js';
import {evidenceIssues,validateAgainstSchema,validateExtraction} from '../.test-build/analysis.js';
test('Source references retrieve exact original slices and unknown IDs remain invalid',()=>{
 const blocks=[{id:'p1',text:'A ﬁnding was reported.\nTable 2\nA | B\nC | −0.233\n',page:1,location:'Page 1'}];
 const units=sourceUnits(blocks);
 const raw={records:[{field:'Finding',value:'test',missing_status:'reported',source_ids:[units[0].id],notes:''}],notes:''};
 validateAgainstSchema(raw,linkedSchema('data'));
 assert.throws(()=>validateAgainstSchema({...raw,records:[{...raw.records[0],source_ids:['p1']}]},linkedSchema('data',units.map(u=>u.id))));
 const result=resolveLinked(raw,units).result;
 assert.deepEqual(evidenceIssues(result.records[0].evidence,blocks),[]);
 assert.equal(result.records[0].evidence[0].quotation,blocks[0].text);
 raw.records[0].source_ids=['invented'];
 assert.match(evidenceIssues(resolveLinked(raw,units).result.records[0].evidence,blocks).join(),/Unknown source/);
});
test('Unicode ligatures match; invented omissions and changed signs do not',()=>{
 const blocks=[{id:'p',text:'A ﬁnding was −0.23. A second finding was 0.20.',page:1,location:'1'}];
 assert.equal(evidenceIssues([{block_id:'p',quotation:'A finding was −0.23.'}],blocks).length,0);
 assert.equal(evidenceIssues([{block_id:'p',quotation:'A finding ... was 0.20.'}],blocks).length,1);
 assert.equal(evidenceIssues([{block_id:'p',quotation:'A finding was 0.23.'}],blocks).length,1);
});
test('PDF table cells remain on their baseline despite cell-level EOL markers',()=>{
 const item=(str,x,y,width=20)=>({str,transform:[1,0,0,1,x,y],width,hasEOL:true});
 assert.equal(pdfTextRows([item('A',0,100),item('B',100,100),item('C',0,80),item('−0.233',100,80)]),'A | B\nC | −0.233');
});
test('A path coefficient cannot pass as Pearson solely because the value appears',()=>{
 assert.match(statisticIssues({correlation_type:'pearson_zero_order',notes:'basis: regression_or_path.',evidence:[{quotation:'Path coefficient = −0.199'}]}).join(),/conflicts/);
 assert.equal(statisticIssues({correlation_type:'other',notes:'basis: correlation_matrix.',evidence:[]}).length,0);
});
test('Standardized beta may exceed one but a Pearson coefficient cannot',()=>{
 const r={study_id:'s',sample_id:'a',construct_x:'A',construct_y:'B',direction_x:'higher',direction_y:'higher',overlap_group:'reviewed',timepoint:'t',subgroup:'all',correlation_type:'standardized_beta',r:1.2,n:100,missing_status:'reported',evidence:[{block_id:'p',quotation:'beta 1.2 n 100'}]};
 const b=[{id:'p',text:'beta 1.2 n 100',page:1,location:'1'}];
 assert.equal(validateExtraction({records:[r],notes:''},b)[0].some(x=>x.startsWith('Reported r')),false);
 assert.equal(validateExtraction({records:[{...r,correlation_type:'pearson_zero_order'}],notes:''},b)[0].some(x=>x.startsWith('Reported r')),true);
});
test('Triangular matrix lookup uses header order, omits diagonals, and checks the cell',()=>{
 const b=[{id:'p',page:1,location:'1',text:'Table 4. Discriminant validity\nFornell-Larcker Criterion\nA | B | C\nA | 0.8\nB | -0.23 | 0.7\nC | 0.4 | 0.5 | 0.9'}];
 const m=matrixCandidates(b);assert.equal(m.length,1);assert.equal(m[0].kind,'fornell_larcker');assert.equal(m[0].cells.length,3);assert.equal(m[0].cells[0].value,-0.23);
 const r={row:'B',column:'A',r:0.4,notes:'basis: correlation_matrix.',evidence:[{block_id:'p'}]};
 assert.match(matrixValueIssues(r,b).join(),/does not match/);
 assert.deepEqual(matrixValueIssues({...r,r:-0.23},b),[]);
 const all=[{id:'intro',text:'Intro',page:1,location:'1'},...b,{id:'end',text:'Conclusion',page:2,location:'2'}];
 assert.deepEqual(extractionBatches(all,1000,'matrix-first-v1').flat().map(b=>b.id),['p','intro','end']);
});
test('Matrix parser separates descriptive columns and signed adjacent coefficients',()=>{
 const b=[{id:'p',page:1,location:'1',text:'Table 4 Discriminant validity\nConstructs | RP | NDI | SPB | TS | Mean | Std.\nRP | 0.725 | 5.865 | 0.866\nNDI | 0.337** | 0.697 | 4.393 | 0.980\nSPB | 0.348** | 0.315** | 0.721 | 5.594 | 1.050\nTS | 0.064 | −0.233** − 0.035 | 0.807 | 5.411 | 0.937'}];
 const m=matrixCandidates(b);assert.equal(m.length,1);assert.equal(m[0].cells.find(c=>c.row==='TS'&&c.column==='NDI').value,-0.233);
 assert.equal(m[0].cells.find(c=>c.row==='TS'&&c.column==='SPB').value,-0.035);
 assert.equal(m[0].headers.includes('Mean'),false);
});
test('Code reads only mapped requested cells; the model does not supply coefficient values',()=>{
 const blocks=[{id:'p',page:1,location:'1',text:'Table 4. Fornell-Larcker\nA | B | C\nA | 0.8\nB | -0.23 | 0.7\nC | 0.4 | 0.5 | 0.9'}];
 const matrices=matrixCandidates(blocks),units=sourceUnits(blocks);
 const map={matrices:{matrix_0:{columns:{A:{requested:true,construct:'Exposure',direction:'higher'},B:{requested:true,construct:'Outcome',direction:'higher'},C:{requested:false,construct:'Unrelated',direction:''}},interpretation:'inter_construct_correlations',sample_label:'main',n:null,n_source_ids:[],notes:''}},notes:''};
 validateAgainstSchema(map,matrixMappingSchema(matrices,units));
 const out=extractMappedMatrices(map,matrices,units,blocks,'report');
 assert.equal(out.result.records.length,1);assert.equal(out.result.records[0].r,-0.23);
 assert.equal(out.result.records[0].correlation_type,'other');assert.equal(out.result.records[0].n,null);
 assert.deepEqual(evidenceIssues(out.result.records[0].evidence,blocks),[]);
 const asymmetric=[{...blocks[0],text:'Table 2. Correlations and squared correlations\nA | B | C\nA | 1 | 0.5 | 0.6\nB | 0.25 | 1 | 0.7\nC | 0.36 | 0.49 | 1'}];
 assert.equal(matrixCandidates(asymmetric).length,0);
});

test('Explicit user aliases outrank an erroneous model scope exclusion',()=>{
 const blocks=[{id:'p',page:1,location:'1',text:'Table 4. Fornell-Larcker\nDTI | ESC | TRS\nDTI | 0.8\nESC | 0.364 | 0.7\nTRS | 0.586 | 0.226 | 0.9'}];
 const matrices=matrixCandidates(blocks),units=sourceUnits(blocks);
 const raw={matrices:{matrix_0:{columns:Object.fromEntries(['DTI','ESC','TRS'].map(k=>[k,{requested:false,construct:k,direction:''}])),interpretation:'inter_construct_correlations',sample_label:'main',n:null,n_source_ids:[],notes:''}},notes:''};
 const out=extractMappedMatrices(raw,matrices,units,blocks,'report','Destination image (DI; DTI), tourist satisfaction (TS; TRS; SAT), and escapism (ESC).');
 assert.deepEqual(out.result.records.map(r=>r.r),[0.364,0.586,0.226]);
 assert.equal(out.result.records[2].construct_x,'tourist satisfaction');
 assert.equal(raw.matrices.matrix_0.columns.TRS.requested,false);
});

test('An unrelated column marked requested cannot bypass an explicit construct dictionary',()=>{
 const blocks=[{id:'p',page:1,location:'1',text:'Table 4. Fornell-Larcker\nDTI | ENJ | TRS\nDTI | 0.8\nENJ | 0.4 | 0.7\nTRS | 0.586 | 0.2 | 0.9'}];
 const matrices=matrixCandidates(blocks),units=sourceUnits(blocks);
 const raw={matrices:{matrix_0:{columns:{DTI:{requested:true,construct:'DTI',direction:''},ENJ:{requested:true,construct:'Enjoyment',direction:''},TRS:{requested:true,construct:'TRS',direction:''}},interpretation:'inter_construct_correlations',sample_label:'main',n:null,n_source_ids:[],notes:''}},notes:''};
 const out=extractMappedMatrices(raw,matrices,units,blocks,'report','Destination image (DI; DTI), tourist satisfaction (TS; TRS; SAT).');
 assert.equal(out.result.records.length,1);assert.equal(out.result.records[0].r,0.586);
});

test('Source-defined aliases retain requested constructs without accepting unrelated columns',()=>{
 const blocks=[{id:'p',page:1,location:'1',text:'Table 3. Fornell-Larcker\nDI | PA | OS\nDI | 0.8\nPA | 0.4 | 0.7\nOS | 0.703 | 0.2 | 0.9\nPA = place attachment; OS = overall satisfaction; DI = destination image;'}];
 const matrices=matrixCandidates(blocks),units=sourceUnits(blocks);
 const raw={matrices:{matrix_0:{columns:Object.fromEntries(['DI','PA','OS'].map(k=>[k,{requested:true,construct:k,direction:''}])),interpretation:'inter_construct_correlations',sample_label:'main',n:null,n_source_ids:[],notes:''}},notes:''};
 const out=extractMappedMatrices(raw,matrices,units,blocks,'report','Destination image (DI; DTI), tourist satisfaction (TS; TRS; overall satisfaction).');
 assert.equal(out.result.records.length,1);assert.equal(out.result.records[0].r,0.703);
 assert.equal(out.provenance[0].source_defined_aliases[0].text,'OS = overall satisfaction');
});
test('Thousands separators in cited sample size do not produce a missing-N warning',()=>{
 const b=[{id:'p',page:1,location:'1',text:'N = 1,433; r = -0.158'}];
 const r={study_id:'s',sample_id:'a',construct_x:'A',construct_y:'B',direction_x:'higher',direction_y:'higher',overlap_group:'reviewed',timepoint:'t',subgroup:'all',correlation_type:'pearson_zero_order',r:-.158,n:1433,missing_status:'reported',evidence:[{block_id:'p',quotation:b[0].text}]};
 assert.equal(validateExtraction({records:[r],notes:''},b)[0].some(x=>x.startsWith('Sample size is not present')),false);
});
