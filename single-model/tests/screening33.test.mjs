import test from 'node:test';import assert from 'node:assert/strict';
import {consensusResult,checkedScreeningResult,screeningInput} from '../.test-build/screening.js';
const b=[{id:'abstract',text:'This qualitative interview study explored experiences.',page:null,location:'Abstract'}];
const exclude={decision:'exclude',explanation:'Qualitative design.',criteria_checks:[{criterion:'Quantitative study',finding:'not_met',evidence:[{block_id:'abstract',quotation:'qualitative interview study'}]}]};
test('Three exact agreements can avoid adjudication; two and binary-only agreement cannot',()=>{
 assert.equal(consensusResult([exclude,exclude,exclude],b).decision,'exclude');
 assert.equal(consensusResult([exclude,exclude],b),null);
 assert.equal(consensusResult([{...exclude,decision:'include'},{...exclude,decision:'uncertain'},{...exclude,decision:'include'}],b),null);
 assert.equal(consensusResult([exclude,exclude,{...exclude,decision:'uncertain'}],b),null);
});
test('Exclusion without a source-matched failed criterion is referred for review',()=>{
 const invalid={...exclude,criteria_checks:[{...exclude.criteria_checks[0],evidence:[{block_id:'abstract',quotation:'randomized controlled trial'}]}]};
 assert.equal(consensusResult([exclude,exclude,invalid],b).decision,'exclude');
 assert.equal(checkedScreeningResult(invalid,b).review_required,true);assert.equal(invalid.decision,'exclude');
 assert.equal(checkedScreeningResult(exclude,b).review_required,false);
});
test('Eligibility calls never import extraction dictionaries or templates as additional requirements',()=>{
 const input={title:'Example',criteria:'Adults',constructs:'Correlation pairs',extraction_template:'Zero order r required',blocks:b,stage:'screen'};
 const out=screeningInput(input);assert.equal(out.constructs,'');assert.equal(out.criteria,'Adults');assert.deepEqual(out.blocks,b);
});
