import {chunks,type Block} from './analysis';
const number=(s:string)=>{const t=s.trim().replace(/[−–]/g,'-').replace(/[*†‡]+$/g,'');return /^-?(?:\d*\.\d+|\d+)$/.test(t)?Number(t):null;};
function rowNumbers(parts:string[]):number[]|null{
 const s=parts.join(' ').replace(/[−–]/g,'-').replace(/-\s+(?=\d)/g,'-').replace(/[\[\]()*†‡]/g,'');
 if(/[^\d.\s+\-]/.test(s))return null;
 return (s.match(/-?(?:\d*\.\d+|\d+)/g)||[]).map(Number);
}
export function triangleMeaning(text:string){
 const flat=text.replace(/\s+/g,' ');const statements=[...flat.matchAll(/\b(squared\s+correlations?|correlations?\s+squared|r\s*(?:²|\^2)|correlations?)\s+(?:(?:are|appear|reported|shown)\s+)*(above|below)\s+(?:the\s+)?diagonal/gi)];
 const correlations=statements.filter(m=>!/^squared|squared$|^r\s*/i.test(m[1]));
 const squared=statements.filter(m=>!correlations.includes(m));
 const above=correlations.some(m=>m[2].toLowerCase()==='above'),below=correlations.some(m=>m[2].toLowerCase()==='below');
 if(above===below)return null;
 const side=above?'upper':'lower';
 return {side,explicit_mixed:squared.some(m=>m[2].toLowerCase()===(above?'below':'above')),evidence:statements.map(m=>m[0]).join('; ')};
}
function numberedMatrices(block:Block){
 const lines=block.text.split('\n'),found:any[]=[];
 for(let h=0;h<lines.length;h++){
  const parts=lines[h].split('|').map(x=>x.trim()),start=parts.findIndex(x=>/^\(?1\)?$/.test(x));
  if(start<0)continue;
  const nums=parts.slice(start);if(nums.length<2||!nums.every((x,i)=>x.replace(/[()]/g,'')===String(i+1)))continue;
  const rows:any[]=[];let n=h+1;
  for(let k=0;k<nums.length;k++){
   const first=lines[n]?.match(/^(?:\((\d+)\)|(\d+)[.)])\s+([^|]+)\|(.+)$/);if(!first||Number(first[1]||first[2])!==k+1)break;
   // A named row-label header occupies a column already removed by the row regex.
   const descriptive=start-(/^(?:Variables?|Constructs?|Factors?|Items?)$/i.test(parts[0])?1:0);
   const begin=n;const cells=first[4].split('|').map(x=>x.trim()).slice(descriptive);let numeric=cells.join(' ');n++;
   while(n<lines.length&&!/^(?:\(\d+\)|\d+[.)])\s+/.test(lines[n])&&(/^[\d.\s|+−–*†‡()\-]+$/.test(lines[n])||/^[a-z]$/i.test(lines[n].trim()))){
    if(!/^[a-z]$/i.test(lines[n].trim()))numeric+=' '+lines[n].replace(/\|/g,' ');n++;
   }
   const values=rowNumbers([numeric]);if(!values||![k+1,nums.length].includes(values.length))break;
   rows.push({label:first[3].trim(),values,source_line:lines.slice(begin,n).join('\n')});
  }
  if(rows.length!==nums.length)continue;
  const legend_text=lines.slice(n,Math.min(lines.length,n+12)).join('\n');
  const meaning=triangleMeaning(legend_text),lower=rows.every((r,i)=>r.values.length===i+1),square=rows.every(r=>r.values.length===rows.length);const symmetric=square&&rows.every((r,ri)=>r.values.every((v:number,ci:number)=>ri===ci||Math.abs(v-rows[ci].values[ri])<1e-8));
  if(!lower&&!symmetric&&!meaning?.explicit_mixed)continue;
  const headers=rows.map(r=>r.label),side=meaning?.side||'lower';
  const cells=rows.flatMap((r,ri)=>r.values.flatMap((value:number,ci:number)=>(side==='upper'?ci>ri:ci<ri)?[{row:r.label,column:headers[ci],value}]:[]));
  const caption=lines.slice(Math.max(0,h-3),h).find(x=>/^Table\s+\d+/i.test(x))||'';
  found.push({source_block:block.id,page:block.page,caption,kind:meaning||/correlat/i.test(caption)?'correlation':'unclassified_matrix',headers,rows,cells,layout:lower?'lower_triangle':symmetric?'square':`mixed_${side}_r`,diagonal_excluded:true,legend_text,selection_basis:meaning?.evidence||'Ordered matrix columns',source_header:lines[h]});h=n;
 }return found;
}
export function matrixCandidates(blocks:Block[]){
  const found:any[]=[];
  for(const block of blocks){
    found.push(...numberedMatrices(block));
    const lines=block.text.split('\n');
    for(let i=0;i<lines.length;i++){
      let headers=lines[i].split('|').map(x=>x.trim());
      if(headers.length<3||headers.some(h=>!h||number(h)!==null))continue;
      let trailing=0;
      while(/^(Mean|Std\.?|SD|Standard deviation)$/i.test(headers.at(-1)||'')){headers.pop();trailing++;}
      if(/^(Constructs?|Variables?|Factors?)$/i.test(headers[0]))headers.shift();
      let leading=0;while(/^(M|Mean|SD|Std\.?|Standard deviation)$/i.test(headers[0]||'')){headers.shift();leading++;}
      if(headers.length<3||new Set(headers).size!==headers.length)continue;
      const rows:{label:string;values:number[];source_line:string}[]=[];
      for(let n=i+1;n<lines.length;n++){
        const cells=lines[n].split('|').map(x=>x.trim());
        if(cells.length<2)break;
        const values=rowNumbers(cells.slice(1));
        if(!values)break;
        rows.push({label:cells[0],values:(trailing?values.slice(0,-trailing):values).slice(leading),source_line:lines[n]});
      }
      if(rows.length===headers.length-1&&!rows.some(r=>r.label===headers[0]))headers=headers.slice(1);
      if(rows.length!==headers.length||!rows.every((r,n)=>r.label===headers[n]))continue;
      const lower=rows.every((r,n)=>r.values.length===n+1),square=rows.every(r=>r.values.length===headers.length);
      if(!lower&&!square)continue;
      const symmetric=square&&rows.every((r,ri)=>r.values.every((v,ci)=>ri===ci||Math.abs(v-rows[ci].values[ri])<1e-8));
      const legend_text=lines.slice(i+1+rows.length,i+1+rows.length+12).join('\n'),meaning=triangleMeaning(legend_text);
      if(square&&!symmetric&&!meaning?.explicit_mixed)continue;
      const preceding=lines.slice(Math.max(0,i-12),i).join('\n');
      const caption=[...preceding.matchAll(/(?:Table|TABLE)\s+\d+[^\n]*/g)].at(-1)?.[0]||'';
      const kind=/Fornell.?Larcker/i.test(preceding)?'fornell_larcker':/HTMT|heterotrait/i.test(preceding)?'htmt':/correlat/i.test(preceding)?'correlation':'unclassified_matrix';
      const side=meaning?.side||'lower';
      const cells=rows.flatMap((r,ri)=>r.values.flatMap((value,ci)=>(side==='upper'?ci>ri:ci<ri)?[{row:r.label,column:headers[ci],value}]:[]));
      found.push({source_block:block.id,page:block.page,caption,kind,headers,rows,cells,layout:lower?'lower_triangle':symmetric?'square':`mixed_${side}_r`,diagonal_excluded:true,legend_text:meaning?legend_text:'',selection_basis:meaning?.evidence});i+=rows.length;
    }
  }return found;
}
export function extractionBatches(blocks:Block[],limit:number,mode?:string):Block[][]{
  if(mode!=='matrix-first-v1')return chunks(blocks,limit);
  const ids=new Set(matrixCandidates(blocks).map(m=>m.source_block));
  for(const b of blocks)if(/^(?:Table|TABLE)\s+\d+[^\n]*(?:correlat|discriminant|descriptive statistics)/mi.test(b.text))ids.add(b.id);
  // Whole matrix-bearing pages first, then every remaining page. No page is omitted.
  return [...blocks.filter(b=>ids.has(b.id)).map(b=>[b]),...chunks(blocks.filter(b=>!ids.has(b.id)),limit)];
}
export function matrixValueIssues(record:any,blocks:Block[]):string[]{
  if(!/basis: correlation_matrix/.test(record.notes||''))return [];
  const matrices=matrixCandidates(blocks.filter(b=>record.evidence.some((e:any)=>e.block_id===b.id)));
  const matches=matrices.flatMap(m=>m.cells.filter((c:any)=>(c.row===record.row&&c.column===record.column)||(c.row===record.column&&c.column===record.row)).map((c:any)=>({m,c})));
  if(matches.length===1&&record.r!==matches[0].c.value)return ['Matrix cell value does not match the cited row and column.'];
  if(matches.length===1&&matches[0].m.kind==='htmt')return ['HTMT is not an extractable correlation.'];
  return [];
}
