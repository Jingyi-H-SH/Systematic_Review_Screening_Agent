/** Keep adjacent same-baseline PDF cells on one row; preserve source order across columns. */
export function pdfTextRows(items: any[]): string {
  const rows: {y:number; items:any[]}[]=[];
  for(const item of items){
    if(!('str' in item)||!item.str.trim())continue;
    const y=item.transform[5],last=rows.at(-1);
    if(last&&Math.abs(last.y-y)<=2)last.items.push(item);
    else rows.push({y,items:[item]});
  }
  return rows.map(row=>{
    let end:number|null=null, text='';
    for(const item of row.items){
      const x=item.transform[4];
      const gap=end===null?0:x-end;
      text+=(end===null?'':gap>12?' | ':gap<1?'':' ')+item.str;
      end=x+item.width;
    }
    return text;
  }).join('\n');
}
