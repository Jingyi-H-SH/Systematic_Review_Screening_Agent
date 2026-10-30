import {spawnSync} from 'node:child_process';
import {mkdirSync,writeFileSync,readdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
function run(args){const r=spawnSync(process.execPath,args,{cwd:root,stdio:'inherit'});if(r.error)throw r.error;if(r.status!==0)process.exit(r.status??1);}
mkdirSync(path.join(root,'.test-build'),{recursive:true});
writeFileSync(path.join(root,'.test-build/package.json'),JSON.stringify({type:'commonjs'}));
const files=['analysis','model','model-options','screening','optimized-screening','importers','pdf-layout','evidence-extraction','matrices','matrix-extraction','masem-curation'];
run(['node_modules/typescript/bin/tsc',...files.map(f=>`lib/${f}.ts`),'--outDir','.test-build','--module','commonjs','--moduleResolution','node','--target','ES2022','--lib','ES2022,DOM','--esModuleInterop','--skipLibCheck']);
run(['--test',...readdirSync(path.join(root,'tests')).filter(f=>f.endsWith('.test.mjs')).map(f=>`tests/${f}`)]);
