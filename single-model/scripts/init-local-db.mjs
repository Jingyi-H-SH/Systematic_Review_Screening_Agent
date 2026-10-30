import {spawnSync} from 'node:child_process';
import {existsSync,readdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
if(!existsSync(path.join(root,'dist/server/wrangler.json')))throw new Error('Run npm run build before npm run db:init.');
for(const name of readdirSync(path.join(root,'drizzle')).filter(n=>n.endsWith('.sql')).sort()){
 const r=spawnSync(process.execPath,['--import','./scripts/sites-env.mjs','./node_modules/wrangler/bin/wrangler.js','d1','execute','DB','--local','--config','dist/server/wrangler.json','--persist-to','.wrangler/state','--file',`drizzle/${name}`],{cwd:root,stdio:'inherit'});
 if(r.error)throw r.error;if(r.status!==0)process.exit(r.status??1);
}
