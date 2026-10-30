import { COMMON, MODEL, validateAgainstSchema } from './analysis';
export async function modelCall(key: string, instructions: string, input: unknown, schema: Record<string, any>, reasoning: 'low' | 'medium' = 'low', model: string = MODEL) {
  if (!key || !key.startsWith('sk-')) throw new Error('Enter a valid OpenAI API key in Settings.');
  const started = Date.now();
  const common = schema.properties?.decision ? COMMON : COMMON.split(' In title/abstract screening,')[0];
  const systemInstructions=common+'\n'+instructions;
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST', signal: AbortSignal.timeout(180000),
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, store: false, reasoning_effort: reasoning, max_completion_tokens: 12000,
      messages: [{ role: 'system', content: systemInstructions }, { role: 'user', content: JSON.stringify(input) }],
      response_format: { type: 'json_schema', json_schema: { name: 'analysis', strict: true, schema } },
    }),
  });
  if (!response.ok) {
    const errors: Record<number, string> = { 400: 'The selected model or request is not available for this key. Check model access in your OpenAI account.', 401: 'API key rejected.', 403: 'This key cannot access the requested model.', 404: 'The selected model is unavailable for this key. Choose an available model for a new task.', 429: 'API rate or credit limit reached. Retry after checking your API account.' };
    throw new Error(errors[response.status] || `Model request failed (HTTP ${response.status}). Retry this task.`);
  }
  const payload = await response.json() as any;
  const choice = payload.choices?.[0];
  if (choice?.finish_reason !== 'stop' || choice?.message?.refusal) throw new Error('Model did not return a complete analysis. Retry or use smaller source blocks.');
  const result = JSON.parse(choice.message.content);
  validateAgainstSchema(result, schema);
  const serializedInput = JSON.stringify(input);
  const inputSha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(serializedInput)))).map(x=>x.toString(16).padStart(2,'0')).join('');
  const source = input as {blocks?:{id:string}[]; context_blocks?:{id:string}[]};
  return { result, metadata: { model: payload.model, requested_model: model, reasoning_effort:reasoning, response_id: payload.id, usage: payload.usage, elapsed_ms: Date.now() - started, created: new Date().toISOString(), instructions: systemInstructions, input_sha256:inputSha256, source_block_ids:source.blocks?.map(b=>b.id)||[], context_block_ids:source.context_blocks?.map(b=>b.id)||[] } };
}
