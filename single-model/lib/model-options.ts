export const SCREENING_MODEL = 'gpt-6.1-sol';
export const MODEL_OPTIONS = [
  {id: SCREENING_MODEL, label: 'GPT-6.1 Sol'},
  {id: 'gpt-6-astra', label: 'GPT-6 Astra'},
  {id: 'gpt-5-2025-08-07', label: 'GPT-5'},
  {id: 'gpt-5-nano-2025-08-07', label: 'GPT-5 nano'},
] as const;
export const EXTRACTION_MODEL = 'gpt-5-nano-2025-08-07';
export function selectedModel(value: unknown, fallback: string = SCREENING_MODEL): string {
  if (value === undefined || value === null || value === '') return fallback;
  if (typeof value !== 'string' || !MODEL_OPTIONS.some(m=>m.id===value)) throw new Error('Choose a supported model in API settings.');
  return value;
}
