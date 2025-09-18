import os
import asyncio
import pandas as pd
from typing import Literal, Type, Optional, List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# --- Schemas (Data Structures) ---
ReviewerDecision = Literal["include", "read full-text to decide", "exclude"]
FinalBinaryDecision = Literal["include", "exclude"]

class ReviewResult(BaseModel):
    decision: ReviewerDecision
    explanation: str

class FinalDecision(BaseModel):
    final_decision: FinalBinaryDecision
    explanation: str
    notes: Optional[str] = None

class FewShotExample(BaseModel):
    title: str
    abstract: str
    reasoning: str

class GeneratedSamples(BaseModel):
    include_samples: List[FewShotExample]
    exclude_samples: List[FewShotExample]

# --- Prompts ---
SYSTEM_TEMPLATE = "You are a specialized systematic-review screener. Follow your specific instructions precisely. Output MUST be valid JSON.".strip()

SAMPLE_GENERATOR_PROMPT = """
You are an expert in scientific writing and systematic reviews. Your task is to create high-quality, synthetic few-shot examples for training other AI agents.
You will be given a small batch of real articles with their ground truth labels ('include' or 'exclude').

**Your Instructions:**
1.  For each article provided, you **MUST NOT** use the original title or abstract.
2.  You **MUST** rewrite the title and abstract to be conceptually similar but phrased differently.
3.  Based on the `ground_truths` label, write a concise `reasoning` that clearly explains **why** the article was included or excluded, referencing the screening criteria.
4.  Generate separate lists for 'include' and 'exclude' samples.

**Screening Criteria for context:**
{criteria}

**Articles to rewrite:**
{articles_json}

Return a single JSON object. The top-level keys MUST be "include_samples" and "exclude_samples".
Within each sample object, the keys MUST be exactly "title", "abstract", and "reasoning".
"""

REVIEWER_PROMPT_TEMPLATE = """
Your Persona: {persona}

{examples_section}
CRITERIA:
{criteria}

ARTICLE:
Title: {title}
Abstract: {abstract}

TASK:
{task}
Your "decision" MUST be one of the exact following strings: "include", "read full-text to decide", or "exclude".
Return JSON with your "decision" and a detailed "explanation".
"""

ADJUDICATOR_PROMPT = """
You are the Adjudicator, a senior researcher with deep expertise in systematic reviews. Your final decision MUST be 'include' or 'exclude'.

You have received three reviews from screeners with different biases:
- Reviewer 1 (High-Recall): Biased towards **inclusion**.
- Reviewer 2 (High-Precision): Biased towards **exclusion**.
- Reviewer 3 (Balanced): Aims for objectivity.

THE REVIEWS:
- Reviewer 1 (Inclusion-focused) said: {r1}
- Reviewer 2 (Exclusion-focused) said: {r2}
- Reviewer 3 (Balanced) said: {r3}

YOUR ADJUDICATION TASK (Chain-of-Thought Self-Correction):
Follow these steps meticulously to arrive at the most robust decision.

Step 1: **Deconstruct the Inclusion Criteria.** Break down each inclusion criterion into its core concepts.
Step 2: **Evaluate Reviewer Arguments against Concepts.** Analyze the explanations. Does the abstract *imply* a concept even if it doesn't use the exact keyword?
Step 3: **Assess the Risk of a False Negative.** State the strongest possible argument for **inclusion**. What is the risk of missing this study if you exclude it?
Step 4: **Perform a Self-Correction Check (Crucial Step).**
   - **If your initial inclination is to 'exclude'**: Challenge yourself. Ask: "Is there *any reasonable, good-faith interpretation* of the abstract that could meet all inclusion criteria, even if it's a stretch?" If the answer is yes, you MUST reverse your inclination to 'include' and add a note.
Step 5: **Make a Final, Risk-Averse Decision.**
   Decide to **'exclude'** if you are highly confident that core concepts are definitively absent.
Step 6: **Add Notes for Ambiguity.** If your decision is 'include' as a result of the Self-Correction Check, you MUST add a `notes` field with the exact value "need full-text to double-check".

Provide a brief `explanation` summarizing your reasoning. Return a JSON object with the primary key "final_decision".
"""

# --- Agent Class & Handlers ---
class Agent:
    def __init__(self, name: str, instructions: str, output_type: Type[BaseModel], model: str = "gpt-4o"):
        self.name = name
        self.instructions = instructions
        self.output_type = output_type
        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def run(self, prompt: str):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.instructions}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return self.output_type.model_validate_json(content)
        except Exception as e:
            print(f"Error for agent {self.name}: {e}\nRaw content from API might be unavailable.")
            if self.output_type == ReviewResult: return ReviewResult(decision="exclude", explanation=f"Error: {e}")
            elif self.output_type == FinalDecision: return FinalDecision(final_decision="exclude", explanation=f"Error: {e}", notes="Error during adjudication.")
            elif self.output_type == GeneratedSamples: return GeneratedSamples(include_samples=[], exclude_samples=[])

agent_sample_generator = Agent(name="sample_generator", instructions=SYSTEM_TEMPLATE, output_type=GeneratedSamples)
agent_inclusion = Agent(name="reviewer_inclusion", instructions=SYSTEM_TEMPLATE, output_type=ReviewResult)
agent_exclusion = Agent(name="reviewer_exclusion", instructions=SYSTEM_TEMPLATE, output_type=ReviewResult)
agent_balanced = Agent(name="reviewer_balanced", instructions=SYSTEM_TEMPLATE, output_type=ReviewResult)
agent_adjudicator = Agent(name="adjudicator", instructions=SYSTEM_TEMPLATE, output_type=FinalDecision)

def format_samples(title: str, samples: List[FewShotExample]) -> str:
    if not samples: return ""
    header = f"--- {title} EXAMPLES (for learning) ---\n"
    formatted = [f"Title: {s.title}\nAbstract: {s.abstract}\nReasoning: {s.reasoning}\n" for s in samples]
    return header + "\n".join(formatted) + "---------------------------------------\n\n"

async def call_sample_generator(articles_df: pd.DataFrame, criteria: str):
    prompt = SAMPLE_GENERATOR_PROMPT.format(criteria=criteria, articles_json=articles_df.to_json(orient='records', indent=2))
    return await agent_sample_generator.run(prompt)

async def call_specialized_reviewer(agent: Agent, persona: str, task: str, criteria: str, article: dict, include_examples: str, exclude_examples: str):
    examples_section = include_examples + exclude_examples
    prompt = REVIEWER_PROMPT_TEMPLATE.format(persona=persona, examples_section=examples_section, criteria=criteria, title=article["title"], abstract=article["abstract"], task=task)
    return await agent.run(prompt)

async def call_adjudicator(r1, r2, r3):
    prompt = ADJUDICATOR_PROMPT.format(r1=r1.model_dump_json(), r2=r2.model_dump_json(), r3=r3.model_dump_json())
    return await agent_adjudicator.run(prompt)

# --- Main Orchestration Logic ---
async def process_row(idx: int, row: pd.Series, df_ref: pd.DataFrame, criteria: str, include_examples_str: str, exclude_examples_str: str):
    article = row.to_dict()
    p_inclusion = ("You are a 'high-recall' screener.", "Analyze with a strong bias towards inclusion. Lean towards 'include' or 'read full-text to decide' if not explicitly excluded.")
    p_exclusion = ("You are a 'high-precision' screener.", "Scrutinize against all Exclusion Criteria first. If ambiguous, lean towards 'exclude'.")
    p_balanced = ("You are a neutral, balanced screener.", "Weigh all factors and make an objective decision.")

    tasks = [
        call_specialized_reviewer(agent_inclusion, p_inclusion[0], p_inclusion[1], criteria, article, include_examples_str, exclude_examples_str),
        call_specialized_reviewer(agent_exclusion, p_exclusion[0], p_exclusion[1], criteria, article, include_examples_str, exclude_examples_str),
        call_specialized_reviewer(agent_balanced, p_balanced[0], p_balanced[1], criteria, article, include_examples_str, exclude_examples_str)
    ]
    r1, r2, r3 = await asyncio.gather(*tasks)
    final = await call_adjudicator(r1, r2, r3)

    for i, r in enumerate([r1, r2, r3], start=1):
        df_ref.loc[idx, f"rev{i}_decision"] = r.decision
        df_ref.loc[idx, f"rev{i}_explanation"] = r.explanation
    df_ref.loc[idx, "final_decision"] = final.final_decision
    df_ref.loc[idx, "final_explanation"] = final.explanation
    df_ref.loc[idx, "notes"] = final.notes or ""

async def main_orchestration(df: pd.DataFrame, criteria: str, include_examples_str: str, exclude_examples_str: str):
    for i in range(1, 4):
        df[f"rev{i}_decision"], df[f"rev{i}_explanation"] = "", ""
    df["final_decision"], df["final_explanation"], df["notes"] = "", "", ""

    tasks = [process_row(i, row, df, criteria, include_examples_str, exclude_examples_str) for i, row in df.iterrows()]
    
    # Using asyncio.gather for concurrent execution
    await asyncio.gather(*tasks)
    
    return df

# --- Entry Point Function ---
async def run_screening_process(df_articles: pd.DataFrame, criteria: str) -> pd.DataFrame:
    """The main function that orchestrates the entire screening process."""
    rename_map = {"Article Title": "title", "Abstract": "abstract"}
    df_articles.rename(columns={k: v for k, v in rename_map.items() if k in df_articles.columns}, inplace=True)

    if "title" not in df_articles.columns or "abstract" not in df_articles.columns:
        raise ValueError("CSV must contain 'title' and 'abstract' columns.")

    include_examples_str, exclude_examples_str = "", ""
    if 'ground_truths' in df_articles.columns and df_articles['ground_truths'].notna().any():
        print("Found 'ground_truths' column. Starting SampleGeneratorAgent...")
        gt_df = df_articles.dropna(subset=['ground_truths'])
        include_samples_df = gt_df[gt_df['ground_truths'] == 'include'].sample(n=min(2, len(gt_df[gt_df['ground_truths'] == 'include'])))
        exclude_samples_df = gt_df[gt_df['ground_truths'] == 'exclude'].sample(n=min(2, len(gt_df[gt_df['ground_truths'] == 'exclude'])))
        
        if not include_samples_df.empty or not exclude_samples_df.empty:
            samples_to_generate_df = pd.concat([include_samples_df, exclude_samples_df])
            generated_samples = await call_sample_generator(samples_to_generate_df, criteria)
            include_examples_str = format_samples("INCLUDE", generated_samples.include_samples)
            exclude_examples_str = format_samples("EXCLUDE", generated_samples.exclude_samples)
            print("Dynamic few-shot samples generated.")
    else:
        print("No 'ground_truths' column found. Running in zero-shot mode.")

    df_to_screen = df_articles.copy()
    final_df = await main_orchestration(df_to_screen, criteria, include_examples_str, exclude_examples_str)

    # Drop the ground_truths column from the final output if it exists
    if 'ground_truths' in final_df.columns:
        final_df = final_df.drop(columns=['ground_truths'])

    return final_df
