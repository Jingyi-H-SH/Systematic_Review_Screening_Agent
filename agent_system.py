import os
import io
import asyncio
import pandas as pd
from typing import Literal, Type, Optional, List, Callable, Awaitable
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, AuthenticationError, APIError

# Schemas (Simplified as per your new code)
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
    title: str = Field(..., description="A rewritten, synthetic title for the article.")
    abstract: str = Field(..., description="A rewritten, synthetic abstract for the article.")
    reasoning: str = Field(..., description="A concise explanation for the ground_truth label, based on the criteria.")

class GeneratedSamples(BaseModel):
    include_samples: List[FewShotExample]
    exclude_samples: List[FewShotExample]

# Prompts (Updated as per your new code)
SYSTEM_TEMPLATE = "You are a specialized systematic-review screener. Follow your specific instructions precisely. Output MUST be valid JSON.".strip()

SAMPLE_GENERATOR_PROMPT = """
You are an expert in scientific writing and systematic reviews. Your task is to create high-quality, synthetic few-shot examples for training other AI agents.
You will be given a small batch of real articles with their ground truth labels ('include' or 'exclude').
**Your Instructions:**
1.  For each article provided, you **MUST NOT** use the original title or abstract.
2.  You **MUST** rewrite the title and abstract to be conceptually similar but phrased differently. The goal is to teach the reasoning process, not to memorize specific text.
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
Return JSON with your "decision" and a detailed "explanation" justifying your choice by citing specific criteria.
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
Step 1: **Deconstruct the Inclusion Criteria.** Mentally break down each inclusion criterion into its core concepts.
Step 2: **Evaluate Reviewer Arguments against Concepts.** Analyze the reviewers' explanations. Does the abstract *imply* a concept even if it doesn't use the exact keyword?
Step 3: **Assess the Risk of a False Negative.** State the strongest possible argument for **inclusion**. What is the risk of missing this study if you exclude it?
Step 4: **Perform a Self-Correction Check (Crucial Step).**
   - **If your initial inclination is to 'exclude'**: Challenge yourself. Ask: "Is there *any reasonable, good-faith interpretation* of the abstract that could meet all inclusion criteria, even if it's a stretch?" If the answer is yes, you MUST reverse your inclination to 'include' and add a note. This is your primary directive to maximize recall.
Step 5: **Make a Final, Risk-Averse Decision.** Decide to **'exclude'** if, after the Self-Correction Check, you are confident that one or more core concepts of the inclusion criteria are definitively absent and cannot be reasonably inferred.
Step 6: **Add Notes for Ambiguity.** If your final decision is 'include' as a result of the Self-Correction Check or to resolve significant ambiguity, you MUST add a `notes` field with the exact value "need full-text to double-check".
Finally, provide a brief `explanation`. Return your final judgment as a JSON object, ensuring the primary decision key is "final_decision".
""".strip()


# Agent Class
class Agent:
    def __init__(self, name: str, instructions: str, output_type: Type[BaseModel], model: str = "gpt-5-nano"): # Model changed from gpt-5
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
        except AuthenticationError as e:
            error_message = "OpenAI Authentication Error: Please check your OPENAI_API_KEY."
            print(f"CRITICAL ERROR for agent {self.name}: {error_message}")
            raise RuntimeError(error_message) from e
        except APIError as e:
            error_message = f"OpenAI API Error: {e.message}"
            print(f"API ERROR for agent {self.name}: {error_message}")
            raise RuntimeError(error_message) from e
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            print(f"UNEXPECTED ERROR for agent {self.name}: {error_message}\nRaw content: {content if 'content' in locals() else 'N/A'}")
            raise RuntimeError(error_message) from e

# Agent Handlers
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
    prompt = REVIEWER_PROMPT_TEMPLATE.format(persona=persona, examples_section=examples_section, criteria=criteria, title=article.get("title", ""), abstract=article.get("abstract", ""), task=task)
    return await agent.run(prompt)

async def call_adjudicator(r1, r2, r3):
    prompt = ADJUDICATOR_PROMPT.format(r1=r1.model_dump_json(), r2=r2.model_dump_json(), r3=r3.model_dump_json())
    return await agent_adjudicator.run(prompt)

# Main Orchestration Function for the Web App
async def run_screening_process(
    df_articles: pd.DataFrame,
    criteria: str,
    progress_callback: Callable[[str], Awaitable[None]]
) -> pd.DataFrame:
    await progress_callback("流程开始...")

    include_examples_str, exclude_examples_str = "", ""
    if 'ground_truths' in df_articles.columns and df_articles['ground_truths'].notna().any():
        await progress_callback("发现'ground_truths'列，正在生成少样本示例...")
        gt_df = df_articles.dropna(subset=['ground_truths'])
        include_samples_df = gt_df[gt_df['ground_truths'] == 'include'].sample(n=min(2, len(gt_df[gt_df['ground_truths'] == 'include'])))
        exclude_samples_df = gt_df[gt_df['ground_truths'] == 'exclude'].sample(n=min(2, len(gt_df[gt_df['ground_truths'] == 'exclude'])))
        
        if not include_samples_df.empty or not exclude_samples_df.empty:
            samples_to_generate_df = pd.concat([include_samples_df, exclude_samples_df])
            generated_samples = await call_sample_generator(samples_to_generate_df, criteria)
            include_examples_str = format_samples("INCLUDE", generated_samples.include_samples)
            exclude_examples_str = format_samples("EXCLUDE", generated_samples.exclude_samples)
            await progress_callback("少样本示例已生成。")
        else:
            await progress_callback("未找到足够数据生成样本，将以零样本模式运行。")
    else:
        await progress_callback("未发现'ground_truths'列，将以零样本模式运行。")

    df_to_screen = df_articles.copy()
    
    # Initialize result columns (without confidence)
    result_cols = [
        "rev1_decision", "rev1_explanation",
        "rev2_decision", "rev2_explanation",
        "rev3_decision", "rev3_explanation",
        "final_decision", "final_explanation", "notes"
    ]
    for col in result_cols:
        df_to_screen[col] = ""

    total_rows = len(df_to_screen)
    for idx, row in df_to_screen.iterrows():
        await progress_callback(f"正在处理第 {idx + 1}/{total_rows} 篇文章...")

        article = row.to_dict()
        p_inclusion = ("You are a 'high-recall' screener...", "Analyze with a strong bias towards inclusion...")
        p_exclusion = ("You are a 'high-precision' screener...", "Scrutinize against all Exclusion Criteria first...")
        p_balanced = ("You are a neutral, balanced screener...", "Weigh all factors and make a balanced decision...")

        semaphore = asyncio.Semaphore(3)
        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        task1 = run_with_semaphore(call_specialized_reviewer(agent_inclusion, p_inclusion[0], p_inclusion[1], criteria, article, include_examples_str, exclude_examples_str))
        task2 = run_with_semaphore(call_specialized_reviewer(agent_exclusion, p_exclusion[0], p_exclusion[1], criteria, article, include_examples_str, exclude_examples_str))
        task3 = run_with_semaphore(call_specialized_reviewer(agent_balanced, p_balanced[0], p_balanced[1], criteria, article, include_examples_str, exclude_examples_str))

        r1, r2, r3 = await asyncio.gather(task1, task2, task3)
        final = await call_adjudicator(r1, r2, r3)

        df_to_screen.loc[idx, "rev1_decision"] = r1.decision
        df_to_screen.loc[idx, "rev1_explanation"] = r1.explanation
        
        df_to_screen.loc[idx, "rev2_decision"] = r2.decision
        df_to_screen.loc[idx, "rev2_explanation"] = r2.explanation

        df_to_screen.loc[idx, "rev3_decision"] = r3.decision
        df_to_screen.loc[idx, "rev3_explanation"] = r3.explanation
        
        df_to_screen.loc[idx, "final_decision"] = final.final_decision
        df_to_screen.loc[idx, "final_explanation"] = final.explanation
        df_to_screen.loc[idx, "notes"] = final.notes or ""

    return df_to_screen

