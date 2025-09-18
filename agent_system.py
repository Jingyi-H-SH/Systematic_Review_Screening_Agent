import os
import io
import asyncio
import pandas as pd
from typing import Literal, Type, Optional, List, Callable, Awaitable
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# Schemas
ReviewerDecision = Literal["include", "read full-text to decide", "exclude"]
FinalBinaryDecision = Literal["include", "exclude"]

class ReviewResult(BaseModel):
    decision: ReviewerDecision
    confidence_score: int = Field(..., description="A confidence score from 1 (very unsure) to 5 (very sure).")
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

# Prompts
SYSTEM_TEMPLATE = "You are a specialized systematic-review screener. Follow your specific instructions precisely. Output MUST be valid JSON.".strip()

SAMPLE_GENERATOR_PROMPT = """
You are an expert in scientific writing. Your task is to create high-quality, synthetic few-shot examples for training other AI agents.
You will be given real articles with ground truth labels. You MUST rewrite the title and abstract to be conceptually similar but phrased differently.
Based on the `ground_truths` label, write a concise `reasoning` that clearly explains the decision, referencing the screening criteria.
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
1.  Make a "decision" which MUST be one of the exact strings: "include", "read full-text to decide", or "exclude".
2.  Provide a detailed "explanation" justifying your choice by citing specific criteria.
3.  Provide a "confidence_score" from 1 (very unsure) to 5 (very sure). Justify your score in the explanation.
Return a single JSON object with these three keys.
"""

ADJUDICATOR_PROMPT = """
You are the Adjudicator, a senior researcher leading a screening committee. Your final decision MUST be 'include' or 'exclude'.
You have received three reviews, each with a confidence score (1=low, 5=high):
- Reviewer 1 (High-Recall): {r1}
- Reviewer 2 (High-Precision): {r2}
- Reviewer 3 (Balanced): {r3}
YOUR ADJUDICATION TASK:
Follow this structured reasoning to make a balanced and robust decision.
1.  **Analyze Confidence and Arguments**: Look at both the decision and the confidence score from each reviewer.
2.  **Apply a Weighted Decision Rule**:
    - **High-Confidence Exclusion**: If the High-Precision reviewer gives a confidence score of 4 or 5 for 'exclude' AND their explanation cites a clear violation of an exclusion criterion, your final decision should be **'exclude'**.
    - **High-Confidence Inclusion**: If the High-Recall reviewer gives a confidence score of 4 or 5 for 'include' AND the High-Precision reviewer's reason for exclusion has low confidence (1-2), your final decision should be **'include'**.
    - **Default to Safety in Conflict/Ambiguity**: In all other cases of disagreement or ambiguity, default to **'include'**.
3.  **Add Notes for a Safe-Default Decision**: If your final decision is 'include' because you defaulted to safety, you MUST add a `notes` field with the exact value "need full-text to double-check".
4.  **Explain Your Final Judgment**: Provide a brief `explanation` for your decision.
Return your final judgment as a JSON object, ensuring the primary decision key is "final_decision".
"""

class Agent:
    def __init__(self, name: str, instructions: str, output_type: Type[BaseModel], model: str = "gpt-5"):
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
            print(f"Error during API call or parsing for agent {self.name}: {e}")
            # Provide a fallback error object that matches the expected schema
            if self.output_type == ReviewResult:
                return ReviewResult(decision="exclude", confidence_score=1, explanation=f"Error: {e}")
            elif self.output_type == FinalDecision:
                return FinalDecision(final_decision="exclude", explanation=f"Error: {e}", notes="Processing error")
            elif self.output_type == GeneratedSamples:
                return GeneratedSamples(include_samples=[], exclude_samples=[])
            return None


# Agent Handlers
agent_sample_generator = Agent(name="sample_generator", instructions=SYSTEM_TEMPLATE, output_type=GeneratedSamples)
agent_inclusion = Agent(name="reviewer_inclusion", instructions=SYSTEM_TEMPLATE, output_type=ReviewResult)
agent_exclusion = Agent(name="reviewer_exclusion", instructions=SYSTEM_TEMPLATE, output_type=ReviewResult)
agent_balanced = Agent(name="reviewer_balanced", instructions=SYSTEM_TEMPLATE, output_type=ReviewResult)
agent_adjudicator = Agent(name="adjudicator", instructions=SYSTEM_TEMPLATE, output_type=FinalDecision)

def format_samples(title: str, samples: List[FewShotExample]) -> str:
    if not samples: return ""
    header = f"--- {title} EXAMPLES ---\n"
    formatted = [f"Title: {s.title}\nAbstract: {s.abstract}\nReasoning: {s.reasoning}\n" for s in samples]
    return header + "\n".join(formatted) + "-------------------\n\n"

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

async def run_full_screening_process(
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
            await progress_callback("⚠未找到足够数据生成样本，将以零样本模式运行。")
    else:
        await progress_callback("未发现'ground_truths'列，将以零样本模式运行。")

    df_to_screen = df_articles.copy()
    for i in range(1, 4):
        df_to_screen[f"rev{i}_decision"], df_to_screen[f"rev{i}_confidence"], df_to_screen[f"rev{i}_explanation"] = "", None, ""
    df_to_screen["final_decision"], df_to_screen["final_explanation"], df_to_screen["notes"] = "", "", ""

    total_rows = len(df_to_screen)
    for idx, row in df_to_screen.iterrows():
        await progress_callback(f"正在处理第 {idx + 1}/{total_rows} 篇文章...")

        article = row.to_dict()
        p_inclusion = ("'high-recall' screener", "Your goal is to avoid incorrectly excluding studies. Lean towards 'include' or 'read full-text to decide'.")
        p_exclusion = ("'high-precision' screener", "Your goal is to filter out irrelevant studies. Scrutinize against Exclusion Criteria first.")
        p_balanced = ("neutral, balanced screener", "Make the most objective judgment possible.")

        task1 = call_specialized_reviewer(agent_inclusion, p_inclusion[0], p_inclusion[1], criteria, article, include_examples_str, exclude_examples_str)
        task2 = call_specialized_reviewer(agent_exclusion, p_exclusion[0], p_exclusion[1], criteria, article, include_examples_str, exclude_examples_str)
        task3 = call_specialized_reviewer(agent_balanced, p_balanced[0], p_balanced[1], criteria, article, include_examples_str, exclude_examples_str)

        r1, r2, r3 = await asyncio.gather(task1, task2, task3)
        final = await call_adjudicator(r1, r2, r3)

        df_to_screen.loc[idx, "rev1_decision"] = r1.decision
        df_to_screen.loc[idx, "rev1_confidence"] = r1.confidence_score
        df_to_screen.loc[idx, "rev1_explanation"] = r1.explanation

        df_to_screen.loc[idx, "rev2_decision"] = r2.decision
        df_to_screen.loc[idx, "rev2_confidence"] = r2.confidence_score
        df_to_screen.loc[idx, "rev2_explanation"] = r2.explanation

        df_to_screen.loc[idx, "rev3_decision"] = r3.decision
        df_to_screen.loc[idx, "rev3_confidence"] = r3.confidence_score
        df_to_screen.loc[idx, "rev3_explanation"] = r3.explanation

        df_to_screen.loc[idx, "final_decision"] = final.final_decision
        df_to_screen.loc[idx, "final_explanation"] = final.explanation
        df_to_screen.loc[idx, "notes"] = final.notes or ""

    return df_to_screen

