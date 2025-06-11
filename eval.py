import json

import dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageSystem,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import JSONSchema

dotenv.load_dotenv()


with open("writer_summaries.json", "r") as f:
    summaries = json.load(f)

DATASET = [Sample(input=s["article"], target=s["summary"]) for s in summaries]
DATASET = DATASET[:1]  # Remove this


@solver
def simple_summarizer() -> Solver:
    INSTRUCTION_PROMPT = """Summarize the following article in 3 sentences"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.insert(0, ChatMessageSystem(content=INSTRUCTION_PROMPT))

        state = await generate(state)
        return state

    return solve


# TODO: make other scorers for L1, etc. ?
@scorer(metrics=[accuracy()])
def compare_generated_with_target():
    """Compares the generated summary w/ the 'ground truth' freelance writer generated summary"""
    # Configure to return an explanation + score
    config = GenerateConfig(
        response_schema=ResponseSchema(
            name="score_and_explanation",
            description="The given score and its explanation",
            json_schema=JSONSchema(
                type="object",
                properties={
                    "score": JSONSchema(type="integer"),
                    "explanation": JSONSchema(type="string"),
                },
            ),
        )
    )

    # Prompt to grade.
    PROMPT = """You are an expert grader. You are discerning and sharp. 
Grade the new summary on how similar it is to the reference. 
Some ideas for grading 
- does it capture the same main points? 
- Is it similar, e.g. could one infer the same details from both?

Return a score from 1-5, and an explanation for the score

<New Summary>
{new_summary}
<End New Summary>

<Reference Summary>
{reference_summary}
<End Reference Summary>
"""

    async def score(state: TaskState, target: Target):
        new_summary = state.output.completion
        reference = target.text
        grader_model = get_model("openai/gpt-4o", config=config)

        response = await grader_model.generate(
            PROMPT.format(new_summary=new_summary, reference_summary=reference)
        )

        rj = json.loads(response.completion)

        if not {"score", "explanation"}.issubset(rj):
            return Score(value="Err", answer="Mistake buddy")

        return Score(
            value=rj["score"], answer=new_summary, explanation=rj["explanation"]
        )

    return score


@task
def test_summarizers() -> Task:
    return Task(
        dataset=DATASET,
        solver=simple_summarizer(),
        scorer=compare_generated_with_target(),
    )


if __name__ == "__main__":
    eval(test_summarizers(), model="openai/gpt-4o")
