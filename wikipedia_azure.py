# %%

import mlflow

mlflow.dspy.autolog()

# This is optional. Create an MLflow Experiment to store and organize your traces.
mlflow.set_experiment("DSPy")

# %%
import os
import dspy
from dotenv import load_dotenv

load_dotenv()

# You can use any model: https://docs.litellm.ai/docs/providers
lm = dspy.LM(
    "azure/o3-mini",
    api_base="https://dspy-dnhb-test.openai.azure.com",
    api_key=os.environ["AZURE_API_KEY"],
    api_version="2025-01-01-preview",
    temperature=1.0,
    max_tokens=5000,
)

lm = dspy.LM(
    "azure/gpt-4o-mini",
    api_base="https://dspy-dnhb-test.openai.azure.com",
    api_key=os.environ["AZURE_API_KEY"],
    api_version="2025-01-01-preview",
    temperature=1.0,
    max_tokens=5000,
)
dspy.configure(lm=lm)

# %%
math = dspy.ChainOfThought("question -> answer: float")
math(question="Two dice are tossed. What is the probability that the sum equals two?")

# %%
from dspy.datasets import HotPotQA

HotPotQA(train_seed=2024, train_size=500)


# %%

import dspy
from dspy.datasets import HotPotQA


def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(
        query, k=3
    )
    return [x["text"] for x in results]


trainset = [
    x.with_inputs("question") for x in HotPotQA(train_seed=2024, train_size=500).train
]
react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
optimized_react = tp.compile(react, trainset=trainset)


# %%
react.save("baseline.json")

optimized_react.save("optimized.json")
optimized_react.save("./dspy_program/", save_program=True)
# %%

cost = sum(
    [x["cost"] for x in lm.history if x["cost"] is not None]
)  # in USD, as calculated by LiteLLM for certain providers
# %%
