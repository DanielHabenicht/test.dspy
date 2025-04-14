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
    "azure/gpt-4o-mini",
    api_base="https://dspy-dnhb-test.openai.azure.com",
    api_key=os.environ["AZURE_API_KEY"],
    api_version="2025-01-01-preview",
    temperature=1.0,
    max_tokens=5000,
)
dspy.configure(lm=lm)

# %%
qa = dspy.Predict("question: str -> response: str")
response = qa(question="what are high memory and low memory on linux?")

print(response.response)

# %%
dspy.inspect_history(n=1)


# %%

cot = dspy.ChainOfThought("question -> response")
cot(question="should curly braces appear on their own line?")


# %%
import ujson
from dspy.utils import download

# Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
download(
    "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl"
)

with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]

# %%
data = [dspy.Example(**d).with_inputs("question") for d in data]

# Let's pick an `example` here from the data.
example = data[2]
example

# %%
from dspy.evaluate import SemanticF1

# Instantiate the metric.
metric = SemanticF1(decompositional=True)

# Produce a prediction from our `cot` module, using the `example` above as input.
pred = cot(**example.inputs())

# Compute the metric score for the prediction.
score = metric(example, pred)

print(f"Question: \t {example.question}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {pred.response}\n")
print(f"Semantic F1 Score: {score:.2f}")

# %%
dspy.inspect_history(n=1)
# %%
# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(
    devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=2
)

# Evaluate the Chain-of-Thought program.
evaluate(cot)
# %%
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")

# %%
max_characters = 6000  # for truncating >99th percentile of documents
topk_docs_to_retrieve = 5  # number of documents to retrieve per search query

with open("ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [ujson.loads(line)["text"][:max_characters] for line in f]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")

embedder = dspy.Embedder(
    "azure/text-embedding-3-small",
    api_base="https://dspy-dnhb-test.openai.azure.com",
    api_key=os.environ["AZURE_API_KEY_EMBEDDING"],
    api_version="2023-05-15",
    dimensions=512,
)

search = dspy.retrievers.Embeddings(
    embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve
)


# %%
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)


# %%
rag = RAG()
rag(question="what are high memory and low memory on linux?")
# %%
dspy.inspect_history()
# %%
evaluate(RAG())

# %%
import random

random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]

len(trainset), len(devset), len(testset)

# %%
tp = dspy.MIPROv2(
    metric=metric, auto="medium", num_threads=24
)  # use fewer threads if your rate limit is small

optimized_rag = tp.compile(
    RAG(),
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    requires_permission_to_run=False,
)
optimized_rag.save("optimized_rag.json")

# %%

baseline = rag(question="cmd+tab does not work on hidden or minimized windows")
print(baseline.response)

# %%

pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
print(pred.response)

# %%

evaluate(optimized_rag)
