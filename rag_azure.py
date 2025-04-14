# %%
import mlflow

mlflow.dspy.autolog()

# This is optional. Create an MLflow Experiment to store and organize your traces.
mlflow.set_experiment("DSPy-Elterngeld")

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
response = qa(question="Wie beantrage ich Elterngeld?")

print(response.response)

# %%
dspy.inspect_history(n=1)


# %%

cot = dspy.ChainOfThought("question -> response")
cot(question="Wie beantrage ich Elterngeld?")


# %%
import ujson


with open("azure_search/examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]

# %%
import random

data = [dspy.Example(**d).with_inputs("question") for d in data]
random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]

len(trainset), len(devset), len(testset)

# Let's pick an `example` here from the data.
example = data[0]
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
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType

search_service_key = os.environ["AZURE_SEARCH_API_KEY"]
search_service_endpoint = os.environ["AZURE_SEARCH_API_ENDPOINT"]
search_client = SearchClient(
    credential=AzureKeyCredential(search_service_key),
    endpoint=search_service_endpoint,
    index_name="gptkbindex",
)


def search_documents(keywords: str, result_count: int = 5):
    if result_count < 5:
        result_count = 5
    results = search_client.search(
        search_text=keywords,
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="default",
        top=result_count,
    )

    content = ""
    for result in results:
        content = (
            content
            + "["
            + result["sourcefile"]
            + "#"
            + result["sourcepage"]
            + "]:\n"
            + result["content"]
            + "\n\n"
        )
    return content


search_documents("elterngeld")


# %%
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = search_documents(question)
        return self.respond(context=context, question=question)


# %%
rag = RAG()
rag(question="Wie beantrage ich Elterngeld?")
# %%
dspy.inspect_history()
# %%
evaluate(RAG())


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

baseline = rag(question="Wie beantrage ich elterngeld?")
print(baseline.response)

# %%

pred = optimized_rag(question="Wie beantrage ich elterngeld?")
print(pred.response)

# %%

evaluate(optimized_rag)
