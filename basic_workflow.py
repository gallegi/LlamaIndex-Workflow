import asyncio
import time
from typing import List

from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows

# Load documents using BeautifulSoupWebReader
reader = BeautifulSoupWebReader()
documents = reader.load_data(["https://docs.llamaindex.ai/en/stable/getting_started/customization/"])

class RetrievalEvent(Event):
    nodes: List[NodeWithScore]

class StandardRAGWorkflow(Workflow):
    DEFAULT_CONTEXT_PROMPT = (
        "Here is some context that may be relevant:\n"
        "-----\n"
        "{node_context}\n"
        "-----\n"
        "Please write a response to the following question, using the above context:\n"
        "{query_str}\n"
    )

    def __init__(self, documents, timeout: int = 60, verbose: bool = False):
        super().__init__(timeout=timeout, verbose=verbose)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        )
        self.retriever = index.as_retriever(similarity_top_k=1)
        self.llm = Ollama(model="llama3.2:1b", request_timeout=60.0)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrievalEvent:
        query_str = ev.query_str
        nodes = await self.retriever.aretrieve(query_str)
        await ctx.set("query_str", query_str)
        return RetrievalEvent(nodes=nodes)

    def _prepare_query_with_context(self, query_str: str, nodes: List[NodeWithScore]) -> str:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"Context Chunk {idx}:\n{node_text}\n\n"
        return self.DEFAULT_CONTEXT_PROMPT.format(node_context=node_context, query_str=query_str)

    @step
    async def llm_response(self, ctx: Context, retrieval_ev: RetrievalEvent) -> StopEvent:
        nodes = retrieval_ev.nodes
        query_str = await ctx.get("query_str")
        query_with_ctx = self._prepare_query_with_context(query_str, nodes)
        response = await self.llm.astream_complete(query_with_ctx)
        return StopEvent(result=response)

async def main():
    workflow = StandardRAGWorkflow(documents, timeout=60)
    draw_all_possible_flows(workflow, filename="basic_flow.html")
    while True:
        query_str = input("Ask a question: ")  # Example: How to parse my documents into smaller chunk?
        if query_str == "exit":
            break
        response = await workflow.run(query_str=query_str)
        print("\nAssistant:")
        async for res in response:
            print(res.delta, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())