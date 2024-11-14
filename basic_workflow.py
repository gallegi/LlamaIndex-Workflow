import asyncio

from typing import List

from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeRelationship, NodeWithScore, QueryBundle
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context


from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)

reader = BeautifulSoupWebReader()
documents = reader.load_data(
        ["https://docs.anthropic.com/claude/docs/tool-use"]
    )
llm = Ollama(model="llama3.2:1b", request_timeout=60.0)

class RetrievalEvent(Event):
    nodes: List[NodeWithScore]

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
)
retriever = index.as_retriever(similarity_top_k=6)
    

class StandardRAGWorkflow(Workflow):
    DEFAULT_CONTEXT_PROMPT = (
        "Here is some context that may be relevant:\n"
        "-----\n"
        "{node_context}\n"
        "-----\n"
        "Please write a response to the following question, using the above context:\n"
        "{query_str}\n"
    ) 
    
    @step
    async def aretrieve(self, ctx: Context, ev: StartEvent) -> RetrievalEvent:
        # retrieve from context
        query_str = ev.query_str
        nodes = await retriever.aretrieve(query_str)
        result_ev = RetrievalEvent(nodes=nodes)
        await ctx.set("query_str", query_str)
        return result_ev
    
    def _prepare_query_with_context(
        self,
        query_str: str,
        nodes: List[NodeWithScore],
    ) -> str:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"Context Chunk {idx}:\n{node_text}\n\n"

        formatted_context = self.DEFAULT_CONTEXT_PROMPT.format(
            node_context=node_context, query_str=query_str
        )
        
        return formatted_context

    @step
    async def llm_response(self,  ctx: Context, retrieval_ev: RetrievalEvent) -> StopEvent:
        nodes = retrieval_ev.nodes
        query_str = await ctx.get("query_str")
        query_with_ctx = self._prepare_query_with_context(query_str, nodes)
        response = await llm.acomplete(query_with_ctx)
        return StopEvent(result=str(response))
    
async def main():
    workflow = StandardRAGWorkflow(timeout=60)
    result = await workflow.run(query_str="What is the definition of tool use?")
    print(result)
    draw_all_possible_flows(workflow, filename="flow.html")

if __name__ == "__main__":
    asyncio.run(main())