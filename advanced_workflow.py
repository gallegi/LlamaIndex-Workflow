import asyncio
from typing import List

from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows

# Initialize the web reader and load documents
reader = BeautifulSoupWebReader()
documents = reader.load_data(["https://docs.llamaindex.ai/en/stable/getting_started/customization/"])

def convert_message_list_to_str(messages):
    """
    Convert a list of ChatMessage objects to a single string.
    """
    return "\n".join([f"{message.role}: {message.content}" for message in messages])

class CondenseQueryEvent(Event):
    condensed_query_str: str

class RetrievalEvent(Event):
    nodes: List[NodeWithScore]

class AdvancedRAGWorkflow(Workflow):
    SUMMARY_TEMPLATE = (
        "Given our current conversation:\n"
        "{chat_history_str}"
        "\nSummarize what user and assistant have been talking:\n"
    )
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
        self.retriever = index.as_retriever(similarity_top_k=5)
        self.llm = Ollama(model="llama3.2:1b", request_timeout=60.0)
        self.reranker = ColbertRerank(top_n=2)

    @step
    async def condense_history_to_query(self, ctx: Context, ev: StartEvent) -> CondenseQueryEvent:
        """
        Condense chat history and query into a single query string.
        """
        query_str = ev.query_str
        chat_history = ev.chat_history
        await ctx.set("query_str", query_str)
        await ctx.set("chat_history", chat_history)

        formated_query = ""
        if len(chat_history) > 0:
            chat_history_str = convert_message_list_to_str(chat_history)
            formated_query = self.SUMMARY_TEMPLATE.format(chat_history_str=chat_history_str)
            history_summary = await self.llm.acomplete(formated_query)
            condensed_query = "Context:" + history_summary.text + "\nQuestion:" + query_str
        else:
            condensed_query = query_str

        return CondenseQueryEvent(condensed_query_str=condensed_query)
    
    @step
    async def retrieve(self, ev: CondenseQueryEvent) -> RetrievalEvent:
        """
        Retrieve relevant nodes based on the condensed query string.
        """
        condensed_query_str = ev.condensed_query_str
        nodes = await self.retriever.aretrieve(condensed_query_str)
        return RetrievalEvent(nodes=nodes)
    
    def _prepare_query_with_context(self, query_str: str, nodes: List[NodeWithScore]) -> str:
        """
        Prepare the query string with context from retrieved nodes.
        """
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"Context Chunk {idx}:\n{node_text}\n\n"

        formatted_context = self.DEFAULT_CONTEXT_PROMPT.format(
            node_context=node_context, query_str=query_str
        )
        
        return formatted_context

    @step
    async def llm_response(self, ctx: Context, retrieval_ev: RetrievalEvent) -> StopEvent:
        """
        Generate a response from the LLM based on the query and context.
        """
        nodes = retrieval_ev.nodes
        query_str = await ctx.get("query_str")
        query_with_ctx = self._prepare_query_with_context(query_str, nodes)
        chat_history = await ctx.get("chat_history")
        response = await self.llm.astream_chat(chat_history + [ChatMessage(role=MessageRole.USER, content=query_with_ctx)])
        chat_history.append(ChatMessage(role=MessageRole.USER, content=query_str))
        return StopEvent(result=response)
    
async def main():
    """
    Main function to run the AdvancedRAGWorkflow.
    """
    workflow = AdvancedRAGWorkflow(documents, timeout=60)
    draw_all_possible_flows(workflow, filename="advanced_flow.html")
    chat_history = []
    while True:
        query_str = input("User: ")
        if query_str == "exit":
            break
        handler = workflow.run(query_str=query_str, chat_history=chat_history)
        response = await handler
        chat_history = await handler.ctx.get("chat_history")
        full_response_str = ""
        print("\nAssistant:")
        async for res in response:
            print(res.delta, end="", flush=True)
            full_response_str += res.delta
        chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=full_response_str))
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())