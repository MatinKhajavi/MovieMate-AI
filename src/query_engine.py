from typing import Optional
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.llms import LLM
from llama_index.core import Response
from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer


class EnhancedQueryEngine(CustomQueryEngine):
    """
    Enhanced RAG Query Engine.

    This query engine extends the basic Retrieval-Augmented Generation (RAG) approach by 
    implementing a custom query method for enhanced control over the RAG process.
    """

    retriever: BaseRetriever
    llm: LLM
    streaming: bool


    def _get_response_synthesizer(self) -> BaseSynthesizer:
        """
        Initialize and return the response synthesizer with provided prompts.

        :return: The configured response synthesizer.
        :rtype: BaseSynthesizer
        """
        return get_response_synthesizer(
            llm=self.llm,
            response_mode="tree_summarize",
            streaming=self.streaming,
        )


    def custom_query(self, query_str: str) -> Response:
        """
        Execute a query using customized methods.

        :param query_str: The query string to process.
        :type query_str: str
        :return: The generated response to the query.
        :rtype: Response
        """
        nodes = self.retriever.retrieve(str_or_query_bundle=query_str)
        response_synthesizer = self._get_response_synthesizer()
        return response_synthesizer.synthesize(
            query=query_str,
            nodes=nodes,
        )
    
    # TODO: Implement async version of custom_query method
    # async def acustom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
    #     pass

