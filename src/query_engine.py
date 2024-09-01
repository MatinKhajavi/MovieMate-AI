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

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        summary_template: Optional[BasePromptTemplate] = None,
        streaming: bool = False
    ) -> None:
        """
        Initialize the EnhancedQueryEngine.

        :param retriever: The retriever to use for fetching relevant nodes.
        :type retriever: BaseRetriever
        :param llm: The language model to use for generating responses.
        :type llm: LLM
        :param text_qa_template: Template for text QA, defaults to None
        :type text_qa_template: Optional[BasePromptTemplate]
        :param refine_template: Template for refining answers, defaults to None
        :type refine_template: Optional[BasePromptTemplate]
        :param summary_template: Template for summarizing, defaults to None
        :type summary_template: Optional[BasePromptTemplate]
        :param streaming: Whether to enable streaming of responses, defaults to False
        :type streaming: bool
        """
        super().__init__()
        self._text_qa_template = text_qa_template
        self._refine_template = refine_template
        self._summary_template = summary_template
        self._retriever = retriever
        self._llm = llm
        self._streaming = streaming

        self._response_synthesizer = self._get_response_synthesizer()


    def _get_response_synthesizer(self) -> BaseSynthesizer:
        """
        Initialize and return the response synthesizer with provided prompts.

        :return: The configured response synthesizer.
        :rtype: BaseSynthesizer
        """
        return get_response_synthesizer(
            text_qa_template=self._text_qa_template,
            refine_template=self._refine_template,
            summary_template=self._summary_template,
            response_mode="tree_summarize",
            streaming=self._streaming
        )


    def custom_query(self, query_str: str) -> Response:
        """
        Execute a query using customized methods.

        :param query_str: The query string to process.
        :type query_str: str
        :return: The generated response to the query.
        :rtype: Response
        """
        nodes = self._retriever.retrieve(str_or_query_bundle=query_str)
        
        return self._response_synthesizer.synthesize(
            query=query_str,
            nodes=nodes,
            llm=self._llm
        )
    
    # TODO: Implement async version of custom_query method
    # async def acustom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
    #     pass

