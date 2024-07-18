from typing import Any, List, Optional, Dict, Union
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.base.embeddings.base import BaseEmbedding



FilterValueType = Union[str, float, bool, List[str]]


class PineconeRetriever(BaseRetriever):
    """
    A custom retriever that leverages a Pinecone vector store for querying vectors based on the similarity of their embeddings,
    incorporating metadata filters to refine search results according to specific criteria.

    :param vector_store: The Pinecone vector store used for storing and retrieving vectors.
    :param embed_model: The model used to generate embeddings.
    :param query_mode: Mode of the query, defaults to "default".
    :param similarity_top_k: Number of top similar items to retrieve, defaults to 10.
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embed_model: Optional[BaseEmbedding] = None,
        query_mode: str = "default",
        similarity_top_k: int = 10
    ) -> None:
        """
        Initializes the PineconeRetriever with necessary components for executing a retrieval task.
        
        :param vector_store: The vector storage system where vectors are indexed.
        :param embed_model: The model used to generate embeddings.
        :param query_mode: The querying mode, defaults to 'default'.
        :param similarity_top_k: The number of top results to return, defaults to 10.
        """
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self._filters = {}
        super().__init__()

    def set_filters(self, filters: Dict[str, FilterValueType]) -> None:
        """
        Sets the metadata filter for querying the vector store.
        
        :param filter: Metadata key-value pairs used to refine the search.
        """
        self._filters = filters

    def _retrieve(
        self, 
        query_bundle: QueryBundle, 
    ) -> List[NodeWithScore]:
        """
        Private method to execute the core retrieval logic. This method is called by the public `retrieve` method,
        facilitating the retrieval of the most relevant nodes based on the provided query bundle and optional metadata filters.
        
        This method should not be called directly; instead, use the `retrieve` method which ensures the proper handling
        of additional preprocessing and postprocessing steps if necessary.

        :param query_bundle: The query and potential embedding provided by the user.
        :return: A list of nodes with their associated scores based on the similarity of their vectors.
        """

        if query_bundle.embedding is None:
            if self._embed_model is None:
                raise ValueError("Embedding model is not available to generate query embeddings.")
            query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        else:
            query_embedding = query_bundle.embedding

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )

        query_result = self._vector_store.query(vector_store_query, pinecone_query_filters=self._filters)
        nodes_with_scores = [
            NodeWithScore(node=node, score=query_result.similarities[i] if query_result.similarities else None)
            for i, node in enumerate(query_result.nodes)
        ]

        return nodes_with_scores
