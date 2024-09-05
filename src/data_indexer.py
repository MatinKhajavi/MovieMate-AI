import os
from typing import List
from pinecone import Pinecone, Index, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode

class DataIndexer:
    """
    A class to handle the embedding and indexing of data using a provided embedding model and Pinecone services.
    
    :param str dataset_name: The name of the dataset to be indexed.
    :param int embedding_dimension: The dimensionality of the embedding space.
    :param BaseEmbedding embed_model: The embedding model to use for generating embeddings.
    :param str pinecone_api_key: API key for accessing Pinecone's vector database services.
    :param str distance_metric: The distance metric to use for the Pinecone index (default: "euclidean").
    """
    
    def __init__(self, dataset_name: str, embedding_dimension: int, embed_model: BaseEmbedding, pinecone_api_key: str, distance_metric: str = "dotproduct"):
        self.dataset_name = dataset_name
        self.embedding_dimension = embedding_dimension
        self._embed_model = embed_model
        self._pinecone_api_key = pinecone_api_key
        self._distance_metric = distance_metric
        self._pinecone_client = Pinecone(api_key=self._pinecone_api_key)
        self._pinecone_index = self._setup_or_get_index(self._pinecone_client)
        self._vector_store = PineconeVectorStore(pinecone_index=self._pinecone_index)

    def _setup_or_get_index(self, pc: Pinecone) -> Index:
        """
        Check if an index exists and create one if not, using an existing Pinecone client.
        
        :param Pinecone pc: The initialized Pinecone client.
        :returns: A Pinecone Index connected to the newly created or existing index.
        """
        if self.dataset_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.dataset_name,
                dimension=self.embedding_dimension,
                metric=self._distance_metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        return pc.Index(self.dataset_name)

    def embed_nodes(self, nodes: List[TextNode]) -> None:
        """
        Embed nodes using the OpenAI model and set the embedding directly on each node.

        :param list[TextNode] nodes: A list of TextNode objects to be processed.
        """
        for node in nodes:
            embedding = self._embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = embedding

    def add_to_vector_store(self, nodes: List[TextNode]) -> None:
        """
        Add nodes, which include their embeddings, to the Pinecone vector store.

        :param list[TextNode] nodes: A list of TextNode objects that include embeddings.
        :raises ValueError: If embeddings are not set before adding to the vector store.
        """
        if any(node.embedding is None for node in nodes):
            raise ValueError("Embedding not set for one or more nodes. Please call embed_nodes first.")
        self._vector_store.add(nodes)
    
    def get_vector_store(self) -> PineconeVectorStore:
        """
        Get the Pinecone vector store.

        :returns: The Pinecone vector store.
        """
        return self._vector_store

