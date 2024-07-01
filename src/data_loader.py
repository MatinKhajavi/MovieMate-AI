import pandas as pd
from typing import List, Any, Dict
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

class DataLoader:
    def __init__(self, input_path: str, text_column: str, metadata_columns: List[str], chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Initializes the DataLoader with path to the CSV, the name of the text column,
        and the list of metadata column names, all treated as private variables. It also initializes
        the chunk size and overlap for the node parser.

        :param input_path: str, the path to the CSV file containing the data.
        :param text_column: str, the column name which contains the text for documents.
        :param metadata_columns: list of str, a list of column names to be used as metadata.
        :param chunk_size: int, the size of chunks for the node parser.
        :param chunk_overlap: int, the overlap between chunks for the node parser.
        """
        self._input_path = input_path
        self._text_column = text_column
        self._metadata_columns = metadata_columns
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified CSV file path, intended for internal use only.

        :return: DataFrame, loaded data from CSV.
        """
        return pd.read_csv(self._input_path)

    def _process_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Processes metadata for each row to ensure proper formatting and conversion.

        :param row: pd.Series, the data row from which metadata is extracted and processed.
        :return: Dict[str, Any], a dictionary containing processed metadata ready for use in creating Document objects.
        """
        metadata = {}
        for col in self._metadata_columns:
            if col == 'cast' and pd.notnull(row[col]):
                metadata[col] = row[col].split(',')[:15]
            elif col == 'release_date' and pd.notnull(row[col]):
                metadata[col] = pd.to_datetime(row[col]).date()
            elif col in ['spoken_languages', 'directors', 'genres', 'production_companies'] and pd.notnull(row[col]):
                metadata[col] = row[col].split(',')
            else:
                metadata[col] = row[col]
        return metadata

    def _create_documents(self, data: pd.DataFrame) -> List[Document]:
        """
        Creates a list of Document objects from a DataFrame, intended for internal use only.

        :param data: DataFrame, the data from which to create Documents.
        :return: list of Document objects.
        """
        documents = []
        for _, row in data.iterrows():
            metadata = self._process_metadata(row)
            document = Document(
                text=row[self._text_column],
                metadata=metadata,
                metadata_seperator=", ",
                text_template="Movie Metadata:\n {metadata_str}\n Plot Summary:\n {content}"
            )
            documents.append(document)
        return documents

    def _create_nodes(self, documents: List[Document]) -> List[TextNode]:
        """
        Parses documents into nodes using SentenceSplitter with specified chunk size and overlap,
        intended for internal use only.

        :param documents: list of Document objects.
        :return: list of TextNode objects derived from the documents.
        """
        parser = SentenceSplitter(chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        nodes = parser.get_nodes_from_documents(documents)
        return nodes

    def ingest_data(self) -> List[TextNode]:
        """
        Processes the CSV data by loading it, creating documents, and parsing those into nodes.
        This is the public method intended for external use.

        :return: list of TextNode objects.
        """
        data = self._load_data()
        documents = self._create_documents(data)
        nodes = self._create_nodes(documents)
        return nodes

