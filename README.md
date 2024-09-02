## MovieMate AI

### Introduction
MovieMate AI is a cutting-edge chat engine leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to answer questions about movies, offering personalized film suggestions, discussing cinematic history, analyzing themes and genres, and engaging in in-depth conversations about all aspects of cinema.

Whether you're looking for movie recommendations based on specific scenarios, seeking information about directors and actors, or wanting to explore film trivia, MovieMate AI is your knowledgeable companion for all things related to the world of movies.

### Demo
Check out the live demo of MovieMate AI at: [https://moviemate-ai.streamlit.app/](https://moviemate-ai.streamlit.app/)


### Key Features


### Technical Highlights

<!-- - **Embedding Model**: Utilizes OpenAI's text-embedding-3-large model for generating high-quality embeddings.
- **Language Model**: Powered by OpenAI's GPT-4o for natural language understanding and generation.
- **Data Indexing**: Custom DataIndexer class for efficient embedding and indexing of movie data.
- **Retrieval System**: PineconeRetriever class for sophisticated querying of the vector store with metadata filtering capabilities.
- **Chat Engine**: Flexible chat engine setup supporting various modes of interaction. -->


### Installation


### Usage



### Project Structure

- `src/`: Contains the core functionality of MovieMate AI
  - `chat_engine.py`: Configures the chat engine
  - `data_indexer.py`: Manages data embedding and indexing using Pinecone
  - `data_loader.py`: Handles loading and processing of movie data from CSV files
  - `pinecone_retriever.py`: Implements a custom retriever for the Pinecone vector store
  - `query_engine.py`: Defines the enhanced RAG query engine
- `scripts/`: Contains scripts for data processing and Pinecone setup
    - `data_collection.ipynb`: Contains code for collecting and processing data from IMDb and Wikipedia
    - `data_ingestion.ipynb`: Contains code for testing the DataLoader class
    - `data_indexing.ipynb`: Contains code for testing the DataIndexer class
    - `generate_embeddings_and_save.ipynb`: Contains code for generating embeddings and saving them to the Pinecone vector database.
- `streamlit.py`: Implements the Streamlit-based web interface for user interaction
- `README.md`: Provides project documentation and overview
- `requirements.txt`: Lists project dependencies


### Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License - see the LICENSE file for details.
