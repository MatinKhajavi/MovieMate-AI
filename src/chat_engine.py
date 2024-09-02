from typing import Optional, List, Any
from llama_index.core.chat_engine.types import ChatMode, BaseChatEngine
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms.utils import LLMType
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core import PromptTemplate


MOVIE_AGENT_PROMPT = PromptTemplate(
    """\
    You are MovieMate AI, an expert chat agent specializing in all aspects of cinema. Your role is to engage in conversations about movies, providing information, recommendations, and insights. Use your vast knowledge of films, directors, actors, genres, and cinematic history to assist users.

    Key responsibilities:
    1. Answer questions about movies, directors, actors, and film history.
    2. Offer personalized movie recommendations based on user preferences or specific scenarios.
    3. Analyze and discuss film themes, genres, and cinematic techniques.
    4. Share interesting movie trivia and behind-the-scenes information.
    5. Engage in in-depth conversations about various aspects of cinema.

    Guidelines:
    - Use the information retrieved from the query engine as your primary source to inform your responses.
    - If the retrieved information is insufficient or not available, rely on your general knowledge about movies.
    - Strive to provide accurate and relevant information about movies.
    - If you're unsure about something, admit it and offer to find more information if possible.
    - Tailor your responses to the user's level of film knowledge and interest.
    - Be enthusiastic and passionate about cinema in your interactions.
    - If asked questions not related to movies, politely redirect the conversation back to film-related topics.

    Remember: Your purpose is to be a knowledgeable and engaging companion for all things related to the world of movies. Prioritize the retrieved information when available, but don't hesitate to use your general knowledge when needed. Enjoy your cinematic conversations!
    """
)

CUSTOM_PROMPT = PromptTemplate(
    """\
    Given a conversation about movies (between Human and Assistant) and a follow-up message from Human, \
    rewrite the message to be a standalone question about movies that captures all relevant context \
    from the conversation. Include any specific movie titles, genres, actors, directors, or other film-related \
    details mentioned in the chat history that are relevant to the latest query. Ensure the standalone question \
    maintains focus on the most recent topic while incorporating pertinent information from earlier in the conversation.

    <Chat History>
    {chat_history}

    <Follow Up Message>
    {question}

    <Standalone question>
    """
)


def get_chat_engine(
    chat_mode: ChatMode,
    llm: LLMType,
    query_engine: Optional[BaseQueryEngine] = None,
    **kwargs: Any
) -> BaseChatEngine:
    """
    Get a chat engine based on the specified mode.

    Args:
        chat_mode (ChatMode): The desired chat mode.
        llm (LLMType): The language model to use.
        query_engine (Optional[BaseQueryEngine]): The query engine to use.
        **kwargs: Additional keyword arguments for specific chat engines.

    Returns:
        BaseChatEngine: The appropriate chat engine based on the specified mode.
    """

    if chat_mode in [ChatMode.BEST, ChatMode.REACT, ChatMode.OPENAI]:
        from llama_index.core.agent import AgentRunner

        if query_engine is None:
            raise ValueError("Query engine must be provided for BEST, REACT, or OPENAI mode")

        query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)

        return AgentRunner.from_llm(
            tools=[query_engine_tool],
            llm=llm,
            system_prompt=MOVIE_AGENT_PROMPT,
            **kwargs
        )
    
    elif chat_mode == ChatMode.CONDENSE_QUESTION:
        from llama_index.core.chat_engine import CondenseQuestionChatEngine
        
        if query_engine is None:
            raise ValueError("Query engine must be provided for CONDENSE_QUESTION mode")
        
        return CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            llm=llm,
            condense_question_prompt=CUSTOM_PROMPT,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported chat mode: {chat_mode}")
