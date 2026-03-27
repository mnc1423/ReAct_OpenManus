from typing import List, Optional

from pydantic import Field

from app.llm import LLM
from app.logger import logger
from app.schema import Message
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch, SearchResult


class RAGTool(BaseTool):
    """Retrieval-Augmented Generation (RAG) tool."""

    name: str = "rag_qa"
    description: str = (
        "Answer a user query by retrieving relevant documents (web search or local docs) "
        "and then generating a grounded response with the LLM."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural language question to answer.",
            },
            "num_search_results": {
                "type": "integer",
                "description": "Number of search results to collect when using web search.",
                "default": 5,
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top documents to use in the generation prompt.",
                "default": 3,
            },
            "include_page_content": {
                "type": "boolean",
                "description": "Whether to fetch full web page content for each result.",
                "default": False,
            },
            "local_documents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional local documents to use instead of web search results.",
            },
        },
        "required": ["query"],
    }

    async def execute(
        self,
        query: str,
        num_search_results: int = 5,
        top_k: int = 3,
        include_page_content: bool = False,
        local_documents: Optional[List[str]] = None,
    ) -> ToolResult:
        """Execute the RAG workflow."""
        if not query.strip():
            return self.fail_response("query must not be empty")

        documents: List[str] = []
        sources: List[str] = []

        # Use local documents when provided, else perform web search
        if local_documents:
            documents = [doc.strip() for doc in local_documents if doc and doc.strip()]
            sources = [f"local_doc_{i+1}" for i in range(len(documents))]
            logger.info("Using local documents for RAG retrieval")
        else:
            logger.info("Performing web search for RAG retrieval")
            web_search = WebSearch()
            try:
                search_result = await web_search.execute(
                    query=query,
                    num_results=num_search_results,
                    fetch_content=include_page_content,
                )
            except Exception as e:
                logger.exception(f"Web search failed: {e}")
                return self.fail_response(f"Web search failed: {e}")

            if search_result.error:
                return self.fail_response(f"Web search error: {search_result.error}")

            if not search_result.results:
                return self.fail_response("No search results found for query")

            for result in search_result.results:
                title = result.title or result.url
                text = []
                if result.description:
                    text.append(f"Description: {result.description}")
                if include_page_content and result.raw_content:
                    text.append(f"Content: {result.raw_content}")
                documents.append("\n".join(text) if text else title)
                sources.append(result.url)

        if not documents:
            return self.fail_response("No documents available to answer the query")

        # simple relevance ranking by query-token overlap for local docs
        if local_documents:
            query_tokens = set(query.lower().split())

            def score_doc(doc_text: str) -> float:
                return len(query_tokens & set(doc_text.lower().split()))

            ranked = sorted(
                zip(documents, sources), key=lambda ds: score_doc(ds[0]), reverse=True
            )
            documents, sources = zip(*ranked) if ranked else (documents, sources)
            documents = list(documents)
            sources = list(sources)

        # choose top_k docs
        top_k = max(1, min(top_k, len(documents)))
        docs_for_prompt = documents[:top_k]
        source_for_prompt = sources[:top_k]

        context_text = []
        for i, (doc_text, src) in enumerate(zip(docs_for_prompt, source_for_prompt), start=1):
            context_text.append(
                f"[Document {i}] Source: {src}\n{doc_text if doc_text else 'No content available.'}"
            )

        dag_prompt = f"""
You are a retrieval-augmented generation assistant. Use only the provided documents to answer the question.
If the answer is not present in the documents, say you cannot find a definitive answer.
Cite the source for each fact in brackets, like [Document 1].

Question: {query}

Retrieved Documents:
{chr(10).join(context_text)}

Answer:"""

        llm = LLM()
        try:
            response = await llm.ask(
                messages=[Message.user_message(dag_prompt)],
                stream=False,
                temperature=0.0,
            )
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            return self.fail_response(f"LLM generation failed: {e}")

        output = (
            f"RAG Answer:\n{response}\n\n" f"Sources used: {', '.join(source_for_prompt)}"
        )

        return self.success_response(output)
