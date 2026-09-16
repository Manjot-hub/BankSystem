from bank_chatbot.rag.pipeline import RAGPipeline


def test_rag_retrieves_funds_availability_policy():
    pipeline = RAGPipeline()
    docs = pipeline.vector_store.similarity_search("What is the funds availability policy?", k=3)

    assert docs
    assert docs[0].metadata["title"] == "Funds Availability Policy"


def test_rag_retrieves_lost_card_faq():
    pipeline = RAGPipeline()
    docs = pipeline.vector_store.similarity_search("How do I report a lost debit card?", k=3)

    assert docs
    assert docs[0].metadata["title"] == "How do I report a lost or stolen debit card?"


def test_rag_invoke_returns_citations_in_retrieval_mode():
    pipeline = RAGPipeline()
    result = pipeline.invoke("What is the overdraft fee?")

    assert result["retrieved_count"] > 0
    assert result["citations"]
    assert result["confidence"] > 0
