import sys
sys.path.insert(0, 'src')

from bank_chatbot.rag.pipeline import RAGPipeline

# Test retrieval only
p = RAGPipeline()
print('Retrieval test:')
result = p.vector_store.similarity_search('What is the funds availability policy?', k=3)
print(f'Docs: {len(result)}')
for d in result:
    title = d.metadata.get('title', 'Unknown')
    content = d.page_content[:100]
    print(f'  - {title}: {content}...')

print('\nTest queries:')
queries = [
    'What is the funds availability policy for check deposits?',
    'How do I report a lost debit card?',
    'What are the wire transfer fees?',
    'What is the overdraft fee?',
]

for query in queries:
    print(f'\nQuery: {query}')
    docs = p.vector_store.similarity_search(query, k=3)
    print(f'  Found {len(docs)} docs')
    for d in docs:
        title = d.metadata.get('title', 'Unknown')
        score = 1 - d.metadata.get('_distance', 0)
        print(f'  - {title} (score: {score:.2f})')