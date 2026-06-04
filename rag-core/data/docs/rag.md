# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that combines a language
model with an external knowledge source. Instead of relying solely on the
parameters learned during training, a RAG system retrieves relevant documents at
query time and conditions the model's generation on them. This grounds the
output in specific, up-to-date sources and reduces hallucination.

## The pipeline

A typical RAG pipeline has two phases. In the offline indexing phase, source
documents are split into chunks, each chunk is converted into a vector embedding,
and the embeddings are stored in a vector database. In the online query phase,
the user's question is embedded with the same model, the most similar chunks are
retrieved by vector search, and those chunks are inserted into the language
model's prompt as context for generating an answer.

## Chunking

How documents are split into chunks strongly affects retrieval quality. Chunks
that are too large dilute the embedding and waste the model's context window;
chunks that are too small lose the surrounding meaning needed to answer a
question. A common approach is fixed-size chunks measured in tokens, with a small
overlap between consecutive chunks so that information spanning a boundary
remains retrievable.

## Hybrid search and re-ranking

Dense vector search captures semantic similarity but can miss exact terms such as
names, codes, or acronyms. Keyword search such as BM25 captures those exact
matches but misses paraphrase. Hybrid search runs both and fuses the rankings,
combining their strengths. A further improvement is re-ranking: a cross-encoder
scores each retrieved candidate against the query jointly, which is more accurate
than the first-stage retrievers but too slow to run over an entire corpus, so it
is applied only to the shortlist of candidates.

## Evaluation

RAG systems are evaluated on two fronts. Retrieval quality asks whether the right
chunks were fetched, measured with metrics like hit rate, mean reciprocal rank,
and recall at k. Generation quality asks whether the final answer is faithful to
the retrieved context and actually answers the question. Measuring both is
essential, because a fluent answer built on the wrong context is still wrong.
