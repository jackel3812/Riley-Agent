"""
RILEY - Embeddings and Document Retrieval
This module handles document embeddings and retrieval for RILEY's knowledge base.
"""

import numpy as np
from time import perf_counter

from jarvis.languagemodels.models import get_model, get_model_info


def embed(docs):
    """Compute embeddings for a batch of documents

    Embeddings are computed by running the first 512 tokens of each doc
    through a forward pass of the embedding model. The last hidden state
    is mean pooled to produce a single vector

    Documents will be processed in batches. The batch size is fixed at 64
    as this size was found to maximize throughput on a number of test
    systems while limiting memory usage.
    """

    tokenizer, model = get_model("embedding")
    model_info = get_model_info("embedding")

    start_time = perf_counter()

    tokens = [tokenizer.encode(doc[:8192]).ids[:512] for doc in docs]

    def mean_pool(last_hidden_state):
        embedding = np.mean(last_hidden_state, axis=0)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    bs = 64
    embeddings = []
    for i in range(0, len(docs), bs):
        outputs = model.forward_batch(tokens[i : i + bs])
        embeddings += [mean_pool(lhs) for lhs in np.array(outputs.last_hidden_state)]

    model_info["requests"] = model_info.get("requests", 0) + len(tokens)

    in_toks = sum(len(d) for d in tokens)
    model_info["input_tokens"] = model_info.get("input_tokens", 0) + in_toks

    runtime = perf_counter() - start_time
    model_info["runtime"] = model_info.get("runtime", 0) + runtime

    return embeddings


def search(query, docs, count=16):
    """Return `count` `docs` sorted by match against `query`

    :param query: Input to match in search
    :param docs: List of docs to search against
    :param count: Number of document to return
    :return: List of (doc_num, score) tuples sorted by score descending
    """

    prefix = get_model_info("embedding").get("query_prefix", "")

    query_embedding = embed([f"{prefix}{query}"])[0]

    scores = np.dot([d.embedding for d in docs], query_embedding)

    return [(i, scores[i]) for i in reversed(np.argsort(scores)[-count:])]


def get_token_ids(doc):
    """Return list of token ids for a document

    Note that the tokenzier used here is from the generative model.

    This is used for token counting for the context, not for tokenization
    before embedding.
    """

    generative_tokenizer, _ = get_model("instruct", tokenizer_only=True)

    # We need to disable and re-enable truncation here
    # This allows us to tokenize very large documents
    # We won't be feeding the tokens themselves to a model, so this
    # shouldn't cause any problems.
    trunk = generative_tokenizer.truncation
    if trunk:
        generative_tokenizer.no_truncation()
    ids = generative_tokenizer.encode(doc, add_special_tokens=False).ids
    if trunk:
        generative_tokenizer.enable_truncation(
            trunk["max_length"], stride=trunk["stride"], strategy=trunk["strategy"]
        )

    return ids


def chunk_doc(doc, name="", chunk_size=64, chunk_overlap=8):
    """Break a document into chunks

    :param doc: Document to chunk
    :param name: Optional document name
    :param chunk_size: Length of individual chunks in tokens
    :param chunk_overlap: Number of tokens to overlap when breaking chunks
    :return: List of strings representing the chunks
    """
    generative_tokenizer, _ = get_model("instruct", tokenizer_only=True)

    tokens = get_token_ids(doc)

    separator_tokens = [".", "!", "?", ").", "\n\n", "\n", '."']

    separators = [get_token_ids(t)[-1] for t in separator_tokens]

    name_tokens = []

    label = f"From {name} document:" if name else ""

    if name:
        name_tokens = get_token_ids(label)

    i = 0
    chunks = []
    chunk = name_tokens.copy()
    while i < len(tokens):
        token = tokens[i]
        chunk.append(token)
        i += 1

        # Save the last chunk if we're done
        if i == len(tokens):
            chunks.append(generative_tokenizer.decode(chunk))
            break

        if len(chunk) == chunk_size:
            # Backtrack to find a reasonable cut point
            for j in range(1, chunk_size // 2):
                if chunk[chunk_size - j] in separators:
                    ctx = generative_tokenizer.decode(
                        chunk[chunk_size - j : chunk_size - j + 2]
                    )
                    if " " in ctx or "\n" in ctx:
                        # Found a good separator
                        text = generative_tokenizer.decode(chunk[: chunk_size - j + 1])
                        chunks.append(text)
                        chunk = name_tokens + chunk[chunk_size - j + 1 :]
                        break
            else:
                # No semantically meaningful cutpoint found
                # Default to a hard cut
                text = generative_tokenizer.decode(chunk)
                chunks.append(text)
                # Share some overlap with next chunk
                overlap = max(
                    chunk_overlap, chunk_size - len(name_tokens) - (len(tokens) - i)
                )
                chunk = name_tokens + chunk[-overlap:]

    return chunks


class Document:
    """
    A document used for semantic search

    Documents have content and an embedding that is used to match the content
    against other semantically similar documents.
    """

    def __init__(self, content, name="", embedding=None):
        self.content = content
        self.embedding = embedding if embedding is not None else embed([content])[0]
        self.name = name


class RetrievalContext:
    """
    Provides a context for document retrieval

    Documents are embedded and cached for later search.
    """

    def __init__(self, chunk_size=64, chunk_overlap=8):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.clear()

    def clear(self):
        self.docs = []
        self.chunks = []

    def store(self, doc, name=""):
        """Stores a document along with embeddings

        This stores both the document as well as document chunks
        """
        if doc not in self.docs:
            self.docs.append(Document(doc))
            self.store_chunks(doc, name)

    def store_chunks(self, doc, name=""):
        chunks = chunk_doc(doc, name, self.chunk_size, self.chunk_overlap)

        embeddings = embed(chunks)

        for embedding, chunk in zip(embeddings, chunks):
            self.chunks.append(Document(chunk, embedding=embedding))

    def get_context(self, query, max_tokens=128):
        """Gets context matching a query

        Context is capped by token length and is retrieved from stored
        document chunks
        """

        if len(self.chunks) == 0:
            return None

        results = search(query, self.chunks)

        chunks = []
        tokens = 0

        for chunk_id, score in results:
            chunk = self.chunks[chunk_id].content
            chunk_tokens = len(get_token_ids(chunk))
            if tokens + chunk_tokens <= max_tokens and score > 0.1:
                chunks.append(chunk)
                tokens += chunk_tokens

        context = "\n\n".join(chunks)

        return context

    def get_match(self, query):
        if len(self.docs) == 0:
            return None

        return self.docs[search(query, self.docs)[0][0]].content