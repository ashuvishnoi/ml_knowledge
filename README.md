# Data Scientist Interview Revision Guide
### GenAI, Agentic AI, NLP, ML — Concept Refresher

---

## HOW TO USE THIS (given limited time)
Priority order if you're truly short on time: **RAG (incl. cross-encoders & vector DB internals) → Encoder/BERT/GPT architecture → Prompt Engineering → Fine-tuning → Evaluation → System design case studies → Agentic AI → classic ML**. GenAI-heavy roles get asked about the first few disproportionately more, and system-design case studies are where candidates most often freeze up if they haven't practiced the framework.

---

## 1. CLASSIC ML — QUICK HITS

**Bias-Variance Tradeoff**: High bias = underfitting (too simple, misses patterns). High variance = overfitting (memorizes noise). Total error = Bias² + Variance + Irreducible error.

**Regularization**: L1 (Lasso) shrinks some weights to exactly zero → feature selection. L2 (Ridge) shrinks weights smoothly, keeps all features. Dropout (NNs) randomly zeroes activations during training to prevent co-adaptation.

**Evaluation metrics**:
- **Precision** = TP/(TP+FP) — "of predicted positives, how many were right"
- **Recall** = TP/(TP+FN) — "of actual positives, how many did we catch"
- **F1** = harmonic mean of precision/recall — use when classes imbalanced
- **ROC-AUC**: measures ranking quality across thresholds; less useful with heavy class imbalance (use PR-AUC instead there)

**Bagging vs Boosting**: Bagging (Random Forest) trains models in parallel on bootstrapped samples → reduces variance. Boosting (XGBoost, LightGBM) trains sequentially, each model corrects previous errors → reduces bias, more prone to overfitting.

**Cross-validation**: k-fold splits data k times to get robust performance estimate; stratified k-fold preserves class ratios.

**Handling imbalanced data**: resampling (SMOTE, undersampling), class weights, threshold tuning, appropriate metrics (F1/PR-AUC not accuracy).

**Common "explain X" targets**: gradient descent (iteratively move opposite the gradient to minimize loss), logistic regression (linear model + sigmoid, outputs probability), decision trees (splits on info gain/gini), k-means (iteratively assign points to nearest centroid, recompute centroids).

---

## 2. NLP FOUNDATIONS

**Tokenization**: Word-level (large vocab, OOV issues) → Subword is standard now:
- **BPE (Byte Pair Encoding)**: merges most frequent character pairs iteratively (GPT family)
- **WordPiece**: similar, likelihood-based merging (BERT)
- **SentencePiece**: language-agnostic, handles whitespace as a token (T5, LLaMA)

**Embeddings**:
- **Word2Vec/GloVe**: static, one vector per word regardless of context
- **Contextual embeddings (BERT, etc.)**: vector changes based on surrounding context — solves polysemy (e.g., "bank" river vs money)

**Transformer architecture** (the thing everything below is built on):
- **Self-attention**: each token computes a weighted sum over all other tokens based on relevance (Query·Key similarity → softmax → weight Values)
- **Multi-head attention**: run several attention operations in parallel with different learned projections, capturing different relationship types
- **Positional encoding**: since attention has no inherent order sense, inject position info (sinusoidal or learned)
- **Encoder-only (BERT)**: bidirectional context, good for understanding tasks (classification, NER)
- **Decoder-only (GPT family)**: causal/masked attention (only sees past tokens), good for generation
- **Encoder-decoder (T5, BART)**: good for seq2seq tasks (translation, summarization)

**Pretraining objectives**:
- **MLM (Masked Language Modeling)**: mask random tokens, predict them — BERT
- **CLM (Causal Language Modeling)**: predict next token given previous ones — GPT
- **Span corruption**: mask contiguous spans — T5

---

## 3. ENCODER vs BERT vs GPT — ARCHITECTURE DEEP DIVE (PRACTICAL)

This is where interviewers test if you actually understand the building blocks vs just naming models. Go slow on this one.

### The generic Transformer encoder block
Each encoder layer = **bidirectional multi-head self-attention** (every token can attend to every other token, left *and* right) → residual + LayerNorm → **feed-forward network (FFN)** → residual + LayerNorm. Stack N of these. Input = token embeddings + positional embeddings. Output = **one contextual vector per input token**, same sequence length as input. That's it — "encoder" is just this reusable block; it says nothing about how it was trained or what it's used for.

### BERT = a specific pretrained model built from encoder blocks
BERT takes that generic encoder block and adds:
- **Input scheme**: `[CLS] sentence A [SEP] sentence B [SEP]` + **segment embeddings** (which sentence a token belongs to, A or B) on top of token + positional embeddings
- **Pretraining objectives**: **MLM** (mask ~15% of tokens, predict them using bidirectional context) + originally **NSP** (Next Sentence Prediction — predict if sentence B really follows sentence A; later models like RoBERTa dropped NSP, found it didn't help much)
- **Tokenizer**: WordPiece, fixed vocab learned from a large corpus
- Sizes: BERT-base = 12 encoder layers, BERT-large = 24

**So "difference between a plain encoder and BERT"** → the encoder is the architecture (attention + FFN blocks, generic, reusable, could even be trained from scratch for any task); BERT is *one specific instance* of stacking that architecture with a specific input format, specific pretraining objective, specific tokenizer/vocab, and specific weights learned from massive pretraining. Every BERT is a Transformer encoder stack, but not every Transformer encoder stack is "BERT" — e.g., the encoder half of T5 or the original Transformer is architecturally near-identical but trained differently (jointly with a decoder, different objective, e.g. span corruption) and isn't called BERT.

### What is the actual *output* of BERT?
Raw output = a sequence of contextual vectors, one per input token (e.g., 768-dim for base). What you *do* with that raw output depends on the task — this is the practical bit interviewers probe:
- **Sentence-level classification** (sentiment, NLI): take the **[CLS] token's** final vector → feed into a small classifier head (linear + softmax)
- **Token-level classification** (NER, POS tagging): take **every token's** vector → linear+softmax per token
- **Extractive QA**: take every token's vector → two separate linear layers predict "start position" and "end position" logits over the passage tokens
- **Sentence embeddings for search/RAG**: pool token vectors (mean-pooling, or [CLS]) — but a **heads-up interviewers like to hear**: raw/vanilla BERT embeddings are *not* great for cosine-similarity search out of the box, because MLM/NSP pretraining doesn't optimize the vector space for that. This is exactly why **Sentence-BERT (SBERT)** exists — it fine-tunes BERT with a siamese/contrastive objective specifically so [CLS]/pooled embeddings become meaningful for similarity — this is what most embedding models used for RAG retrieval are actually built on.
- **What BERT *cannot* do natively**: open-ended text generation. It has no autoregressive/generation head or causal masking — it's bidirectional, so it can't naturally generate left-to-right. (This is exactly why BART bolts a GPT-style causal decoder onto a BERT-style encoder to get a generation-capable model.)

### GPT (decoder-only) architecture
Each GPT layer = **masked/causal self-attention** (each token can only attend to *itself and previous* tokens — enforced via an upper-triangular mask blocking attention to future positions) → residual + LayerNorm → FFN → residual + LayerNorm. Output at each position = logits over the vocabulary for the *next* token (final linear layer, often weight-tied with the input embedding matrix). Trained purely with **CLM** (predict next token) — fully unsupervised, no labels needed, just raw text.

Key structural point interviewers want to hear: a decoder-only block is basically the same self-attention + FFN pattern as an encoder block, **with one difference — the attention mask**. Encoder = no mask (bidirectional). Decoder-only (GPT) = causal mask (unidirectional). Note also: a "decoder" in a full encoder-decoder model (like the original Transformer/T5) additionally has a **cross-attention** sub-layer that attends into the encoder's output — GPT has *no* such cross-attention layer since there's no separate encoder; it's pure self-attention throughout.

### Practical architecture questions (common in interviews)

**Q: Can we use a decoder-only model (GPT-style) for summarization? What would need to change?**
Yes — and in practice this is exactly what modern LLM-based summarization does (GPT-4, Claude, etc. summarize purely as decoder-only models). No fundamental new component is *required*: you concatenate `[source document] + [instruction] + [target summary]` as one sequence and train/prompt with the normal causal-LM setup. This works because causal masking only blocks attention to the *future* — by the time the model is generating summary tokens, the entire source document is already "in the past," so it gets fully attended to (just not bidirectionally *within itself*).
The real limitation vs. a purpose-built encoder-decoder (BART/T5/Pegasus): the source document only ever gets *unidirectional* self-attention (each source token only sees earlier source tokens, not later ones), whereas a true encoder bidirectionally digests the whole source first. For small/specialized models this mattered a lot (why BART/T5 were designed with an encoder); at LLM scale it mostly stops mattering because scale compensates.
**If you specifically wanted to fix this without building a full encoder-decoder**: change the **attention mask**, not the architecture — use a **Prefix-LM mask** (as in UniLM/GLM/PrefixLM): make attention **bidirectional within the source/prefix segment**, and keep it **causal only for the generated/target segment**. This gives the source document encoder-like bidirectional treatment while keeping a single unified stack — a nice middle-ground answer that shows you understand attention masks are the real lever, not "swap the architecture."
The other real option if you want an actual encoder-decoder: add **cross-attention layers** in a separate decoder that attend into a separately-encoded (bidirectional) representation of the source — that's just describing how T5/BART work.

**Q: Encoder-decoder vs decoder-only for seq2seq tasks — which and why?**
Encoder-decoder (T5, BART) tends to be more parameter-efficient for well-defined seq2seq tasks (translation, summarization) at smaller scale, because the encoder specializes purely in *understanding* the source and the decoder specializes purely in *generating*, bridged by cross-attention. Decoder-only wins at scale/generality — one architecture, one training recipe, handles seq2seq, chat, reasoning, code, etc., which is why the entire industry converged on decoder-only for general-purpose LLMs.

**Q: Why can't you just use BERT for text generation without any changes?**
Because it has no causal masking and no generation head — every token sees every other token including "future" ones during training, so there's no mechanism to generate one token at a time without leaking the answer. You'd need to add causal masking and a generation objective — at which point you've essentially built a decoder.

---

## 4. GENERATIVE AI / LLM BASICS

**Autoregressive generation**: model predicts one token at a time, each new token conditioned on all previous ones (including its own prior outputs).

**Decoding/sampling strategies**:
- **Greedy**: always pick highest-probability token — deterministic, often repetitive
- **Temperature**: scales logits before softmax; low temp = more confident/deterministic, high temp = more random/creative
- **Top-k sampling**: sample from the k most likely tokens
- **Top-p (nucleus) sampling**: sample from smallest set of tokens whose cumulative probability ≥ p — adapts set size dynamically, generally preferred over top-k

**Hallucination**: model generates fluent but factually incorrect/unsupported content — happens because LLMs optimize for plausible token sequences, not truth verification. Mitigations: RAG, grounding, fact-checking layers, lower temperature, citations.

**Context window**: max tokens (input+output) model can process at once. "Lost in the middle" — models tend to recall info at the start/end of context better than the middle.

---

## 5. RAG (Retrieval-Augmented Generation) — HIGH PRIORITY

**Why RAG**: gives LLMs access to up-to-date/private/domain knowledge without retraining; reduces hallucination by grounding answers in retrieved evidence; cheaper than fine-tuning for knowledge injection.

**Pipeline**:
1. **Indexing**: split documents into chunks → embed each chunk → store vectors in a vector DB
2. **Retrieval**: embed the user query → find nearest chunks by similarity
3. **Augmentation**: insert retrieved chunks into the prompt as context
4. **Generation**: LLM generates answer grounded in that context

**Chunking strategies**: fixed-size (simple, may cut mid-thought), recursive/semantic (split on paragraph/sentence boundaries, keeps meaning intact), sliding window with overlap (preserves context across chunk boundaries).

**Vector search**:
- **Cosine similarity**: standard metric for embedding comparison
- **ANN (Approximate Nearest Neighbor)** algorithms like **HNSW**: make search fast at scale vs exact search
- Popular vector DBs: FAISS (library, not a full DB), **Pinecone** (managed/serverless, easy to scale, no infra to manage), **Weaviate** (open-source, supports hybrid search + schema/filtering out of the box), **pgvector** (a Postgres extension — good choice when you already run Postgres and want vector search without adding a new DB to your stack, trades some raw scale/speed vs purpose-built vector DBs), Chroma, Milvus.

**Retrieval types**:
- **Dense retrieval**: embedding-based semantic similarity
- **Sparse retrieval (BM25/TF-IDF)**: keyword/term-frequency based — good for exact matches, rare terms
- **Hybrid search**: combine dense + sparse (often best in practice)
- **Reranking**: use a cross-encoder to re-score top-k retrieved candidates for better precision (retrieval is cheap+broad, reranking is expensive+precise)

### Cross-encoders in RAG — practical deep dive

**Bi-encoder vs cross-encoder (the core distinction interviewers want)**:
- **Bi-encoder** (used for the first-pass retrieval): encodes the query and each document **independently** into fixed vectors, compares them via cosine/dot-product similarity. Document embeddings can be **precomputed and cached offline**, so retrieval at query time is just a fast vector lookup (ANN search) — this is what makes it scale to millions of documents.
- **Cross-encoder**: concatenates **`[query, document]` into a single input** and runs it jointly through a transformer, so query tokens and document tokens directly attend to each other, then outputs a single relevance score. Because of that direct interaction, it captures fine-grained semantic matching (negations, exact requirement matching, nuanced relevance) far better than comparing two static vectors — but it cannot be precomputed, since the score depends on the specific query-document pair.

**Why use a cross-encoder at all if it's slower?** Because bi-encoder similarity is a coarse approximation — two documents can land close in embedding space for the wrong reasons (topical overlap without actually answering the query). A cross-encoder catches relevance errors the bi-encoder's first pass misses, meaningfully improving precision of what actually reaches the LLM's context.

**Why not use cross-encoders for the *entire* corpus (cost angle — this is exactly the question you asked)?**
Yes, cost/latency is the whole reason it's used only for reranking:
- Bi-encoder: O(1) query encoding at request time + a cheap ANN lookup against precomputed doc vectors
- Cross-encoder: requires a **full transformer forward pass for every single (query, candidate) pair, at query time** — nothing can be precomputed because the pair changes every request. Cost scales linearly with however many candidates you rerank.
- This is exactly why the standard pattern is **two-stage retrieve-then-rerank**: bi-encoder (or BM25/hybrid) narrows millions of docs down to a small candidate set (top 20-100), *then* the cross-encoder reranks only that small set. Running a cross-encoder over the full corpus per query would be far too slow/expensive to be viable.
- Mitigations in practice: keep the reranked candidate set small, use smaller/distilled cross-encoders (e.g., MiniLM-based rerankers), batch the reranking forward passes, cache reranking results for repeated/similar queries.

**Types of cross-encoders / rerankers you'll see in practice**:
- **BERT-based cross-encoders** (e.g., MiniLM cross-encoders from the `sentence-transformers` ecosystem, trained on MS MARCO-style relevance data) — the classic, cheaper option
- **Managed reranking APIs** (e.g., Cohere Rerank, or open-weight rerankers like BGE-reranker) — production-friendly, no self-hosting needed, priced per rerank call
- **ColBERT-style "late interaction" models** — a middle ground: encode query and document tokens *separately* (like a bi-encoder, so doc-side token embeddings can still be precomputed), but score via token-level max-similarity matching instead of one squashed vector. Captures more fine-grained signal than a plain bi-encoder without paying the full cross-encoder cost — a good answer if asked "is there something between bi-encoder and cross-encoder?"
- **LLM-as-reranker**: directly prompt an LLM to score/order candidate passages — most flexible and can reason about relevance semantically, but the most expensive and highest-latency option; typically reserved for smaller candidate sets where quality matters more than cost.

**Advanced RAG techniques** (good to mention, shows depth):
- **HyDE (Hypothetical Document Embeddings)**: generate a hypothetical answer first, embed *that* to retrieve — often matches document style better than the raw question
- **Query rewriting/expansion**: reformulate vague queries before retrieval
- **Multi-hop RAG**: chain multiple retrieval steps for complex questions
- **GraphRAG**: build a knowledge graph over documents, retrieve via graph traversal — good for relational/multi-entity questions
- **Self-RAG**: model decides when to retrieve and self-critiques retrieved content

**RAG failure modes**: irrelevant retrieval, chunking that breaks context, context window overflow, retrieval fetches right doc but wrong section, generation ignores retrieved context (still hallucinates).

**RAG evaluation** (see also Section 9): faithfulness (does answer stay grounded in retrieved context?), context precision/recall (did we retrieve the right, and only the right, chunks?), answer relevance.

---

## 6. VECTOR DATABASES & VECTOR SEARCH — ARCHITECTURE DEEP DIVE

You already saw vector DBs named in the RAG section (Pinecone, Weaviate, pgvector, etc.) — this section is the "why do they exist and how do they actually work" that interviewers dig into once they hear you can name-drop them.

### Why use a vector DB at all?

An embedding turns text/images/anything into a dense numeric vector where **semantic similarity = geometric closeness**. Once you have millions of these, you need to answer "which stored vectors are closest to this query vector?" fast, repeatedly, in production. That's the entire problem a vector DB exists to solve.

**Why not just brute-force it in a normal DB/array?** Brute-force nearest-neighbor search compares the query vector against **every single stored vector** (O(n·d) — n vectors, d dimensions). At a few thousand vectors this is instant; at millions/billions it becomes far too slow for a chat-latency budget. This is the entire reason **Approximate Nearest Neighbor (ANN)** indexing exists — trade a small amount of recall (you might miss the *absolute* single best match occasionally) for orders-of-magnitude faster search.

**Vector DB vs a plain ANN library (e.g., FAISS)** — a distinction interviewers like to probe: FAISS is just an **in-memory indexing/search library** — fast ANN algorithms, but no persistence, no CRUD API, no metadata filtering, no multi-tenancy, no built-in scaling/replication out of the box. A **vector database** (Pinecone, Weaviate, Milvus, Qdrant) wraps an ANN index with all the production plumbing: durable storage, insert/update/delete APIs, metadata filtering, sharding across nodes, replication for availability, and often hybrid (dense+sparse) search built in. pgvector is the middle case — it bolts ANN indexing onto Postgres, so you inherit Postgres's durability/transactions/tooling but at somewhat lower raw ANN performance/scale ceiling than purpose-built vector DBs.

### Core architecture — the four layers

1. **Storage layer**: raw vectors + associated metadata (source doc, timestamp, tags, ACL info) persisted to disk, often alongside the original text/ID for retrieval after a match is found.
2. **Indexing layer**: the ANN data structure built over the vectors (see below — this is the part people mean by "vector DB internals").
3. **Query layer**: takes an incoming query vector, traverses the index to find approximate nearest neighbors, optionally applies metadata filters, returns top-k IDs + scores.
4. **Distribution layer** (at scale): sharding (split the index across multiple nodes so no single machine holds everything) and replication (duplicate shards for availability/read throughput).

### ANN indexing algorithms — the part that actually gets asked

**HNSW (Hierarchical Navigable Small World)** — the most common default today (Weaviate, Qdrant, Milvus, pgvector all support it):
- Builds a **multi-layer graph** of vectors. Top layers are sparse with long-range links (like an "express lane"); the bottom layer is dense with short-range links connecting every vector to its close neighbors.
- Search starts at the top layer, greedily walks toward the query vector, then **drops down a layer** once it can't get closer, repeating until it reaches the bottom layer for a fine-grained answer. Conceptually similar to a skip-list, but generalized to high-dimensional similarity instead of sorted order.
- Key tunable params: **M** (max connections per node — higher = better recall, more memory), **ef_construction** (how thorough the graph-build search is — higher = better graph quality, slower to build), **ef_search** (how thorough the query-time search is — higher = better recall, slower queries). This recall-vs-speed-vs-memory triangle is exactly what interviewers want you to articulate.
- Strength: very good recall/speed tradeoff, handles dynamic inserts reasonably well. Weakness: graph structure uses more memory than some alternatives.

**IVF (Inverted File Index)**:
- Offline step: cluster all vectors into N buckets via k-means (each bucket has a centroid).
- Query time: find the nearest few cluster centroids to the query, then **only** brute-force search *within those clusters* instead of the whole dataset — this is the speedup (search a tiny fraction of the data).
- Tunable: **nlist** (number of clusters — more clusters = finer partitioning, faster search, slightly lower recall if you search too few of them) and **nprobe** (how many clusters to check per query — higher = better recall, slower).

**Product Quantization (PQ)** — usually paired with IVF (as "IVF-PQ") for **memory compression**, not speed alone:
- Splits each vector into sub-vectors, and quantizes (compresses) each sub-vector into a small code from a learned codebook, rather than storing full-precision floats.
- Massively reduces memory footprint (this is the direct answer to "how do you keep vector DB costs down at billions-of-vectors scale") at the cost of some accuracy — you're now searching over compressed/approximate representations. Common at very large scale where storing raw float32 vectors for billions of items is prohibitively expensive.

**Flat/brute-force index**: exact search, no approximation — only viable at small scale (thousands–low millions) or when you truly need 100% recall (e.g., a small curated candidate set after filtering); still shows up as an interview question ("when would you use exact search over ANN?").

**LSH (Locality-Sensitive Hashing)**: older approach, hashes similar vectors into the same buckets with high probability; largely superseded by HNSW/IVF in modern vector DBs but worth recognizing by name.

### Distance metrics
- **Cosine similarity**: measures angle between vectors, ignores magnitude — standard choice for text embeddings, since embedding *direction* carries the semantic meaning.
- **Dot product**: similar to cosine but magnitude-sensitive — used when the embedding model was specifically trained for it (some retrieval models bake normalization in, making dot product equivalent to cosine).
- **Euclidean (L2) distance**: straight-line distance — more common in non-text embedding spaces (e.g., some image/audio embeddings), less common for standard text embedding retrieval.

### Metadata filtering — pre-filter vs post-filter
- **Post-filtering**: run the ANN search first, then discard results that don't match metadata constraints (e.g., `category=beachwear`) — simple, but risks returning fewer than top-k valid results if many top matches get filtered out.
- **Pre-filtering**: restrict the search space to only vectors matching the metadata constraint *before* running ANN search — more accurate results, but not all ANN index structures support efficient pre-filtering natively (this is a real differentiator between vector DB products — some handle "filtered ANN search" much better than others).
- This is exactly the mechanism used in the beachwear case study earlier: filter to in-stock beachwear items *first*, then rank by similarity within that valid set.

### Practical Q&A

**Q: Why not just use a SQL database with cosine similarity computed in application code?**
Works fine at small scale, but you lose ANN indexing (so it degrades to brute-force as data grows), and you'd be reinventing sharding/replication/filtering that purpose-built vector DBs already solve.

**Q: When would exact (flat) search be the right choice over ANN?**
When the candidate set is already small (e.g., after hard metadata filtering it's down to a few hundred items) or when 100% recall is a hard requirement — the approximation error in ANN isn't worth the speed gain at small scale anyway.

**Q: How do you reduce vector DB cost/memory at large scale?**
Product quantization (or scalar quantization) to compress stored vectors, choosing IVF-PQ over full HNSW when memory is the binding constraint, and reducing embedding dimensionality (e.g., using a smaller embedding model or a dimensionality-reduced variant) where recall requirements allow it.

**Q: What happens when you insert/delete vectors in an HNSW index — is it easy?**
Inserts are generally supported natively and reasonably efficient (HNSW was designed for dynamic data). Deletes are trickier — many implementations use "tombstoning" (mark as deleted, filter out at query time) rather than truly removing the node immediately, since ripping a node out of the graph structure is more expensive; periodic index rebuilds/compaction clean up tombstoned nodes.

---

## 7. FINE-TUNING — HIGH PRIORITY

**When to fine-tune vs prompt engineer vs RAG**:
- Prompt engineering: fastest, no training, good for format/behavior control
- RAG: best for injecting *knowledge* (facts, docs) that changes over time
- Fine-tuning: best for teaching *behavior/style/format* consistently, or domain-specific patterns not expressible via prompting alone

**Full fine-tuning**: update all model weights — expensive, needs lots of data/compute, risk of catastrophic forgetting (losing general capabilities).

**PEFT (Parameter-Efficient Fine-Tuning)** — the practical default now:
- **LoRA (Low-Rank Adaptation)**: freeze original weights, inject small trainable low-rank matrices (A×B) into layers; drastically fewer trainable params, easy to swap/merge adapters
- **QLoRA**: LoRA + quantizing the base model (e.g., to 4-bit) → fine-tune large models on consumer-grade GPUs
- **Prefix/Prompt tuning**: learn continuous "soft prompt" vectors prepended to input, rest of model frozen
- **Adapters**: small bottleneck layers inserted between transformer layers, only these are trained

**Instruction tuning**: fine-tune on (instruction, response) pairs so the model follows natural-language instructions better — this is what turns a base LLM into a "chat" model.

**RLHF (Reinforcement Learning from Human Feedback)**:
1. Collect human preference data (rank/compare model outputs)
2. Train a reward model to predict human preference
3. Use RL (typically PPO) to fine-tune the LLM to maximize reward model score

**DPO (Direct Preference Optimization)**: newer, simpler alternative to RLHF — directly optimizes on preference pairs without needing a separate reward model or RL loop. Increasingly preferred for being simpler/more stable.

**Catastrophic forgetting**: model loses previously learned general capabilities when fine-tuned narrowly on new data — mitigate with PEFT, lower learning rates, mixing in general data.

---

## 8. PROMPT ENGINEERING — HIGH PRIORITY

**Zero-shot**: just ask, no examples. **Few-shot**: include 2-5 examples in the prompt to demonstrate the pattern/format.

**Chain-of-Thought (CoT)**: ask the model to "think step by step" — improves performance on reasoning/math tasks by forcing intermediate reasoning tokens instead of jumping to an answer.

**Self-consistency**: sample multiple CoT reasoning paths (with temperature>0), take the majority-vote answer — more robust than a single CoT pass.

**Tree of Thoughts (ToT)**: explore multiple reasoning branches, evaluate/backtrack — more compute but better for complex planning/search problems than a single linear CoT.

**ReAct (Reason + Act)**: interleave reasoning traces with tool-use actions ("Thought → Action → Observation" loop) — foundational pattern for agents (see Section 10).

**Role/system prompting**: setting persona/behavior instructions in a system prompt to steer tone, constraints, output format.

**Structured output prompting**: explicitly requesting JSON/XML schemas, often paired with few-shot examples of the exact format wanted.

**Prompt chaining**: break a complex task into a sequence of simpler prompts, feeding output of one into the next (vs one giant prompt).

**Least-to-most prompting**: decompose a hard problem into an ordered sequence of easier subproblems, solve sequentially.

**In-context learning**: the general phenomenon where LLMs adapt behavior purely from prompt content (examples/instructions) without any weight updates.

---

## 9. EVALUATION — HIGH PRIORITY

**Classic NLP metrics** (mostly for translation/summarization):
- **BLEU**: n-gram precision overlap with reference text — used for translation
- **ROUGE**: n-gram recall overlap — used for summarization
- **METEOR**: like BLEU but accounts for synonyms/stemming
- **Perplexity**: how "surprised" a language model is by text — lower = better fit; measures fluency, not correctness

**LLM benchmark suites** (know these exist, not deep detail): MMLU (broad knowledge/reasoning across subjects), HellaSwag (commonsense), HumanEval (code), TruthfulQA (factuality/misconceptions), BIG-Bench.

**LLM-as-judge**: use a strong LLM to score/compare outputs (e.g., pairwise preference or rubric-based scoring) — scalable but has known biases (favors longer/more confident-sounding answers, position bias in pairwise comparisons).

**Human evaluation**: gold standard but expensive/slow; often used to validate that automated metrics/LLM-judges correlate with human judgment.

**Task-specific qualitative metrics**: faithfulness/groundedness (is the answer supported by given context?), relevance, coherence, toxicity/safety, helpfulness.

**RAG-specific evaluation (RAGAS framework concepts)**:
- **Context precision/recall**: did retrieval fetch the relevant chunks, and only those?
- **Faithfulness**: is the generated answer actually supported by retrieved context (no hallucination beyond it)?
- **Answer relevance**: does the answer actually address the question?

**Online vs offline eval**: offline = static test sets/benchmarks before deployment; online = A/B testing, live user feedback, guardrail monitoring in production.

---

## 10. AGENTIC AI

**What makes something an "agent"** (vs a plain LLM call): it perceives (input/observations), reasons/plans, takes actions (often via tools), and often maintains memory across steps — operating in a loop rather than single-shot.

**ReAct loop**: Thought (reason about what to do) → Action (call a tool) → Observation (see result) → repeat until done. This is the core pattern behind most agent frameworks.

**Tool use / function calling**: the LLM outputs a structured call (function name + args) that your code executes, feeding the result back into context — lets LLMs use calculators, search, APIs, databases, etc., extending them beyond pure text generation.

**Planning/task decomposition**: breaking a high-level goal into ordered subtasks — can be done upfront (plan-and-execute) or dynamically step-by-step (ReAct-style).

**Memory**:
- **Short-term**: the current context window / conversation history
- **Long-term**: external storage (often a vector DB) the agent can write to/retrieve from across sessions

**Multi-agent systems**: multiple specialized agents (e.g., planner, researcher, coder, critic) collaborate or debate, sometimes with a supervisor/orchestrator agent routing work. Common patterns: **sequential** (output of one feeds the next), **hierarchical** (a supervisor/orchestrator delegates to worker agents), **debate/critique** (agents check each other's outputs to reduce errors).

**Autonomous workflows**: agents that run multi-step tasks with minimal human intervention between steps — the risk/reliability tradeoff is the key discussion point (more autonomy = more efficiency but more risk of compounding errors or unwanted actions), so most production systems add checkpoints, approval gates, or guardrails before high-stakes actions.

**Agentic AI frameworks** (know what each is *for*, not deep API syntax):
- **LangChain**: general-purpose framework for chaining LLM calls, tools, memory, and retrieval into pipelines — most widely adopted "glue" layer for LLM apps.
- **LangGraph**: built by the LangChain team specifically for **agentic** workflows — models the app as a graph of nodes/edges (state machine) rather than a linear chain, so it handles loops, branching, and multi-agent orchestration more naturally than plain LangChain.
- **LlamaIndex**: focused specifically on the **data/retrieval side** — connecting LLMs to your documents/data sources, with strong tooling around indexing, chunking, and query engines (often used *for* the RAG layer, sometimes alongside LangChain/LangGraph for orchestration).
- **Semantic Kernel**: Microsoft's SDK for integrating LLMs into applications — conceptually similar goals to LangChain (plugins/"skills", memory, planners) but rooted in the .NET/enterprise app ecosystem, with Python/Java support too.
- **CrewAI, AutoGPT-style agents**: higher-level multi-agent orchestration frameworks — define agents with roles/goals and let them collaborate on a task with less manual graph-building than LangGraph.

**Tool calling in practice**: model output is constrained to a schema (usually JSON) matching a function signature; your application code validates and executes it, then returns the result as a new message in the conversation for the model to continue reasoning with. Reliability tips interviewers may probe: schema validation, retries on malformed calls, timeouts, and never letting the model execute destructive actions without a check.

### MCP (Model Context Protocol) and A2A (Agent2Agent) — the two interoperability protocols

These come up increasingly in agentic-AI interviews because they solve the "every integration is custom code" problem at two different layers. Know the one-line distinction cold: **MCP = agent-to-tool/data. A2A = agent-to-agent.**

**MCP (Model Context Protocol)** — introduced by Anthropic (Nov 2024), now widely adopted industry-wide:
- **Problem it solves**: before MCP, every AI app needed bespoke integration code for every tool/data source it wanted to use (an "N×M" integration problem — N apps × M tools = N×M custom integrations). MCP standardizes the interface so one MCP server implementation works with *any* MCP-compatible client.
- **Architecture**: client-server model. An **MCP server** exposes capabilities from a specific tool/data source (e.g., a GitHub server, a Slack server, a database server). An **MCP client** (embedded in an AI app — Claude, an IDE, a custom agent) connects to one or more servers and lets the LLM use their capabilities.
- **Three core primitives** a server can expose: **Tools** (functions the model can call — "search issues," "create a ticket"), **Resources** (readable data the model can pull in as context — files, DB rows), **Prompts** (reusable prompt templates the server provides).
- **Transport**: JSON-RPC 2.0 messages, over either **stdio** (for local servers running as a subprocess) or **HTTP/SSE** (for remote servers).
- **Common analogy** (safe to use in an interview): "USB-C for AI applications" — one standard connector instead of a different cable for every device.

**A2A (Agent2Agent Protocol)** — announced by Google (April 2025), contributed to the **Linux Foundation** for vendor-neutral governance shortly after, with broad industry backing (Microsoft, AWS, Salesforce, SAP, and others have adopted/integrated it):
- **Problem it solves**: MCP connects *one* agent to tools/data — it doesn't define how two independent *agents* (potentially built on completely different frameworks/vendors) discover each other, negotiate what they can do, and delegate work. A2A is that layer.
- **Core mechanics**: 
  - **Agent Cards**: a JSON manifest an agent publishes describing its capabilities/skills/endpoint — how a client agent discovers *what a remote agent can do* before delegating to it (conceptually similar to an API spec or service-discovery document).
  - **Task lifecycle**: a client agent sends a task to a remote agent, which moves through defined states (submitted → working → input-required → completed/failed/canceled) — this explicit lifecycle is what lets a client track and manage a long-running delegated task rather than treating it as a single blocking call.
  - **Transport**: built on existing standards rather than inventing new ones — HTTP, Server-Sent Events (for streaming task updates), and JSON-RPC 2.0, with OAuth 2.0 for agent-to-agent authentication.
- **Why it's a separate layer from MCP, not a competitor**: MCP answers "how does an agent use a tool/data source"; A2A answers "how does an agent find and delegate to *another agent*" — a full agentic system in production often uses both together (each agent uses MCP for its own tools, and A2A to hand off tasks to peer agents).

**Practical interview framing**: if asked "how would two agents built on different frameworks collaborate," the strong answer is A2A (Agent Cards for discovery + task delegation), not "just have them call each other's APIs directly" — the point of A2A is avoiding custom point-to-point integration between every pair of agents, the same problem MCP solved for agent-to-tool integration.

**Key challenges**: error compounding across multi-step loops, infinite loops/lack of termination, cost (many LLM calls per task), reliability of tool-call formatting, evaluating agent trajectories (not just final output). For A2A/multi-agent specifically: trust/security across organizational boundaries (are you willing to let another company's agent execute tasks in your system?), and the protocol layer being new enough (2025-2026) that production track record is still thinner than the marketing suggests — worth acknowledging that maturity caveat if asked how production-ready it is.

---

## 11. ML ENGINEERING / MLOPS

**Docker**: packages your model + code + dependencies into a single portable container so "works on my machine" becomes "works everywhere." For DS interviews, know: a `Dockerfile` defines the image (base image, install deps, copy code, entrypoint); an image is the blueprint, a container is a running instance of it; containers are lighter than VMs because they share the host OS kernel instead of virtualizing hardware.

**Kubernetes (K8s)**: orchestrates containers at scale — handles deploying, scaling, restarting, and load-balancing across many containers/machines. Key concepts:
- **Pod**: smallest deployable unit, usually one container (sometimes a few tightly coupled ones)
- **Deployment**: manages desired state — how many replicas of a pod should run, handles rolling updates
- **Service**: stable network endpoint to reach a set of pods (pods are ephemeral, IPs change)
- **Horizontal Pod Autoscaler (HPA)**: automatically adds/removes pod replicas based on CPU/GPU/custom metrics (like request load)
- Why it matters for ML: lets you auto-scale inference servers with traffic, roll out new model versions with zero downtime, and isolate resource-heavy training jobs from serving.

**Scaling models (serving/inference)**:
- **Horizontal scaling**: run more replicas of the model server behind a load balancer — standard approach for handling more requests
- **Vertical scaling**: bigger machine/more GPU memory — helps with bigger models, not concurrent request volume
- **Batching**: group multiple incoming requests into one forward pass to use GPU more efficiently (common in LLM serving — e.g., continuous/dynamic batching)
- **Model optimization for serving**: quantization (lower precision weights, e.g., FP16/INT8/INT4 — smaller/faster, small accuracy tradeoff), distillation (train a smaller "student" model to mimic a larger "teacher"), caching (e.g., KV-cache reuse in LLMs, or caching repeated identical requests)
- **Serving frameworks** to know by name: TorchServe, TensorFlow Serving, Triton Inference Server, vLLM/TGI for LLM-specific serving (optimized for batching + KV-cache)
- **Online vs batch inference**: online = low-latency, one request at a time (real-time recommendations); batch = process large volumes on a schedule, latency doesn't matter (nightly scoring jobs)

**Observability / monitoring**:
- **Data/feature drift**: input data distribution shifts from training distribution over time — model quietly gets stale
- **Concept drift**: the relationship between inputs and target changes (the thing you're predicting behaves differently now)
- **Model performance monitoring**: track live metrics (accuracy proxies, latency, error rates) since ground truth labels are often delayed or unavailable in production
- **Logging**: predictions, inputs, latencies, errors — needed for debugging and retraining triggers
- **Alerting**: automated triggers when drift/error metrics cross thresholds
- **LLM-specific observability**: token usage/cost tracking, hallucination/guardrail flags, prompt-response logging, latency per call, tracing multi-step agent/chain executions
- **LangSmith**: LangChain's own observability/eval tool — traces every step of a chain/agent run (inputs, outputs, tool calls, latency per node), plus dataset-based evaluation and prompt versioning. Tightest fit if you're already on LangChain/LangGraph.
- **Langfuse**: open-source, framework-agnostic alternative to LangSmith — tracing, prompt management, cost/latency analytics, and evaluation scoring; popular when teams don't want vendor lock-in to one orchestration framework.
- **AI evaluation frameworks** (beyond RAGAS): tools/libraries that run automated eval suites (LLM-as-judge scoring, regression tests on prompts, dataset-based checks) as part of CI, so prompt/model changes don't silently degrade quality.

**Model versioning**:
- Track not just code (git) but **model artifacts, training data version, hyperparameters, and environment** together — a model result is only reproducible if all four are pinned
- **Model registry**: central store of model versions with metadata/stage (e.g., staging/production/archived) — MLflow Model Registry, or cloud-native equivalents (SageMaker Model Registry, Vertex AI Model Registry)
- **Experiment tracking**: logging metrics/params per training run so you can compare experiments — MLflow, Weights & Biases
- **Data versioning**: tools like DVC pair with git to version large datasets that git itself can't handle well
- **Rollback**: ability to revert to a previous model version quickly if a new deployment regresses — this is why versioning + registry matters operationally, not just for record-keeping

---

## 12. DEVOPS & DEPLOYMENT FOR AI APPS

**CI/CD pipelines**: automated pipeline that runs on every code change — CI (Continuous Integration) runs tests/builds automatically; CD (Continuous Delivery/Deployment) automatically pushes passing builds toward staging/production. For ML/LLM apps this typically includes: unit tests, prompt/eval regression tests (so a prompt change that quietly breaks quality gets caught before merge), building the container image, and triggering deployment.

**Git**: version control basics interviewers may probe conceptually — branching strategy (feature branches → PR → merge to main), why PRs/code review matter even for prompt or notebook changes, and treating prompts/configs as versioned artifacts just like code (not something edited ad hoc in production).

**Performance optimization** (for AI apps specifically): reduce latency via smaller/distilled models where acceptable, caching repeated queries/embeddings, parallelizing independent retrieval or tool calls, streaming tokens back to the user instead of waiting for the full response, and choosing the right model size per task (don't use the biggest model for a task a small one handles fine).

**Cost optimization** (a very common GenAI-specific interview angle):
- **Model selection**: use a cheaper/smaller model for simple sub-tasks, reserve the largest model for the hardest step (a common pattern: small model for routing/classification, large model only for final generation)
- **Prompt/context minimization**: trim unnecessary context, summarize long history instead of resending it every call, cache system prompts
- **Caching**: cache identical/similar requests and embeddings so you don't recompute or re-call the API
- **Batching**: combine requests where latency isn't critical
- **Token usage monitoring**: track cost per request/user/feature so runaway usage (e.g., an agent stuck in a loop) is caught early

---

## 13. APIS, MICROSERVICES, SECURITY & UI

**REST APIs**: the standard way an AI service is exposed to other systems — stateless requests, resource-based URLs, standard verbs (GET/POST/PUT/DELETE), JSON payloads. For an AI app this is usually how you expose "send a prompt/query, get a response" to other internal systems or a frontend.

**Microservices**: splitting an application into small, independently deployable services (e.g., a retrieval service, an embedding service, an LLM-orchestration service, an auth service) rather than one monolith — lets each piece scale and deploy independently, at the cost of added operational complexity (network calls, service discovery, more moving pieces to monitor).

**API security**: authentication (API keys, OAuth tokens) to verify who's calling; authorization (role/permission checks) to control what they can do; rate limiting to prevent abuse/cost overrun; input validation/sanitization especially important for LLM apps (prompt injection attempts can arrive as "user input" through an API).

**Data security**: encryption in transit (TLS) and at rest; avoiding sending sensitive/PII data to third-party LLM APIs unless contractually covered; redaction/masking of sensitive fields before they reach a prompt or log; access controls on who can query what data via the AI system (especially relevant for RAG over internal/enterprise documents — the AI shouldn't surface documents a given user isn't authorized to see).

**Basic React or Streamlit integration**: for a DS/AI role this is usually about being able to stand up a lightweight demo/UI to showcase a model or RAG app — **Streamlit** is the fast option (pure Python, minimal frontend code, great for prototypes/internal tools); **React** is used when the AI feature needs to live inside a proper production frontend alongside a design system — know the distinction (prototype/internal tool vs production-grade UI) even if you're not expected to be a frontend expert.

---

## 14. GENAI SYSTEM DESIGN CASE STUDIES

These "design a chatbot for X" questions are really testing whether you reach for the right pieces (NLU, RAG, tools, ranking, guardrails, evaluation) in the right order — not whether you know one exact right answer. Use the framework below on *any* case study you get, then two fully worked examples, then two more to practice on your own.

### Reusable framework (apply this to any "design a chatbot/agent for X" question)

1. **Clarify the intent & inputs**: what is the user actually trying to accomplish, and what's in a typical query? (Informal/slang language changes your NLU approach.)
2. **Entity/intent extraction**: what needs to be pulled out of the raw text (location, date, product type, sentiment, etc.), and is rule-based NER enough or do you need an LLM-based extractor because phrasing is unpredictable?
3. **Context enrichment**: what *external* info does the system need beyond the query itself (weather, user profile/history, live inventory, account data)? These become **tool calls**.
4. **Retrieval/grounding**: what "knowledge" does the answer need to be grounded in (product catalog, docs, policies)? This is your RAG layer — and it must be **live/queried**, not baked into the LLM, whenever the underlying data changes (stock, prices, policies).
5. **Ranking/personalization**: once you have candidates (products, docs, answers), how do you pick/order the best ones — relevance, personalization signals, business rules (stock, trending), and is a reranker (cross-encoder) justified?
6. **Generation**: how does the LLM turn ranked results into a final response — tone, structured output (cards/JSON vs plain text), and critically: is it instructed to **only** reference retrieved/verified data (no hallucinated products/facts)?
7. **Guardrails**: what can go wrong (hallucinated claims, inappropriate content, prompt injection, giving advice outside scope) and what stops it?
8. **Latency/cost**: multi-step pipelines add latency — what can run in parallel (independent tool calls), what can be cached, what can be a smaller/cheaper model?
9. **Evaluation & feedback loop**: offline metrics (precision@k on historical data), online metrics (CTR, conversion, session satisfaction), and how engagement signals feed back into ranking/personalization over time.

---

### Worked Example 1: Gen-Z fashion recommendation chatbot
*"I am Swati, I'm going to Goa beach next Sunday, recommend me something to wear."*

**1. Intent/entity extraction**: Informal, conversational phrasing (typical of Gen-Z users) — rigid regex/NER will break on variety of phrasing, emojis, abbreviations. Use an **LLM-based extractor** (a small/cheap model call, or a structured-output function call) to pull: `{intent: clothing_recommendation, location: "Goa beach", date: "next Sunday", name: "Swati"}`. Name is used for a friendly personalized greeting, not for the recommendation logic itself.

**2. Context enrichment (tool calls)**: 
- Weather API call for Goa, resolved date → affects fabric/coverage suggestions (hot & sunny → breathable, sun protection; rain forecast → quick-dry, avoid light colors)
- User profile lookup (if returning user) → past purchases, saved size, style preference history

**3. Retrieval (RAG over product catalog)**: vector search over product embeddings, filtered by hard constraints first (category = beachwear/resortwear, size in stock, gender if known) — filtering *before* vector similarity keeps the candidate set both relevant and valid (never recommend an out-of-stock item).

**4. Ranking**: rerank the filtered candidates by a blend of: semantic relevance to the query, personalization score (style history/embedding similarity to past likes), business signals (trending among similar users, margin/promo priority) — a lightweight learned ranker or even a cross-encoder reranker is justified here since the candidate set is small (already filtered) and getting the top 4-6 right matters a lot for conversion.

**5. Generation**: LLM composes the final reply in an on-brand, casual tone, referencing the weather context naturally ("it's gonna be sunny and hot 🌴"), presenting the top picks as **structured product cards** (title/price/image/link) rather than a wall of text — the LLM should be explicitly instructed to only mention products that came back from retrieval (never invent an item or price), and can end with a light clarifying question ("bold prints or something low-key?") to keep the conversation going.

**6. Guardrails**: no hallucinated products/prices/stock; no overconfident weather guarantees ("it *will* rain" → phrase as forecast, not certainty); content moderation on both user input and generated output; no storing/using sensitive personal data beyond what's needed for the recommendation.

**7. Latency/cost**: weather lookup and product retrieval are independent → run in **parallel**, not sequentially. Use a smaller/cheaper model for the entity-extraction step, reserve the larger model for the final natural-language generation. Cache weather forecasts (they don't need to be refetched per user).

**8. Evaluation**: offline — precision@k against historical purchase/click data for similar queries; online — CTR on recommended cards, add-to-cart/conversion rate, and session-level engagement; feed click/purchase signals back into the user's style-preference embedding for next time.

---

### Worked Example 2: Enterprise IT helpdesk copilot
*"My VPN keeps disconnecting every 10 minutes, can you help?"*

**1. Intent/entity extraction**: intent = technical troubleshooting; entities = product/system (VPN), symptom (disconnects), frequency (every 10 min). Likely simpler/more structured phrasing than the Gen-Z case, so a lighter-weight classifier can work, but LLM-based extraction is still safer for the long tail of phrasing.

**2. Context enrichment**: pull the user's device/OS info and recent ticket history if available (was this reported before? known outage affecting this user's region?) — this is a **tool call** into the ticketing/asset system, not something the LLM should guess.

**3. Retrieval (RAG)**: search internal IT knowledge base/runbooks for VPN troubleshooting docs — this is a strong case for **access-controlled RAG**: retrieval must respect the querying user's permissions (don't surface docs meant for a different business unit or containing other users' data).

**4. Generation + escalation logic**: if the retrieved runbook resolves it (e.g., "reconnect after updating client to vX"), generate step-by-step instructions grounded in that doc. If retrieval confidence is low or the issue matches a pattern for known unresolved outages, the agent should **escalate** — this is the key design decision interviewers want to hear: know when *not* to let the LLM freelance an answer, and instead create a ticket / hand off to a human, with the full conversation context attached.

**5. Guardrails**: never let the copilot execute destructive actions (e.g., resetting network configs) without explicit confirmation; must not fabricate a fix that isn't grounded in an actual internal doc — "I don't have a documented fix for this, escalating to the network team" is a *correct*, expected output, not a failure.

**6. Evaluation**: ticket deflection rate (issues resolved without human involvement) balanced against re-open rate (did the bot's "fix" actually work, or did the user have to re-file?) — a high deflection rate with a high re-open rate means the bot is closing tickets prematurely, not actually helping.

---

### Two more to practice on your own (framework applies the same way — try structuring these before your interview)

**Practice Case 3 — Multi-agent research/market-analysis assistant**: *"Summarize this week's news on our top 3 competitors and flag anything that affects our Q3 strategy."* Think about: multiple specialized agents (a fetch/search agent, a per-competitor summarizer agent, a "strategic relevance" critic agent) vs one agent doing everything sequentially; how you'd prevent the summarizer from hallucinating relevance links that aren't actually in the source articles; how you'd evaluate "flagged as relevant" quality without ground truth (LLM-as-judge? human spot-check sampling?).

**Practice Case 4 — Customer support chatbot for e-commerce order issues**: *"Where's my order? It's been 5 days."* Think about: this needs a **tool call** into the order-management system (not RAG — this is live transactional data, not a knowledge base), how you'd handle a user who's frustrated (tone/de-escalation in generation), when to auto-issue a resolution (refund/reship) vs escalate to a human agent for judgment calls, and what guardrails prevent the bot from promising something (like a refund) the business hasn't actually authorized it to promise.

---

## 15. RAPID-FIRE Q&A (likely interview questions)

| Question | One-line answer |
|---|---|
| RAG vs fine-tuning? | RAG injects external knowledge at inference time; fine-tuning bakes behavior/patterns into weights. |
| Why does attention scale as O(n²)? | Every token attends to every other token — pairwise comparisons. |
| What's the point of multi-head attention? | Lets the model capture multiple types of relationships (syntax, semantics, position) in parallel. |
| Why LoRA over full fine-tuning? | Far fewer trainable params, less compute/memory, less catastrophic forgetting, easy to swap adapters. |
| Temperature vs top-p? | Temperature reshapes the whole probability distribution's sharpness; top-p truncates to a dynamic-size probable set before sampling. |
| Why does CoT help? | Gives the model "space" to do intermediate computation via generated tokens instead of forcing an instant answer. |
| What's a hallucination and how do you reduce it? | Fluent but unsupported/incorrect output; reduce via RAG/grounding, lower temperature, citation requirements, fact-checking. |
| BLEU/ROUGE limitations? | Surface n-gram overlap — miss semantic equivalence (paraphrases score low even if correct). |
| Why is LLM-as-judge risky? | Biases toward verbosity/confidence, position bias in comparisons, judge's own blind spots propagate. |
| What's the "lost in the middle" problem? | Long-context models recall info at start/end better than middle — matters for RAG context ordering. |
| Container vs VM? | Containers share the host OS kernel (lightweight); VMs virtualize full hardware+OS (heavier, more isolated). |
| Why Kubernetes for ML serving? | Auto-scaling, zero-downtime rollouts, self-healing (restarts failed pods), resource isolation between jobs. |
| Data drift vs concept drift? | Drift = input distribution changes; concept drift = the input→output relationship itself changes. |
| Why quantize a model? | Smaller memory footprint + faster inference, at a small, usually acceptable, accuracy cost. |
| What's a model registry for? | Single source of truth for which model version is in production, with metadata for rollback/audit. |
| LangChain vs LangGraph? | LangChain = linear chains of LLM calls/tools; LangGraph = graph/state-machine model built for loops, branching, and multi-agent orchestration. |
| LlamaIndex vs LangChain? | LlamaIndex specializes in the data/retrieval layer (indexing, chunking, query engines); LangChain is broader general-purpose orchestration — often used together. |
| pgvector vs a dedicated vector DB? | pgvector adds vector search to existing Postgres (simpler stack, less new infra); dedicated DBs (Pinecone/Weaviate/Milvus) scale further and have more retrieval-specific features. |
| LangSmith vs Langfuse? | LangSmith is tightest with LangChain/LangGraph; Langfuse is open-source and framework-agnostic — both trace/eval LLM app runs. |
| How do you cut LLM API costs? | Route easy sub-tasks to smaller/cheaper models, cache repeated calls/embeddings, trim/summarize context, batch where latency allows, monitor token usage per feature. |
| Why prompts belong in version control? | Behavior-changing artifact just like code — needs review, versioning, and rollback capability. |
| Biggest API security risk specific to LLM apps? | Prompt injection via user input reaching the model with elevated trust/permissions — mitigate with input validation and least-privilege tool access. |
| Can GPT (decoder-only) do summarization? | Yes — concatenate source+instruction+target and train/prompt causally; source tokens are already "past" so they get full attention when generating. |
| What's the one architectural lever if you want source text treated more like an encoder would? | Change the attention mask, not the architecture — use a Prefix-LM mask (bidirectional over source, causal over generated target). |
| Encoder vs BERT — what's actually different? | Encoder = the generic bidirectional self-attention+FFN block; BERT = that block + specific input scheme (segment embeddings) + specific pretraining (MLM/NSP) + specific tokenizer/weights. |
| Why are raw BERT [CLS] embeddings bad for similarity search? | MLM/NSP pretraining never optimized the embedding space for cosine similarity — need contrastive fine-tuning (Sentence-BERT) for that. |
| Why not run a cross-encoder over the whole corpus? | Cost scales linearly with candidates and can't be precomputed (score depends on the query-doc pair) — only viable on a small pre-filtered set. |
| Bi-encoder vs cross-encoder vs ColBERT? | Bi-encoder: precomputed vectors, fast, coarser. Cross-encoder: joint attention, most accurate, expensive, query-time only. ColBERT: precomputed per-token vectors + late token-level matching — middle ground. |
| Why do vector DBs need ANN instead of exact search? | Brute-force is O(n·d) per query — too slow at millions/billions of vectors; ANN trades a little recall for large speed gains. |
| HNSW in one line? | Multi-layer graph — search greedily from sparse top layers down to the dense bottom layer, like a skip-list for high dimensions. |
| IVF vs HNSW? | IVF clusters vectors and searches only nearby clusters (faster to build, less memory); HNSW builds a navigable graph (usually better recall/speed tradeoff, more memory). |
| What is Product Quantization for? | Compresses stored vectors into small codes to cut memory footprint at huge scale, at some accuracy cost — often paired with IVF as IVF-PQ. |
| Pre-filtering vs post-filtering metadata in vector search? | Pre-filter restricts the search space before ANN search (more accurate, needs index support); post-filter runs ANN first then discards non-matches (simpler, can return fewer than top-k). |
| FAISS vs a full vector database? | FAISS is just an ANN indexing/search library (no persistence/API/scaling); a vector DB wraps ANN indexing with storage, CRUD, filtering, sharding, and replication. |
| MCP vs A2A in one line? | MCP = standardizes agent-to-tool/data connections; A2A = standardizes agent-to-agent discovery and task delegation. Different layers, used together. |
| What problem does MCP actually solve? | The N×M integration problem — without it, every AI app needs custom code for every tool; MCP makes one server work with any compatible client. |
| What are Agent Cards in A2A? | A JSON manifest a remote agent publishes describing its capabilities — how a client agent discovers what it can delegate before sending it a task. |
| Why not just have two agents call each other's REST APIs directly instead of A2A? | Works ad hoc, but doesn't scale — every new agent pair needs custom integration; A2A standardizes discovery (Agent Cards) and task lifecycle so any compliant agents interoperate without bespoke glue. |

---

**Reminder**: this is a study aid — double check any numeric/benchmark specifics (exact benchmark scores, current SOTA models) closer to your interview since these move fast.
