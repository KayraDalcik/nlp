# Turkish Legal RAG System — Technical Report

**Project**: Turkish Legal Retrieval-Augmented Generation (RAG)  
**Date**: April 2026  
**Status**: Development & Evaluation Phase

---

## 1. DATASETS

### 1.1 Primary Dataset: Turkish Law Chatbot (HuggingFace)

**Source**: Hugging Face Datasets - `turkish-law-chatbot`

**Composition**:
- **Training Split**: Q&A pairs from Turkish legal texts
- **Test Split**: Evaluation questions (if available)
- **Total Samples**: ~12,000+ documents after preprocessing
- **Languages**: Turkish (100%)

**Data Characteristics**:
- **Domain**: Turkish legal domain (criminal law, civil law, administrative law)
- **Format**: Question-Answer pairs with optional context
- **Quality**: Pre-curated legal-specific Q&A pairs
- **Coverage**: Multiple legal domains and practice areas

### 1.2 Benchmark Dataset

**Construction**:
- **Source**: Extracted from HuggingFace test split (or random sample from training)
- **Size**: 100 questions
- **Format**: JSONL with fields:
  ```json
  {
    "id": "bench_0001",
    "question": "...",
    "reference_answer": "...",
    "source_split": "test|train",
    "original_id": "..."
  }
  ```
- **Purpose**: Ablation study and pipeline evaluation
- **Location**: `data/benchmark/test_questions.jsonl`

### 1.3 Processed Data Pipeline

```
Raw Data (HF) → Download → Preprocess → 3 Outputs:
├── hf_processed.jsonl (Q&A pairs, cleaned)
├── retrieval_corpus.jsonl (12,105 documents for indexing)
└── finetune_data.jsonl (ChatML format for LLM fine-tuning)
```

**Preprocessing Steps**:
1. Text cleaning: UTF-8 normalization, whitespace collapsing
2. Field standardization: Handle column name variations
3. Deduplication: Remove duplicate context strings
4. Formatting: Convert to JSONL for efficient processing

---

## 2. METHODS — RETRIEVAL & INDEXING

### 2.1 Dense Retrieval — FAISS-based Semantic Search

**Component**: `src/pipeline/baseline_rag.py` → `FAISSRetriever` class

**Mechanism**:
1. **Embedding Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
   - Architecture: XLM-RoBERTa backbone with sentence-transformer fine-tuning
   - Embedding Dimension: 768
   - Language Support: 50+ languages including Turkish
   - Normalization: L2 normalization for cosine similarity

2. **Index Type**: FAISS `IndexFlatIP` (Inner Product for cosine similarity)
   - Algorithm: Exact search (flat index, no approximation)
   - Similarity Metric: Cosine similarity (via normalized inner product)
   - Scalability: O(n) search complexity; suitable for 10K–100K documents

3. **Indexing Process**:
   ```
   Documents → Batch Encoding (batch_size=32) → Embeddings (768-dim) 
   → FAISS Index Addition → Serialization (index.faiss + metadata.json)
   ```

4. **Search Process**:
   ```
   Query → Encode to 768-dim vector → FAISS top-k search 
   → Return (doc_id, text, similarity_score)
   ```

**Performance**:
- Encoding Time: ~12-15 seconds per batch (32 documents) on CPU
- Index Storage: ~40 MB for 12,100 documents
- Metadata Storage: ~4.1 MB (JSON-serialized doc IDs and texts)

### 2.2 Sparse Retrieval — BM25 Keyword Search

**Component**: `src/retrieval/bm25_retriever.py` → `BM25Retriever` class

**Mechanism**:
1. **Algorithm**: Okapi BM25 (Probabilistic Relevance Framework)
   - Implementation: `rank_bm25` library
   - Parameters:
     - k1 = 1.5 (term frequency saturation)
     - b = 0.75 (length normalization: 0 = off, 1 = full)

2. **Turkish Tokenization**:
   ```python
   Custom tokenizer: lowercase → remove punctuation → split on whitespace
   Stop-word filtering: 30+ Turkish stop words (bir, bu, ve, ile, etc.)
   Token length: minimum 2 characters
   ```

3. **Index Construction**:
   ```
   Documents → Turkish Tokenization → Tokenized Corpus 
   → BM25 Index (TF-IDF with BM25 weighting)
   ```

4. **Search Process**:
   ```
   Query → Tokenize → BM25 Scoring → Top-k documents with scores
   ```

**Advantages**:
- Keyword-based: Captures exact lexical matches
- Fast: O(n) scoring, linear in corpus size
- Interpretable: Score based on term frequency and inverse document frequency
- Orthogonal to Dense: Different matching signals than embeddings

### 2.3 Hybrid Retrieval — Reciprocal Rank Fusion (RRF)

**Component**: `src/pipeline/hybrid_rag.py` → `HybridRAG` class + `reciprocal_rank_fusion()`

**Motivation**: Combine complementary strengths of dense (semantic) and sparse (lexical) retrieval

**Fusion Strategy**: Reciprocal Rank Fusion (RRF)

```python
Score(doc) = Σ [ 1 / (k + rank(doc, method)) ]

where:
  - k: constant (typically 60)
  - rank(doc, method): 0-based rank in dense/sparse results
  - Sum over all retrieval methods
```

**Pipeline**:
1. **Dense Retrieval**: FAISS top-20 candidates
2. **Sparse Retrieval**: BM25 top-20 candidates
3. **RRF Scoring**: Combine and re-rank
4. **Final Selection**: Top-5 documents (configurable)

**Configuration Parameters**:
```yaml
top_k: 5                          # Final documents returned
dense_weight: 0.6                 # Not used (RRF is unweighted)
sparse_weight: 0.4                # Not used (RRF is unweighted)
rrf_k: 60                         # RRF constant parameter
bm25_candidate_k: 20              # Candidates from each retriever
```

**Advantages of RRF**:
- Parameter-free fusion (no weighting tuning needed)
- Robust: Requires agreement across methods
- Empirically strong: Widely used in IR systems

---

## 3. LANGUAGE MODELS

### 3.1 Selected LLM: Qwen2.5-3B-Instruct

**Model**: `Qwen/Qwen2.5-3B-Instruct` (Quantized Qwen 2.5 3B)

**Selection Rationale**:
1. **Lightweight**: 3B parameters
   - Fits on consumer GPU/CPU with modest resources
   - Inference latency: ~500–1000ms per query on CPU
   
2. **Multilingual Support**: Includes Turkish
   - Pre-trained on diverse language corpora
   - Instruction-tuned for Q&A tasks
   
3. **Task Suitability**: Instruction-following capability
   - Designed for chat/Q&A tasks
   - Can follow system prompts and provide structured answers
   
4. **Cost-Effective**: Fully open-source
   - No API fees
   - Run locally (privacy-preserving)
   - No rate limits

### 3.2 Model Loading & Quantization

**Component**: `src/pipeline/baseline_rag.py` → `LLMGenerator` class

**Configuration**:
```python
Model: Qwen/Qwen2.5-3B-Instruct
Precision: float16 (GPU) / float32 (CPU)
Device Map: "auto"
Memory Optimization: low_cpu_mem_usage=True
```

**Generation Parameters**:
```yaml
max_new_tokens: 512               # Maximum output length
temperature: 0.1                  # Low temperature for determinism
do_sample: True                   # Sampling-based generation
pad_token_id: eos_token_id        # Proper padding
```

### 3.3 Prompt Engineering

**System Prompt** (Turkish):
```
"Sen bir Türk hukuku uzmanısın. Aşağıdaki hukuki belgeleri 
kullanarak soruyu yanıtla. Yanıtında kaynak göster 
(örn: 'Kaynak 1'e göre...'). Eğer yanıtı bilmiyorsan 
'Bu konuda yeterli bilgi bulunamadı.' de."
```

**Prompt Structure**:
```
[System Prompt]
[Retrieved Legal Documents] (context)
[User Question]
[Model generates answer]
```

---

## 4. EMBEDDING & SEMANTIC LAYER

### 4.1 Embedding Model Architecture

**Selected Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

**Technical Specifications**:
- **Base Architecture**: XLM-RoBERTa (cross-lingual RoBERTa)
- **Parameters**: ~279M
- **Output Dimension**: 768
- **Pooling**: Mean pooling over token embeddings
- **Normalization**: L2 normalization for cosine similarity

**Training**:
- Pre-trained on multilingual corpora
- Fine-tuned on parallel sentence pairs (MultiNLI, etc.)
- Optimized for semantic similarity across 50+ languages

### 4.2 Implementation Strategy

**Batching for Efficiency**:
```python
Batch Size: 32
Show Progress: True (tqdm progress bar)
Normalize Embeddings: True (L2 norm)
Convert to Numpy: Yes (for FAISS compatibility)
```

**Encoding Pipeline**:
```
Input Documents (list of strings) 
→ Tokenization (subword tokens)
→ BERT-style Encoding
→ Pooling (mean over tokens)
→ L2 Normalization
→ 768-dim float32 vectors
```

### 4.3 Multilingual Properties

**Why This Model for Turkish?**
- Trained on 50+ languages including Turkish
- Cross-lingual transfer: Learns shared semantic space
- Enables document-query matching across language variants
- Better than English-only models for non-English languages

**Evaluation Metrics**: Benchmark on STS (Semantic Textual Similarity)

---

## 5. RERANKING LAYER

### 5.1 Current Approach: RRF Fusion (Retrieval-Level)

**Component**: `reciprocal_rank_fusion()` function in `hybrid_rag.py`

**How It Works**:
- Combines FAISS and BM25 ranked lists
- Reorders candidates based on RRF score
- Simpler than neural reranking; effective for 2-stage hybrid systems

**Limitations**:
- Static scoring: Doesn't learn domain-specific ranking
- No cross-document comparisons

### 5.2 Planned Future Work: Cross-Encoder Reranking

**Planned Component**: `src/reranker/` (currently empty)

**Concept**:
- **Cross-Encoder Model**: E.g., `sentence-transformers/cross-encoder-multilingual-MiniLMv2L12`
- **Input**: (Query, Document) pairs
- **Output**: Relevance score [0, 1]
- **Use Case**: Rerank top-k from hybrid retrieval before LLM

**Implementation Roadmap**:
```python
# Pseudo-code
reranker = CrossEncoder("sentence-transformers/cross-encoder-...")
scores = reranker.predict([
    (query, doc["text"]) for doc in hybrid_results[:50]
])
top_docs = sorted_by_score(scores)[:5]
```

**Advantages**:
- Query-aware: Context-specific scoring
- Fine-tuned: Can adapt to domain (legal)
- Higher accuracy: Typically better than RRF alone

---

## 6. EVALUATION FRAMEWORK

### 6.1 Metrics

**Component**: `src/evaluation/evaluator.py` → `RAGEvaluator` class

#### ROUGE-L (Reference-Oriented Understudy for Gisting Evaluation)

**Metric**: Longest common subsequence-based F1 score
- **Precision**: How much of prediction overlaps with reference
- **Recall**: How much of reference is covered by prediction
- **F1**: Harmonic mean

**Use**: Lexical overlap evaluation

#### BERTScore (Contextual Token-Level Similarity)

**Metric**: Semantic similarity using contextual embeddings
- Computes similarity between prediction and reference tokens
- Uses distributional semantics (not just lexical overlap)
- Language-specific: Turkish fine-tuned (`lang='tr'`)

**Use**: Semantic quality evaluation

#### RAGAS (Retrieval-Augmented Generation Assessment)

**Components** (optional, requires LLM):
1. **Faithfulness**: Does answer follow the context?
2. **Answer Relevance**: Does answer address the question?
3. **Context Recall**: Do retrieved documents contain answer?

**Status**: Disabled by default (requires OpenAI API or local LLM)

### 6.2 Evaluation Protocol

**Ablation Study Setup**:

| Experiment | Dense | Sparse | LLM | Purpose |
|-----------|-------|--------|-----|---------|
| Baseline | FAISS | — | Optional | Dense-only retrieval baseline |
| Hybrid | FAISS | BM25 | Optional | Combine dense + sparse |

**Evaluation Modes**:
1. **Retrieval-Only** (default): Evaluate document ranking without LLM
   - Use top-1 retrieved document as answer
   - Faster evaluation, no LLM costs
   
2. **Full RAG** (--llm flag): Evaluate end-to-end pipeline
   - LLM generates answer from retrieved context
   - Measures generation quality
   - Slower, requires model inference

**Benchmark**:
- **Dataset**: 100 diverse Turkish legal questions
- **Metrics Tracked**:
  - ROUGE-L F1, precision, recall
  - BERTScore F1, precision, recall
  - Average retrieval time (ms)
  - Average generation time (ms)
  - Top-1 retrieval similarity score

---

## 7. FINE-TUNING STRATEGY (Planned)

### 7.1 LLM Fine-Tuning: QLoRA Approach

**Component**: `src/training/` (planned implementation)

**Motivation**: Adapt Qwen2.5-3B to legal domain question-answering

**Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Base Model**: Qwen2.5-3B-Instruct (frozen, 4-bit quantized)
- **Adapters**: LoRA modules with low rank (~8–16)
- **Training Data**: `finetune_data.jsonl` (~3,000+ ChatML formatted examples)

**Configuration** (planned):
```yaml
learning_rate: 5e-4
batch_size: 8 (per GPU) / 1 (CPU)
epochs: 3
LoRA rank: 8
LoRA alpha: 16
target_modules: ["q_proj", "v_proj"]  # Qwen attention projections
data_format: ChatML
```

**Implementation Libraries**:
- `peft` (PEFT — Parameter-Efficient Fine-Tuning)
- `trl` (TRL — Transformer Reinforcement Learning)
- `bitsandbytes` (4-bit quantization)

**Expected Benefits**:
- Better understanding of legal terminology
- Improved answer relevance to domain-specific questions
- More concise, well-formatted responses

### 7.2 Embedding Model Fine-Tuning (Future)

**Concept**: Domain-specific embedding model
- Use legal Q&A pairs as training signal
- Contrastive learning: Positive (Q, A) pairs vs. negative samples
- Improves retrieval precision for legal domain

---

## 8. IMPLEMENTATION STATUS & ROADMAP

### 8.1 Completed

- [x] Data preprocessing pipeline
- [x] Dataset creation and benchmark generation
- [x] Dense retrieval (FAISS + Sentence-Transformers)
- [x] Sparse retrieval (BM25 with Turkish tokenization)
- [x] Hybrid retrieval (RRF fusion)
- [x] LLM integration (Qwen2.5-3B-Instruct)
- [x] Basic evaluation framework (ROUGE-L, BERTScore)
- [x] Ablation study infrastructure

### 8.2 In Progress

- [ ] FAISS index building (currently running: ~1.5 hours for 12K docs on CPU)
- [ ] Initial evaluation runs (Baseline vs. Hybrid)
- [ ] Performance benchmarking

### 8.3 Planned

- [ ] Cross-encoder reranking layer
- [ ] LLM fine-tuning with QLoRA
- [ ] Embedding model domain adaptation
- [ ] RAGAS evaluation (requires LLM API)
- [ ] WandB experiment tracking
- [ ] Gradio demo interface

### 8.4 Potential Improvements

1. **Hybrid Retrieval Enhancement**:
   - Learned weight fusion (instead of RRF)
   - ColBERT-style dense-sparse interaction

2. **LLM Optimization**:
   - Fine-tuning on legal QA
   - Prompt optimization for better citations
   - RAG-specific fine-tuning (REALM, RetrieverTraining)

3. **Domain-Specific Embeddings**:
   - Contrastive learning on legal corpus
   - Multi-task learning (similarity + ranking)

4. **Scalability**:
   - IndexIVF for 100K+ documents
   - Distributed inference
   - Batched LLM generation

---

## 9. TECHNICAL STACK

**Core Libraries**:
```
torch>=2.1.0                          # Deep learning framework
transformers>=4.40.0                  # NLP models
sentence-transformers>=3.0.0          # Embedding models
faiss-cpu>=1.7.4                      # Dense indexing
rank_bm25>=0.2.2                      # Sparse retrieval
```

**Evaluation**:
```
rouge-score>=0.1.2
bert-score>=0.3.13
ragas>=0.1.14
evaluate>=0.4.1
```

**Optimization** (for fine-tuning):
```
peft>=0.10.0                          # LoRA adapters
bitsandbytes>=0.43.0                  # Quantization
trl>=0.8.6                            # Training utils
```

**Tracking & Demo**:
```
wandb>=0.17.0                         # Experiment tracking
gradio>=4.36.0                        # WebUI
```

---

## 10. SYSTEM ARCHITECTURE DIAGRAM

```
+-----------------------------------------------------+
|                     USER QUERY                     |
+-----------------------------------------------------+
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
   +---------+      +---------+    +----------+
   | Embed   |      | Tokenize|    | Keyword  |
   | Query   |      | Query   |    | Extrac.  |
   +----+----+      +----+----+    +----+-----+
        |                |             |
        v                v             v
   +----------+     +----------+  +--------+
   |FAISS Dense |   |  BM25      |  |  Keyword|
   | Retrieval  |   | Retrieval  |  |  Index  |
   |(12K docs)  |   |(12K docs)  |  |         |
   +----+-------+   +----+-------+  +----+----+
        |                |              |
        +--------+-------+--------+-----+
                 |
        +--------+--------+
        |  RRF Fusion     |
        |  Re-ranking     |
        +--------+--------+
                 |
        +--------+----------------+
        | Retrieved Documents (top-5)
        +--------+----------------+
                 |
        +--------+------------------------------+
        |  LLM (Qwen2.5-3B-Instruct)        |
        |  Generate Answer with Context    |
        +--------+------------------------------+
                 |
        +--------+----------------------+
        |  Final Answer with Sources    |
        +---------------------------------+
```

---

## 11. CONCLUSION

The Turkish Legal RAG system implements a hybrid retrieval architecture combining dense (semantic) and sparse (lexical) retrieval methods, fused via RRF, to provide contextually relevant documents for LLM-based answer generation. The system uses lightweight, domain-aware models optimized for Turkish legal text and supports iterative ablation studies to evaluate the contribution of each component. Planned enhancements include cross-encoder reranking, LLM fine-tuning, and domain-specific embedding adaptation to further improve retrieval and generation quality.

---

## 12. REFERENCES & RESOURCES

- **FAISS**: Facebook AI Similarity Search
  - Paper: Johnson et al., "Billion-scale similarity search with GPUs" (2017)
  
- **BM25**: Okapi Best Matching Function
  - Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)

- **Sentence-Transformers**: Pre-trained Transformer models for semantic similarity
  - Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)

- **RRF**: Reciprocal Rank Fusion for combining IR systems
  - Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009)

- **QLoRA**: Efficient Fine-tuning of Quantized LLMs
  - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

- **RAGAS**: RAG Assessment Framework
  - ES et al., "RAGAS: A Referential Relation-Free Metric for Generation-based QA Tasks" (2023)

---

Document Version: 1.0  
Last Updated: April 2026  
Prepared by: Technical Team
