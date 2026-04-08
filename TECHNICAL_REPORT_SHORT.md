# Turkish Legal RAG System — Technical Report (Concise)

**Project**: Turkish Legal Retrieval-Augmented Generation (RAG)  
**Date**: April 2026

---

## 1. DATASETS

### Primary Dataset: Turkish Law Chatbot (HuggingFace)
- **Source**: Hugging Face Datasets - `turkish-law-chatbot`
- **Size**: ~12,100 documents after preprocessing
- **Domain**: Turkish legal domain (criminal, civil, administrative law)
- **Format**: Question-Answer pairs with optional context
- **Split**: Train/Test splits from HuggingFace

### Benchmark Dataset
- **Size**: 100 Turkish legal questions
- **Format**: JSONL with question, reference_answer, source_split
- **Purpose**: Ablation study and pipeline evaluation
- **Location**: `data/benchmark/test_questions.jsonl`

### Processed Data Pipeline
1. **hf_processed.jsonl**: Cleaned Q&A pairs (~6.2 MB)
2. **retrieval_corpus.jsonl**: 12,105 documents for indexing (~4.1 MB)
3. **finetune_data.jsonl**: ChatML format for LLM fine-tuning (~6.7 MB)

**Preprocessing Steps**:
- Text cleaning (UTF-8 normalization, whitespace collapsing)
- Field standardization (handle column name variations)
- Deduplication (remove duplicate contexts)
- JSONL conversion for efficient processing

---

## 2. FINE-TUNING METHODS

### Planned: QLoRA (Quantized Low-Rank Adaptation)

**Rationale**: Adapt Qwen2.5-3B to legal domain without full retraining

**Method**:
- **Base Model**: Qwen2.5-3B-Instruct (frozen, 4-bit quantized)
- **Adapters**: LoRA modules (rank ~8-16)
- **Training Data**: finetune_data.jsonl (~3,000+ ChatML examples)

**Configuration**:
```yaml
learning_rate: 5e-4
batch_size: 1 (CPU) / 8 (GPU)
epochs: 3
LoRA rank: 8
LoRA alpha: 16
target_modules: ["q_proj", "v_proj"]
data_format: ChatML
```

**Implementation Libraries**:
- PEFT (Parameter-Efficient Fine-Tuning) - LoRA adapters
- TRL (Transformer Reinforcement Learning) - training utils
- Bitsandbytes - 4-bit quantization

**Expected Benefits**:
- Better legal terminology understanding
- Improved domain-specific answer relevance
- Reduced hallucinations in legal context
- ~1% trainable parameters vs. 100% full fine-tuning

---

## 3. LLM MODEL & SELECTION RATIONALE

### Selected Model: Qwen2.5-3B-Instruct

**Model Specifications**:
- Parameters: 3 Billion
- Architecture: Transformer-based
- Quantization: Float16 (GPU) / Float32 (CPU)
- Pooling: Mean pooling over tokens

**Selection Rationale**:

1. **Lightweight Design**
   - Fits on consumer CPU/GPU with modest resources
   - Inference latency: ~500-1000ms per query on CPU
   - No GPU required for deployment

2. **Multilingual Support**
   - Pre-trained on 50+ languages including Turkish
   - Cross-lingual transfer capabilities
   - Better than English-only models for Turkish legal text

3. **Task Suitability**
   - Instruction-tuned for Q&A and chat tasks
   - Can follow system prompts and context constraints
   - Designed for RAG scenarios

4. **Cost-Effective**
   - Fully open-source (no API fees)
   - Privacy-preserving (local inference)
   - No rate limits or external dependencies

### Generation Parameters
```yaml
max_new_tokens: 512
temperature: 0.1 (low for determinism)
do_sample: True
pad_token_id: eos_token_id
```

### Prompt Engineering
System Prompt (Turkish):
```
"Sen bir Türk hukuku uzmanısın. Aşağıdaki hukuki belgeleri 
kullanarak soruyu yanıtla. Yanıtında kaynak göster 
(örn: 'Kaynak 1'e göre...'). Eğer yanıtı bilmiyorsan 
'Bu konuda yeterli bilgi bulunamadı.' de."
```

---

## 4. EMBEDDING & RERANKING LAYER

### 4.1 Embedding Model

**Selected**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

**Technical Details**:
- Architecture: XLM-RoBERTa backbone with sentence-transformer fine-tuning
- Output Dimension: 768
- Normalization: L2 (for cosine similarity)
- Languages: 50+ including Turkish

**Why This Model**:
- Cross-lingual semantic space (Turkish + English + other languages)
- Optimized for semantic similarity tasks
- Pre-trained on multilingual corpora and parallel sentence pairs
- Better cross-lingual transfer than English-only embeddings

**Usage**:
- Batch encoding: 32 documents per batch
- Processing time: ~12-15 seconds per batch on CPU
- Storage: ~40 MB for 12,100 documents

### 4.2 Dense Retrieval (FAISS)

**Implementation**: FAISS `IndexFlatIP` (Inner Product for cosine)

**Pipeline**:
1. Encode documents: text → 768-dim vectors (normalized)
2. Build FAISS index: exact search (no approximation)
3. Search: query → encode → FAISS top-k

**Performance**:
- Similarity metric: Cosine (via normalized inner product)
- Complexity: O(n) search
- Suitable for: 10K-100K documents

### 4.3 Sparse Retrieval (BM25)

**Implementation**: Okapi BM25 with Turkish tokenization

**Method**:
- Algorithm: Probabilistic Relevance Framework
- Parameters: k1=1.5 (term saturation), b=0.75 (length norm)
- Tokenizer: Custom Turkish (lowercase, punctuation removal, stop-word filtering)
- Stop-words: 30+ Turkish words (bir, bu, ve, ile, etc.)

**Advantages**:
- Captures exact lexical matches
- Fast: O(n) scoring
- Orthogonal to dense retrieval (different signals)

### 4.4 Hybrid Retrieval (RRF Fusion)

**Method**: Reciprocal Rank Fusion (RRF)

**Formula**:
```
Score(doc) = Σ [ 1 / (k + rank(doc, method)) ]
```

**Pipeline**:
1. Dense retrieval: FAISS top-20 candidates
2. Sparse retrieval: BM25 top-20 candidates
3. RRF scoring: Combine and re-rank all candidates
4. Final output: Top-5 documents

**Advantages**:
- Parameter-free fusion (no weight tuning)
- Robust: Requires agreement across methods
- Empirically strong in IR literature

### 4.5 Reranking Layer (Planned)

**Planned**: Cross-Encoder neural reranking

```python
# Pseudo-code
reranker = CrossEncoder("sentence-transformers/cross-encoder-multilingual-MiniLMv2L12")
scores = reranker.predict([
    (query, doc["text"]) for doc in hybrid_results[:50]
])
top_docs = sorted_by_score(scores)[:5]
```

**Advantages**:
- Query-aware scoring
- Domain-specific fine-tuning possible
- Typically improves accuracy over RRF alone

---

## 5. OTHER METHODS IMPLEMENTED & PLANNED

### Implemented

1. **Data Pipeline**
   - Preprocessing: cleaning, deduplication, standardization
   - JSONL-based storage for efficiency
   - Benchmark dataset construction (100 test questions)

2. **Hybrid Retrieval System**
   - Dense (FAISS) + Sparse (BM25) combination
   - RRF-based fusion strategy
   - Configurable top-k parameters

3. **Evaluation Framework**
   - ROUGE-L metric (lexical overlap F1)
   - BERTScore (semantic similarity via contextual embeddings)
   - Retrieval time and retrieval score tracking
   - Ablation study infrastructure (Baseline vs. Hybrid)

4. **LLM Integration**
   - Qwen2.5-3B-Instruct loading and inference
   - Context-aware prompt engineering
   - Source attribution in answers

### Planned

1. **LLM Fine-tuning**
   - QLoRA-based adaptation to legal domain
   - Training on finetune_data.jsonl
   - Expected improvements in domain accuracy

2. **Cross-Encoder Reranking**
   - Neural reranker for top-50 candidates
   - Domain-specific fine-tuning option
   - Improves final document ranking

3. **Embedding Domain Adaptation**
   - Contrastive learning on legal Q&A pairs
   - Domain-specific semantic space
   - Improved legal document-query matching

4. **RAGAS Evaluation**
   - Faithfulness: Does answer follow context?
   - Answer Relevance: Addresses question?
   - Context Recall: Documents contain answer?
   - Requires OpenAI API or local LLM

5. **Scalability Enhancements**
   - IndexIVF for 100K+ documents
   - Distributed inference
   - Batched LLM generation

6. **Demo Interface**
   - Gradio web UI
   - Live question answering
   - Retrieved document visualization

---

## 6. SUMMARY TABLE

| Component | Status | Method | Rationale |
|-----------|--------|--------|-----------|
| **Datasets** | Complete | Turkish Law Chatbot + Benchmark | Domain-specific, pre-curated legal Q&A |
| **Embedding** | Complete | Multilingual RoBERTa (768-dim) | Cross-lingual Turkish support |
| **Dense Retrieval** | Complete | FAISS IndexFlatIP | Semantic search, exact matching |
| **Sparse Retrieval** | Complete | BM25 + Turkish tokenization | Lexical matching, keyword search |
| **Retrieval Fusion** | Complete | RRF (Reciprocal Rank Fusion) | Parameter-free, robust combination |
| **LLM** | Complete | Qwen2.5-3B-Instruct | Lightweight, multilingual, cost-effective |
| **Generation** | Complete | Prompt engineering + context | Source attribution, domain-aware |
| **LLM Fine-tuning** | Planned | QLoRA (4-bit quantized) | Domain adaptation, efficiency |
| **Reranking** | Planned | Cross-Encoder neural | Query-aware ranking |
| **Evaluation** | Partial | ROUGE-L, BERTScore, RAGAS | Comprehensive RAG assessment |
| **Scaling** | Planned | IndexIVF, Batching | Handle 100K+ documents |

---

## 7. KEY TECHNICAL CHOICES

1. **Why Hybrid (Dense + Sparse)?**
   - Dense: Semantic understanding
   - Sparse: Exact term matches
   - Hybrid: Best of both worlds, especially for legal terminology

2. **Why Turkish-aware tokenization?**
   - Handles Turkish suffixes and morphology
   - Turkish stop-word filtering reduces noise
   - Better than language-agnostic tokenization

3. **Why QLoRA over full fine-tuning?**
   - 1% trainable parameters (efficient)
   - Fits on CPU/consumer GPU
   - Reduces memory 4x via quantization
   - Maintains model quality

4. **Why Qwen2.5-3B over larger models?**
   - 3B is sweet spot: Quality vs. Resource trade-off
   - Multilingual (especially Turkish support)
   - Local deployment (no API dependencies)
   - Lower inference latency (~500ms vs. 2-5s for 13B+)

5. **Why RRF over learned weighting?**
   - Parameter-free: No hyperparameter tuning needed
   - Robust: Works well empirically
   - Interpretable: Clear ranking combination logic
   - Can upgrade to learned weights later if needed

---

**Document Version**: 1.0  
**Last Updated**: April 2026
