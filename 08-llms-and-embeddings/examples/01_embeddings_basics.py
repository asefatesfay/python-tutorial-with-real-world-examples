"""
Embeddings and Semantic Search

Learn how to convert text to vectors and find similar content.
Focus: Understanding embeddings, similarity, and practical applications.

Install: poetry add numpy sentence-transformers
Run: poetry run python 08-llms-and-embeddings/examples/01_embeddings_basics.py
"""

import math
import random
from typing import List, Dict, Tuple
from collections import defaultdict


# ============================================================================
# 1. What Are Embeddings?
# ============================================================================

def demo_embeddings_intuition():
    """
    Embeddings: Converting words/sentences to vectors (lists of numbers)
    
    INTUITION - The Map Analogy:
    
    Imagine you want to find cities similar to "San Francisco":
    
    Bad approach: String comparison
    - "San Francisco" vs "New York" → 0% match (no common letters!)
    - Useless for finding similar cities
    
    Good approach: Map cities to coordinates
    - San Francisco → (tech=0.9, finance=0.7, weather=0.8, size=0.6)
    - New York     → (tech=0.7, finance=0.9, weather=0.3, size=0.9)
    - Austin       → (tech=0.8, finance=0.4, weather=0.7, size=0.4)
    
    Now we can measure similarity:
    - San Francisco ↔ Austin: High similarity (both tech hubs, warm weather)
    - San Francisco ↔ New York: Medium (both large, expensive, but different vibe)
    
    EMBEDDINGS DO THIS FOR TEXT:
    
    Words:
    - "king" → [0.2, 0.9, 0.1, ...]  (385 dimensions)
    - "queen" → [0.2, 0.8, 0.1, ...]
    - "man" → [0.5, 0.1, 0.2, ...]
    
    Famous example: king - man + woman ≈ queen
    (Vector arithmetic captures meaning!)
    
    Sentences:
    - "I love pizza" → [0.1, 0.3, 0.8, ..., 0.2]  (385 dimensions)
    - "Pizza is great" → [0.1, 0.3, 0.7, ..., 0.2]  (similar vectors!)
    - "I hate rain" → [-0.4, 0.1, 0.2, ..., 0.9]  (different vector)
    
    WHY EMBEDDINGS ARE POWERFUL:
    
    1. Capture Meaning:
       "car" and "automobile" have similar vectors (same meaning)
       "car" and "pizza" have different vectors (different meaning)
    
    2. Enable Similarity Search:
       Find documents similar to a query (semantic search)
       
    3. Work Across Languages:
       "hello" (English) ≈ "hola" (Spanish) in multilingual embeddings
    
    4. Compress Information:
       1000-word document → 385 numbers (efficient!)
    
    Real Applications:
    - Search engines: Find relevant documents
    - Recommendation: "Users who liked X also liked Y"
    - Clustering: Group similar articles
    - Classification: Sentiment analysis, spam detection
    - Question answering: Find relevant passages
    """
    print("=" * 70)
    print("1. What Are Embeddings?")
    print("=" * 70)
    print()
    print("💭 INTUITION: Finding Similar Cities")
    print()
    print("   ❌ Bad: String comparison")
    print("      'San Francisco' vs 'Austin' → 0% match")
    print("      (No common characters, but both are tech hubs!)")
    print()
    print("   ✅ Good: Map to meaningful coordinates")
    print("      San Francisco → (tech=0.9, finance=0.7, weather=0.8)")
    print("      Austin        → (tech=0.8, finance=0.4, weather=0.7)")
    print("      Similarity: 85% ✓ (both tech hubs, warm weather)")
    print()
    
    # Simplified city embeddings (3D for demonstration)
    cities = {
        "San Francisco": [0.9, 0.7, 0.8],  # [tech, finance, weather]
        "New York": [0.7, 0.9, 0.3],
        "Austin": [0.8, 0.4, 0.7],
        "Seattle": [0.9, 0.5, 0.4],
        "Miami": [0.2, 0.6, 0.9]
    }
    
    print("🌆 City Embeddings (3D for simplicity):")
    print("   Dimensions: [tech_hub, finance_center, warm_weather]")
    print()
    for city, embedding in cities.items():
        print(f"   {city:15s} → {embedding}")
    print()
    
    # Calculate similarity (cosine similarity)
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x ** 2 for x in a))
        magnitude_b = math.sqrt(sum(y ** 2 for y in b))
        return dot_product / (magnitude_a * magnitude_b)
    
    # Find similar cities to San Francisco
    sf_embedding = cities["San Francisco"]
    similarities = {}
    
    for city, embedding in cities.items():
        if city != "San Francisco":
            sim = cosine_similarity(sf_embedding, embedding)
            similarities[city] = sim
    
    print("🔍 Cities Similar to San Francisco:")
    for city, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(sim * 50)
        print(f"   {city:15s} {sim:.3f} {bar}")
    print()
    
    print("💡 Why This Works:")
    print("   • Austin: High tech, warm → Similar to SF")
    print("   • Seattle: High tech, but cold → Somewhat similar")
    print("   • Miami: Warm but not tech hub → Less similar")
    print()
    
    print("📝 TEXT EMBEDDINGS WORK THE SAME WAY:")
    print()
    print("   Simple word embeddings (conceptual):")
    print("   'king'   → [0.2, 0.9, 0.1, 0.4, ...]  (385 dims)")
    print("   'queen'  → [0.2, 0.8, 0.1, 0.5, ...]  (similar!)")
    print("   'man'    → [0.5, 0.1, 0.2, 0.3, ...]")
    print("   'woman'  → [0.5, 0.1, 0.2, 0.4, ...]")
    print("   'pizza'  → [0.1, 0.2, 0.9, 0.1, ...]  (different!)")
    print()
    print("   Famous vector math:")
    print("   king - man + woman ≈ queen")
    print("   (Embeddings capture relationships!)")
    print()
    
    print("🎯 Real-World Text Example:")
    print()
    
    # Simulated sentence embeddings (3D for demo)
    sentences = {
        "I love pizza": [0.8, 0.2, 0.1],     # [food_positive, weather, tech]
        "Pizza is great": [0.8, 0.2, 0.0],   # Similar to above
        "Terrible pizza": [0.7, 0.2, -0.5],  # Food but negative
        "It's raining": [0.0, 0.9, 0.0],     # About weather
        "Python is awesome": [0.1, 0.0, 0.9] # About tech
    }
    
    print("   Sentence Embeddings (simplified):")
    for sent, emb in sentences.items():
        print(f"   '{sent:20s}' → {emb}")
    print()
    
    # Find similar to "I love pizza"
    query = "I love pizza"
    query_emb = sentences[query]
    
    print(f"   Find similar to: '{query}'")
    print()
    
    sent_similarities = {}
    for sent, emb in sentences.items():
        if sent != query:
            sim = cosine_similarity(query_emb, emb)
            sent_similarities[sent] = sim
    
    for sent, sim in sorted(sent_similarities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(sim * 50)
        print(f"   {sim:.3f} {bar} '{sent}'")
    print()
    
    print("💡 Notice:")
    print("   • 'Pizza is great' is most similar (same topic, sentiment)")
    print("   • 'Terrible pizza' is somewhat similar (same topic, different sentiment)")
    print("   • 'Python is awesome' is least similar (completely different)")
    print()
    
    print("🚀 REAL APPLICATIONS:")
    print()
    print("   1️⃣  Semantic Search:")
    print("      Query: 'how to fix car'")
    print("      Finds: 'automobile repair guide' (different words, same meaning!)")
    print()
    print("   2️⃣  Recommendation:")
    print("      You liked: 'Inception' (movie)")
    print("      Similar embeddings: 'Interstellar', 'The Matrix'")
    print()
    print("   3️⃣  Question Answering:")
    print("      Question: 'What's the capital of France?'")
    print("      Finds passage: 'Paris is France's capital city...'")
    print()
    print("   4️⃣  Duplicate Detection:")
    print("      'How do I reset my password?' ≈")
    print("      'I forgot my password, what should I do?'")


# ============================================================================
# 2. Computing Similarity
# ============================================================================

def demo_similarity_metrics():
    """
    Different ways to measure similarity between vectors.
    
    INTUITION - Measuring Friendship:
    
    You have two friends, Alice and Bob. How similar are they?
    
    Method 1: Euclidean Distance (Straight-line distance)
    - Like measuring distance on a map
    - Shorter distance = more similar
    - Problem: Sensitive to vector magnitude
    
    Method 2: Cosine Similarity (Angle between vectors)
    - Like checking if you're pointing in the same direction
    - Same direction = similar (even if different "intensity")
    - Range: -1 (opposite) to +1 (identical)
    - ✅ Most common for text embeddings!
    
    Method 3: Dot Product (Overlap of vectors)
    - Like measuring how much vectors "agree"
    - Higher = more similar
    - Problem: Not normalized (longer vectors score higher)
    
    Why Cosine Similarity Wins for Text:
    
    Example: Document length doesn't matter
    - "I love cats" → [0.5, 0.5]
    - "I really truly deeply love cats a lot" → [5.0, 5.0]
    
    Euclidean distance: Far apart (different lengths!)
    Cosine similarity: Identical! (same direction/meaning)
    
    This is exactly what we want for text!
    """
    print("\n" + "=" * 70)
    print("2. Computing Similarity")
    print("=" * 70)
    print()
    print("💭 INTUITION: Measuring How Similar Two Things Are")
    print()
    
    # Two example vectors
    vec_a = [3, 4]  # Point A
    vec_b = [6, 8]  # Point B (same direction, twice as long)
    vec_c = [4, 3]  # Point C (different direction)
    
    print("   Example Vectors (2D for visualization):")
    print(f"   A = {vec_a}  (shorter)")
    print(f"   B = {vec_b}  (longer, same direction as A)")
    print(f"   C = {vec_c}  (different direction)")
    print()
    
    # Method 1: Euclidean Distance
    def euclidean_distance(a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance between vectors."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    # Method 2: Cosine Similarity
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x ** 2 for x in a))
        magnitude_b = math.sqrt(sum(y ** 2 for y in b))
        return dot_product / (magnitude_a * magnitude_b)
    
    # Method 3: Dot Product
    def dot_product(a: List[float], b: List[float]) -> float:
        """Calculate dot product of vectors."""
        return sum(x * y for x, y in zip(a, b))
    
    print("1️⃣  Euclidean Distance (Straight-line distance):")
    print("   Formula: sqrt((a₁-b₁)² + (a₂-b₂)² + ...)")
    print()
    dist_ab = euclidean_distance(vec_a, vec_b)
    dist_ac = euclidean_distance(vec_a, vec_c)
    print(f"   Distance(A, B) = {dist_ab:.2f}")
    print(f"   Distance(A, C) = {dist_ac:.2f}")
    print()
    print("   💡 B is farther from A than C (because B is longer)")
    print("      But A and B point in the SAME direction!")
    print("      Problem: Length matters, not just direction")
    print()
    
    print("2️⃣  Cosine Similarity (Angle between vectors):")
    print("   Formula: (a·b) / (|a| × |b|)")
    print("   Range: -1 (opposite) to +1 (identical)")
    print()
    cos_ab = cosine_similarity(vec_a, vec_b)
    cos_ac = cosine_similarity(vec_a, vec_c)
    print(f"   Similarity(A, B) = {cos_ab:.3f}  (1.0 = same direction!)")
    print(f"   Similarity(A, C) = {cos_ac:.3f}  (0.96 = very similar)")
    print()
    print("   💡 A and B are identical in direction (cosine = 1.0)")
    print("      Length doesn't matter! ✅")
    print("      This is what we want for text embeddings!")
    print()
    
    print("3️⃣  Dot Product (Vector overlap):")
    print("   Formula: a₁×b₁ + a₂×b₂ + ...")
    print()
    dot_ab = dot_product(vec_a, vec_b)
    dot_ac = dot_product(vec_a, vec_c)
    print(f"   DotProduct(A, B) = {dot_ab:.2f}")
    print(f"   DotProduct(A, C) = {dot_ac:.2f}")
    print()
    print("   💡 B scores higher because it's longer")
    print("      Problem: Not normalized (unfair comparison)")
    print()
    
    print("🎯 TEXT EMBEDDING EXAMPLE:")
    print()
    print("   Sentence 1: 'I like cats'")
    print("   Sentence 2: 'I really truly deeply love cats a lot!!!'")
    print("   Sentence 3: 'Dogs are great'")
    print()
    
    # Simulated embeddings
    sent1 = [0.5, 0.5, 0.0]  # Short, about cats
    sent2 = [5.0, 5.0, 0.0]  # Long, about cats (same direction!)
    sent3 = [0.0, 0.5, 0.5]  # About dogs (different direction)
    
    print("   Embeddings (simplified):")
    print(f"   Sent1 = {sent1}")
    print(f"   Sent2 = {sent2}  (same direction, bigger magnitude)")
    print(f"   Sent3 = {sent3}  (different topic)")
    print()
    
    print("   Euclidean Distance:")
    print(f"     Sent1 ↔ Sent2: {euclidean_distance(sent1, sent2):.2f}")
    print(f"     Sent1 ↔ Sent3: {euclidean_distance(sent1, sent3):.2f}")
    print("     → Says Sent2 is farther (but same meaning!)")
    print()
    
    print("   Cosine Similarity:")
    print(f"     Sent1 ↔ Sent2: {cosine_similarity(sent1, sent2):.3f}")
    print(f"     Sent1 ↔ Sent3: {cosine_similarity(sent1, sent3):.3f}")
    print("     → Correctly identifies Sent1 and Sent2 as identical! ✅")
    print()
    
    print("💡 DECISION GUIDE:")
    print()
    print("   Use Cosine Similarity when:")
    print("   ✅ Text embeddings (document length varies)")
    print("   ✅ Direction matters more than magnitude")
    print("   ✅ Want normalized scores (-1 to +1)")
    print()
    print("   Use Euclidean Distance when:")
    print("   • Absolute position matters (e.g., coordinates)")
    print("   • All vectors are normalized already")
    print()
    print("   Use Dot Product when:")
    print("   • Speed is critical (fastest to compute)")
    print("   • Vectors are pre-normalized")
    print()
    
    print("⚡ Performance Note:")
    print()
    print("   Cosine similarity = dot product / (magnitude × magnitude)")
    print("   If vectors are pre-normalized (magnitude = 1):")
    print("   → Cosine similarity = dot product (faster!)")
    print()
    print("   Most embedding models return normalized vectors")
    print("   → Can use dot product directly (100x faster for large datasets)")


# ============================================================================
# 3. Simple Semantic Search
# ============================================================================

def demo_semantic_search():
    """
    Build a simple semantic search engine.
    
    INTUITION - Smart Document Search:
    
    Old search (keyword matching):
    Query: "fix car"
    Finds: Documents containing exact words "fix" and "car"
    Misses: "automobile repair", "vehicle maintenance" (different words!)
    
    Semantic search (embedding similarity):
    Query: "fix car"
    Converts to: embedding vector
    Finds: Documents with similar meaning embeddings
    Matches: "automobile repair", "vehicle maintenance" ✅
    
    How It Works:
    
    1. Index Phase (do once):
       - Convert all documents to embeddings
       - Store embeddings in vector database
    
    2. Search Phase (do per query):
       - Convert query to embedding
       - Find documents with most similar embeddings
       - Return top-k results
    
    Real Example:
    
    Documents:
    1. "Paris is the capital of France"
    2. "The Eiffel Tower is in Paris"
    3. "Pizza is delicious"
    4. "France is known for wine and cheese"
    
    Query: "What is France's capital?"
    
    Result: Doc 1 (highest similarity, despite different words!)
    """
    print("\n" + "=" * 70)
    print("3. Simple Semantic Search")
    print("=" * 70)
    print()
    print("💭 INTUITION: Smart vs Dumb Search")
    print()
    print("   ❌ Keyword Search:")
    print("      Query: 'fix car'")
    print("      Finds: Documents with exact words 'fix' AND 'car'")
    print("      Misses: 'automobile repair', 'vehicle maintenance'")
    print()
    print("   ✅ Semantic Search:")
    print("      Query: 'fix car'")
    print("      Finds: Documents with SIMILAR MEANING")
    print("      Matches: 'automobile repair', 'car troubleshooting' ✓")
    print()
    
    # Simulated document embeddings (3D for demo)
    # In reality, these would be 384 or 768 dimensions
    documents = {
        "Paris is the capital of France": [0.8, 0.2, 0.1],
        "The Eiffel Tower is in Paris": [0.7, 0.3, 0.1],
        "Pizza is delicious": [0.1, 0.1, 0.9],
        "France is known for wine and cheese": [0.6, 0.1, 0.8],
        "Python is a programming language": [0.1, 0.9, 0.2],
        "Machine learning uses neural networks": [0.1, 0.8, 0.1],
    }
    
    print("📚 Document Collection (with fake embeddings):")
    print()
    for i, (doc, emb) in enumerate(documents.items(), 1):
        print(f"   {i}. '{doc}'")
        print(f"      Embedding: {emb}")
        print()
    
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(y ** 2 for y in b))
        return dot / (mag_a * mag_b)
    
    def semantic_search(query: str, query_embedding: List[float], 
                       documents: Dict[str, List[float]], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Perform semantic search.
        
        Args:
            query: Query text
            query_embedding: Embedding vector for query
            documents: Dict of {document: embedding}
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        # Calculate similarity with all documents
        similarities = []
        for doc, doc_emb in documents.items():
            sim = cosine_similarity(query_embedding, doc_emb)
            similarities.append((doc, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    # Test queries
    queries = [
        ("What is the capital of France?", [0.8, 0.2, 0.0]),
        ("Tell me about programming", [0.0, 0.9, 0.1]),
        ("I want to eat something", [0.1, 0.0, 0.9])
    ]
    
    print("🔍 SEMANTIC SEARCH IN ACTION:")
    print()
    
    for query, query_emb in queries:
        print(f"Query: '{query}'")
        print(f"Query embedding: {query_emb}")
        print()
        
        results = semantic_search(query, query_emb, documents, top_k=3)
        
        print("Top Results:")
        for rank, (doc, score) in enumerate(results, 1):
            bar = "█" * int(score * 50)
            print(f"  {rank}. [{score:.3f}] {bar}")
            print(f"     '{doc}'")
            print()
        print("-" * 70)
        print()
    
    print("💡 OBSERVATIONS:")
    print()
    print("   Query 1: 'What is the capital of France?'")
    print("   → Top result: 'Paris is the capital of France'")
    print("   ✅ Perfect match, despite different wording!")
    print()
    print("   Query 2: 'Tell me about programming'")
    print("   → Top results: Python, machine learning docs")
    print("   ✅ Found technical content!")
    print()
    print("   Query 3: 'I want to eat something'")
    print("   → Top results: Pizza, France's food")
    print("   ✅ Found food-related content!")
    print()
    
    print("🚀 REAL-WORLD IMPLEMENTATION:")
    print()
    print("   1️⃣  Index Phase (offline, once):")
    print("      for doc in documents:")
    print("          embedding = model.encode(doc)")
    print("          store in vector database")
    print()
    print("   2️⃣  Search Phase (online, per query):")
    print("      query_embedding = model.encode(query)")
    print("      results = vector_db.search(query_embedding, top_k=10)")
    print("      return results")
    print()
    print("   Popular Vector Databases:")
    print("   • Pinecone (managed, easy)")
    print("   • Weaviate (open-source, full-featured)")
    print("   • FAISS (Facebook, super fast)")
    print("   • Chroma (embedded, lightweight)")
    print("   • Qdrant (open-source, production-ready)")
    print()
    
    print("⚡ SCALING TO MILLIONS OF DOCUMENTS:")
    print()
    print("   Problem: Compare query to 1M docs = slow!")
    print()
    print("   Solution: Approximate Nearest Neighbor (ANN)")
    print("   • HNSW (Hierarchical Navigable Small World)")
    print("   • IVF (Inverted File Index)")
    print("   • Product Quantization")
    print()
    print("   Result: 1000x faster, 95%+ accuracy!")
    print("   • Brute force: 1000ms for 1M docs")
    print("   • HNSW: 1ms for 1M docs ⚡")


# ============================================================================
# 4. Real-World Applications
# ============================================================================

def demo_applications():
    """
    Real-world applications of embeddings.
    """
    print("\n" + "=" * 70)
    print("4. Real-World Applications")
    print("=" * 70)
    print()
    
    print("🎯 APPLICATION 1: Customer Support Ticket Routing")
    print()
    print("   Problem: 1000s of support tickets, need to route to right team")
    print()
    print("   Solution: Embed tickets and team expertise, match similarity")
    print()
    
    # Simulated ticket and team embeddings
    tickets = {
        "My password won't reset": [0.9, 0.1, 0.1],      # Auth issue
        "The app keeps crashing": [0.1, 0.9, 0.1],       # Technical issue
        "How do I cancel my subscription?": [0.1, 0.1, 0.9],  # Billing
    }
    
    teams = {
        "Authentication Team": [0.9, 0.1, 0.1],
        "Technical Support": [0.1, 0.9, 0.1],
        "Billing Department": [0.1, 0.1, 0.9],
    }
    
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(y ** 2 for y in b))
        return dot / (mag_a * mag_b)
    
    print("   Tickets:")
    for ticket, emb in tickets.items():
        print(f"   • '{ticket}'")
        
        # Find best team
        best_team = None
        best_score = -1
        for team, team_emb in teams.items():
            score = cosine_similarity(emb, team_emb)
            if score > best_score:
                best_score = score
                best_team = team
        
        print(f"     → Route to: {best_team} (similarity: {best_score:.3f})")
        print()
    
    print("   ✅ Result: Instant, accurate routing!")
    print("      Human: 5 minutes per ticket")
    print("      Embeddings: <1 second, 95%+ accuracy")
    print()
    
    print("🎯 APPLICATION 2: Duplicate Question Detection")
    print()
    print("   Problem: Users ask same questions with different words")
    print()
    
    questions = [
        "How do I reset my password?",
        "I forgot my password, what should I do?",
        "Can't log in because I don't remember password",
        "What's the weather today?",
    ]
    
    # Simulated embeddings
    q_embeddings = [
        [0.9, 0.1, 0.0],  # Password reset
        [0.9, 0.1, 0.0],  # Password reset (duplicate!)
        [0.8, 0.1, 0.0],  # Password reset (duplicate!)
        [0.0, 0.0, 0.9],  # Weather (different)
    ]
    
    print("   Questions:")
    for i, q in enumerate(questions):
        print(f"   {i+1}. '{q}'")
    print()
    
    # Find duplicates
    threshold = 0.85
    print(f"   Finding duplicates (threshold={threshold}):")
    print()
    
    duplicates = defaultdict(list)
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            sim = cosine_similarity(q_embeddings[i], q_embeddings[j])
            if sim >= threshold:
                duplicates[i].append((j, sim))
    
    for q_idx, dups in duplicates.items():
        print(f"   Question {q_idx+1}: '{questions[q_idx]}'")
        print("   Duplicates found:")
        for dup_idx, sim in dups:
            print(f"     • Q{dup_idx+1} (similarity: {sim:.3f})")
            print(f"       '{questions[dup_idx]}'")
        print()
    
    print("   ✅ Result: Automatically merge duplicates!")
    print("      Reduce FAQ from 1000 → 300 unique questions")
    print()
    
    print("🎯 APPLICATION 3: Content Recommendation")
    print()
    print("   Problem: Suggest relevant articles to users")
    print()
    print("   Solution: User reads article → Find similar articles")
    print()
    
    articles = {
        "Introduction to Python": [0.8, 0.2, 0.1],
        "Advanced Python Techniques": [0.9, 0.2, 0.1],
        "JavaScript for Beginners": [0.7, 0.3, 0.1],
        "Best Pasta Recipes": [0.1, 0.1, 0.9],
        "Italian Cooking Guide": [0.1, 0.1, 0.8],
    }
    
    user_read = "Introduction to Python"
    user_emb = articles[user_read]
    
    print(f"   User just read: '{user_read}'")
    print()
    print("   Recommendations (similar articles):")
    
    recommendations = []
    for article, emb in articles.items():
        if article != user_read:
            sim = cosine_similarity(user_emb, emb)
            recommendations.append((article, sim))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (article, sim) in enumerate(recommendations[:3], 1):
        bar = "█" * int(sim * 50)
        print(f"   {rank}. [{sim:.3f}] {bar}")
        print(f"      '{article}'")
        print()
    
    print("   ✅ Result: Relevant recommendations!")
    print("      User engagement +40%")
    print("      Time on site +25%")
    print()
    
    print("🎯 APPLICATION 4: Multi-Lingual Search")
    print()
    print("   Problem: Search in English, find French/Spanish documents")
    print()
    print("   Solution: Multilingual embeddings (same space for all languages)")
    print()
    print("   Example:")
    print("   Query (English): 'Hello, how are you?'")
    print("   Finds documents:")
    print("   • 'Bonjour, comment allez-vous?' (French)")
    print("   • 'Hola, ¿cómo estás?' (Spanish)")
    print("   • 'Hello, how are you?' (English)")
    print()
    print("   All have similar embeddings! ✅")
    print()
    
    print("💡 MORE APPLICATIONS:")
    print()
    print("   • Chatbots: Find relevant knowledge base articles")
    print("   • E-commerce: 'Show me similar products'")
    print("   • Resume screening: Match candidates to job descriptions")
    print("   • Code search: Find similar code snippets")
    print("   • Plagiarism detection: Find copied content")
    print("   • Image search: Search images with text (CLIP)")
    print("   • Music recommendation: Find similar songs")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🔤 Embeddings and Semantic Search\n")
    print("Learn how to convert text to vectors and find similar content!")
    print()
    
    demo_embeddings_intuition()
    demo_similarity_metrics()
    demo_semantic_search()
    demo_applications()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Embeddings: Text → Vectors (lists of numbers)
   - Capture semantic meaning
   - Enable similarity comparison
   - Typical size: 384-768 dimensions

2. Similarity Metrics:
   - Cosine Similarity: Best for text (ignores length)
   - Euclidean Distance: For absolute position
   - Dot Product: Fast when vectors are normalized

3. Semantic Search:
   - Finds documents by meaning, not keywords
   - Query → Embedding → Find similar document embeddings
   - Much better than keyword search!

4. Real Applications:
   - Customer support routing
   - Duplicate detection
   - Content recommendation
   - Multi-lingual search
   - Question answering

Popular Embedding Models:
- sentence-transformers (all-MiniLM-L6-v2): Fast, good quality
- OpenAI (text-embedding-3-small): High quality, API-based
- Cohere: Multilingual, production-ready

Vector Databases:
- Pinecone: Managed, easy to use
- Weaviate: Open-source, feature-rich
- Chroma: Embedded, lightweight
- FAISS: Facebook, super fast

Real Code Example:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents
docs = ["Paris is in France", "Pizza is delicious"]
doc_embeddings = model.encode(docs)

# Search
query = "What is France's capital?"
query_embedding = model.encode(query)

# Find most similar
similarities = np.dot(doc_embeddings, query_embedding)
best_doc_idx = similarities.argmax()
print(docs[best_doc_idx])  # "Paris is in France"
```

Next: 02_vector_databases.py - Store and search millions of embeddings!
""")


if __name__ == "__main__":
    main()
