"""
Pandora AIOS Knowledge Base & Documentation RAG System
--------------------------------------------------------
Retrieval-Augmented Generation (RAG) system for Pandora's literature.
Indexes and queries all existing documentation, code, ethics, and research.

Features:
- Document indexing and embedding
- Semantic search over Pandora literature
- Context-aware retrieval
- Citation tracking
- Multi-document synthesis

Philosophy: Knowledge preservation, accessible wisdom, standing on documented shoulders
"""

import os
import re
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

@dataclass
class Document:
    """Represents a document in the knowledge base"""
    doc_id: str
    title: str
    content: str
    doc_type: str  # code, ethics, readme, script, config
    file_path: str
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = None  # For semantic search
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'content': self.content,
            'doc_type': self.doc_type,
            'file_path': self.file_path,
            'categories': self.categories,
            'keywords': self.keywords,
            'citations': self.citations,
            'metadata': self.metadata
        }


class DocumentIndexer:
    """Indexes Pandora documentation and code"""
    
    # Document type patterns
    DOC_PATTERNS = {
        'ethics': ['ETHICS', 'CORE_PRINCIPLES', 'SUMMARY', 'ethics/', 'moral'],
        'readme': ['README', 'IMPLEMENTATION', 'WSL_TERMINAL'],
        'code': ['.py$'],
        'script': ['.sh$', '.bat$', '.ps1$'],
        'config': ['.json$', '.desktop$', '.service$'],
        'scientific': ['SCIENTIFIC', 'research', 'quantum', 'overlay']
    }
    
    # Key concepts to extract
    KEY_CONCEPTS = [
        # Ethics & Philosophy
        'bhagavad gita', 'bible', 'dead sea scrolls', 'stoicism', 'socrates',
        'aristotle', 'kant', 'confucius', 'taoism', 'tesla', 'einstein', 'hawking',
        'ethics', 'morals', 'virtue', 'compassion', 'truth', 'justice',
        
        # Technical Concepts
        'quantum', 'qubit', 'entanglement', 'superposition', 'overlay',
        'wormhole', 'hive', 'castle', 'empire', 'omega',
        'diagnostic', 'compatibility', 'wsl', 'terminal', 'boot manager',
        
        # Scientific Concepts
        'relativity', 'black hole', 'singularity', 'gravitational wave',
        'solar', 'cern', 'mit', 'particle physics', 'string theory',
        
        # System Concepts
        'architecture', 'security', 'firewall', 'antivirus', 'safe mode',
        'fabric', 'orchestrator', 'self-learning', 'adaptation'
    ]
    
    def __init__(self, repo_path: str = None):
        if repo_path is None:
            repo_path = os.getcwd()
        self.repo_path = Path(repo_path)
        self.documents: List[Document] = []
    
    def index_repository(self):
        """Index all documents in repository"""
        print("[INFO] Indexing Pandora repository...")
        
        # Find all relevant files
        files_to_index = []
        
        for pattern in ['**/*.md', '**/*.py', '**/*.sh', '**/*.ps1', '**/*.bat', 
                       '**/*.json', '**/*.txt', '**/*.desktop', '**/*.service']:
            for file_path in self.repo_path.glob(pattern):
                # Skip hidden and git files
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                files_to_index.append(file_path)
        
        print(f"[INFO] Found {len(files_to_index)} files to index")
        
        # Index each file
        for file_path in files_to_index:
            try:
                doc = self._index_file(file_path)
                if doc:
                    self.documents.append(doc)
            except Exception as e:
                print(f"[WARNING] Failed to index {file_path}: {e}")
        
        print(f"[SUCCESS] Indexed {len(self.documents)} documents")
        
        return self.documents
    
    def _index_file(self, file_path: Path) -> Optional[Document]:
        """Index a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return None
        
        if not content.strip():
            return None
        
        # Generate document ID
        doc_id = hashlib.md5(str(file_path).encode()).hexdigest()[:12]
        
        # Determine document type
        doc_type = self._classify_document(file_path)
        
        # Extract title
        title = self._extract_title(file_path, content)
        
        # Extract categories and keywords
        categories = self._extract_categories(file_path, content)
        keywords = self._extract_keywords(content)
        
        # Extract citations (references to other docs/concepts)
        citations = self._extract_citations(content)
        
        # Build document
        doc = Document(
            doc_id=doc_id,
            title=title,
            content=content,
            doc_type=doc_type,
            file_path=str(file_path.relative_to(self.repo_path)),
            categories=categories,
            keywords=keywords,
            citations=citations,
            metadata={
                'size': len(content),
                'lines': content.count('\n') + 1,
                'extension': file_path.suffix
            }
        )
        
        return doc
    
    def _classify_document(self, file_path: Path) -> str:
        """Classify document type"""
        path_str = str(file_path)
        
        for doc_type, patterns in self.DOC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, path_str, re.IGNORECASE):
                    return doc_type
        
        return 'other'
    
    def _extract_title(self, file_path: Path, content: str) -> str:
        """Extract document title"""
        # Try to find title in content
        lines = content.split('\n')
        
        # Check for markdown headers
        for line in lines[:20]:
            if line.startswith('#'):
                return line.lstrip('#').strip()
        
        # Check for docstring title
        if '"""' in content[:200]:
            match = re.search(r'"""\s*(.+?)\s*\n', content)
            if match:
                return match.group(1)
        
        # Use filename
        return file_path.stem.replace('_', ' ').title()
    
    def _extract_categories(self, file_path: Path, content: str) -> List[str]:
        """Extract categories from document"""
        categories = []
        path_str = str(file_path).lower()
        content_lower = content.lower()
        
        if 'ethics' in path_str or 'ethics' in content_lower[:500]:
            categories.append('ethics')
        
        if 'quantum' in path_str or 'quantum' in content_lower[:500]:
            categories.append('quantum')
        
        if 'security' in path_str or 'firewall' in content_lower[:500]:
            categories.append('security')
        
        if 'diagnostic' in path_str or 'diagnostic' in content_lower[:500]:
            categories.append('diagnostic')
        
        if 'wsl' in path_str or 'windows' in content_lower[:500]:
            categories.append('wsl')
        
        if 'scientific' in path_str or 'research' in content_lower[:500]:
            categories.append('scientific')
        
        if 'boot' in path_str or 'multi-os' in content_lower[:500]:
            categories.append('boot')
        
        if not categories:
            categories.append('general')
        
        return categories
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        content_lower = content.lower()
        found_keywords = []
        
        for concept in self.KEY_CONCEPTS:
            if concept in content_lower:
                found_keywords.append(concept)
        
        return found_keywords[:20]  # Limit to top 20
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations and references"""
        citations = []
        
        # Find quoted names/texts
        quotes = re.findall(r'"([^"]+)"', content)
        for quote in quotes[:10]:
            if len(quote) > 20 and len(quote) < 200:
                citations.append(quote[:100])
        
        # Find references to other files
        file_refs = re.findall(r'`([a-z_]+\.(?:py|md|sh))`', content, re.IGNORECASE)
        citations.extend(file_refs[:10])
        
        return citations


class SimpleVectorSearch:
    """Simple TF-IDF based vector search (no external dependencies)"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
    
    def build_index(self, documents: List[Document]):
        """Build search index"""
        self.documents = documents
        print(f"[INFO] Building search index for {len(documents)} documents...")
        
        # Build vocabulary
        for doc in documents:
            words = self._tokenize(doc.content + " " + doc.title)
            for word in set(words):
                self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
        
        # Calculate IDF
        num_docs = len(documents)
        for word, doc_freq in self.vocabulary.items():
            self.idf[word] = math.log(num_docs / (doc_freq + 1))
        
        # Build document vectors
        for doc in documents:
            self.doc_vectors[doc.doc_id] = self._vectorize(doc.content + " " + doc.title)
        
        print(f"[SUCCESS] Index built with {len(self.vocabulary)} terms")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                     'can', 'has', 'was', 'one', 'our', 'out', 'this', 'that',
                     'with', 'from', 'have', 'will', 'been', 'more', 'when'}
        words = [w for w in words if w not in stop_words]
        
        return words
    
    def _vectorize(self, text: str) -> Dict[str, float]:
        """Convert text to TF-IDF vector"""
        words = self._tokenize(text)
        vector = {}
        
        # Calculate term frequency
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate TF-IDF
        total_words = len(words)
        for word, count in word_counts.items():
            if word in self.idf:
                tf = count / total_words
                vector[word] = tf * self.idf[word]
        
        # Normalize
        magnitude = math.sqrt(sum(v**2 for v in vector.values()))
        if magnitude > 0:
            vector = {k: v/magnitude for k, v in vector.items()}
        
        return vector
    
    def search(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        query_vector = self._vectorize(query)
        
        # Calculate cosine similarity
        scores = []
        for doc in self.documents:
            # Apply filters
            if filters:
                if 'doc_type' in filters and doc.doc_type not in filters['doc_type']:
                    continue
                if 'categories' in filters:
                    if not any(cat in doc.categories for cat in filters['categories']):
                        continue
            
            doc_vector = self.doc_vectors[doc.doc_id]
            
            # Cosine similarity
            score = sum(query_vector.get(word, 0) * doc_vector.get(word, 0) 
                       for word in set(query_vector.keys()) | set(doc_vector.keys()))
            
            # Boost by keyword matches
            for keyword in doc.keywords:
                if keyword in query.lower():
                    score += 0.1
            
            scores.append((doc, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class PandoraKnowledgeBase:
    """Main knowledge base system"""
    
    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or os.getcwd()
        
        # Initialize components
        self.indexer = DocumentIndexer(self.repo_path)
        self.search = SimpleVectorSearch()
        
        # Storage
        self.db_path = os.path.expanduser("~/.pandora/knowledge_base.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._initialize_database()
        
        # Statistics
        self.stats = {
            'total_docs': 0,
            'by_type': defaultdict(int),
            'by_category': defaultdict(int)
        }
    
    def _initialize_database(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                doc_type TEXT,
                file_path TEXT,
                categories TEXT,
                keywords TEXT,
                citations TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                timestamp TEXT,
                results TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords ON documents(keywords)')
        
        self.conn.commit()
    
    def build_knowledge_base(self):
        """Build complete knowledge base"""
        print("\n" + "="*70)
        print("Building Pandora AIOS Knowledge Base")
        print("="*70)
        
        # Index repository
        documents = self.indexer.index_repository()
        
        # Build search index
        self.search.build_index(documents)
        
        # Save to database
        self._save_documents(documents)
        
        # Calculate statistics
        self._calculate_statistics(documents)
        
        print("\n" + "="*70)
        print("Knowledge Base Statistics")
        print("="*70)
        print(f"Total Documents: {self.stats['total_docs']}")
        print(f"\nBy Type:")
        for doc_type, count in sorted(self.stats['by_type'].items()):
            print(f"  {doc_type}: {count}")
        print(f"\nBy Category:")
        for category, count in sorted(self.stats['by_category'].items()):
            print(f"  {category}: {count}")
        print("="*70)
    
    def _save_documents(self, documents: List[Document]):
        """Save documents to database"""
        cursor = self.conn.cursor()
        
        for doc in documents:
            cursor.execute('''
                INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc.doc_id,
                doc.title,
                doc.content,
                doc.doc_type,
                doc.file_path,
                json.dumps(doc.categories),
                json.dumps(doc.keywords),
                json.dumps(doc.citations),
                json.dumps(doc.metadata)
            ))
        
        self.conn.commit()
    
    def _calculate_statistics(self, documents: List[Document]):
        """Calculate knowledge base statistics"""
        self.stats['total_docs'] = len(documents)
        
        for doc in documents:
            self.stats['by_type'][doc.doc_type] += 1
            for category in doc.categories:
                self.stats['by_category'][category] += 1
    
    def query(self, question: str, context: str = None, top_k: int = 3) -> Dict[str, Any]:
        """Query knowledge base"""
        # Search for relevant documents
        results = self.search.search(question, top_k=top_k)
        
        # Build response
        response = {
            'question': question,
            'sources': [],
            'context': []
        }
        
        for doc, score in results:
            if score > 0.01:  # Minimum relevance threshold
                response['sources'].append({
                    'title': doc.title,
                    'type': doc.doc_type,
                    'file': doc.file_path,
                    'score': round(score, 3),
                    'keywords': doc.keywords[:5]
                })
                
                # Extract relevant excerpt
                excerpt = self._extract_relevant_excerpt(doc.content, question)
                response['context'].append({
                    'title': doc.title,
                    'excerpt': excerpt
                })
        
        # Save query
        self._save_query(question, response)
        
        return response
    
    def _extract_relevant_excerpt(self, content: str, query: str, context_size: int = 300) -> str:
        """Extract most relevant excerpt from document"""
        query_words = set(self.search._tokenize(query))
        
        # Split into sentences/paragraphs
        paragraphs = content.split('\n\n')
        
        # Score each paragraph
        best_score = 0
        best_paragraph = ""
        
        for para in paragraphs:
            if len(para) < 50:
                continue
            
            para_words = set(self.search._tokenize(para))
            score = len(query_words & para_words)
            
            if score > best_score:
                best_score = score
                best_paragraph = para
        
        # Truncate if too long
        if len(best_paragraph) > context_size:
            best_paragraph = best_paragraph[:context_size] + "..."
        
        return best_paragraph or content[:context_size] + "..."
    
    def _save_query(self, query_text: str, results: Dict):
        """Save query for analytics"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO queries (query_text, timestamp, results)
            VALUES (?, ?, ?)
        ''', (
            query_text,
            datetime.now().isoformat(),
            json.dumps(results)
        ))
        
        self.conn.commit()
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT * FROM documents WHERE doc_id = ?', (doc_id,))
        row = cursor.fetchone()
        
        if row:
            return Document(
                doc_id=row[0],
                title=row[1],
                content=row[2],
                doc_type=row[3],
                file_path=row[4],
                categories=json.loads(row[5]),
                keywords=json.loads(row[6]),
                citations=json.loads(row[7]),
                metadata=json.loads(row[8])
            )
        
        return None
    
    def get_related_documents(self, doc_id: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Find related documents"""
        doc = self.get_document_by_id(doc_id)
        if not doc:
            return []
        
        # Use document content as query
        query = doc.title + " " + " ".join(doc.keywords)
        results = self.search.search(query, top_k=top_k + 1)
        
        # Filter out self
        results = [(d, s) for d, s in results if d.doc_id != doc_id]
        
        return results[:top_k]
    
    def get_documentation_summary(self) -> str:
        """Generate summary of available documentation"""
        summary = []
        summary.append("="*70)
        summary.append("PANDORA AIOS KNOWLEDGE BASE SUMMARY")
        summary.append("="*70)
        summary.append(f"\nTotal Documents: {self.stats['total_docs']}")
        summary.append("\nAvailable Documentation:")
        
        # Group by type
        for doc_type, count in sorted(self.stats['by_type'].items()):
            summary.append(f"\n{doc_type.upper()} ({count} documents):")
            
            # Get sample documents
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT title, file_path FROM documents 
                WHERE doc_type = ? LIMIT 5
            ''', (doc_type,))
            
            for row in cursor.fetchall():
                summary.append(f"  - {row[0]} ({row[1]})")
        
        summary.append("\n" + "="*70)
        summary.append("Use query() to search this knowledge base")
        summary.append("="*70)
        
        return "\n".join(summary)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pandora Knowledge Base Manager")
    parser.add_argument("--build", action="store_true", help="Build/rebuild knowledge base")
    parser.add_argument("--query", type=str, help="Query the knowledge base")
    parser.add_argument("--summary", action="store_true", help="Show knowledge base summary")
    parser.add_argument("--repo-path", type=str, help="Repository path")
    
    args = parser.parse_args()
    
    kb = PandoraKnowledgeBase(repo_path=args.repo_path)
    
    if args.build:
        kb.build_knowledge_base()
    
    elif args.query:
        results = kb.query(args.query)
        
        print("\n" + "="*70)
        print(f"Query: {results['question']}")
        print("="*70)
        
        print(f"\nFound {len(results['sources'])} relevant sources:")
        for i, source in enumerate(results['sources'], 1):
            print(f"\n{i}. {source['title']} ({source['type']})")
            print(f"   File: {source['file']}")
            print(f"   Score: {source['score']}")
            print(f"   Keywords: {', '.join(source['keywords'])}")
        
        print("\n" + "="*70)
        print("Context Excerpts:")
        print("="*70)
        for i, ctx in enumerate(results['context'], 1):
            print(f"\n[{i}] {ctx['title']}:")
            print(ctx['excerpt'])
        
        print("\n" + "="*70)
    
    elif args.summary:
        print(kb.get_documentation_summary())
    
    else:
        print("Use --build to build knowledge base, --query to search, or --summary for info")


if __name__ == "__main__":
    from datetime import datetime
    main()
