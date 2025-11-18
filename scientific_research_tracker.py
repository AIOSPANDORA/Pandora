"""
Pandora AIOS Scientific Research Tracker & Literature Database
----------------------------------------------------------------
Tracks breakthroughs and maintains comprehensive literature database for:
- Relativistic Astrophysics (Einstein, Hawking legacy)
- Singularities and Black Holes
- Solar Physics and Stellar Research
- Quantum Physics and Computing
- Qubit Coding and Quantum Information Theory
- All Relevant Scientific Fields

Data Sources: arXiv, MIT, CERN, NASA, ESA, Major Universities
Philosophy: Standing on the shoulders of giants - Einstein, Hawking, Tesla
"""

import os
import json
import time
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

class ResearchCategory(Enum):
    """Categories of scientific research"""
    # Astrophysics & Cosmology
    BLACK_HOLES = "black_holes"
    SINGULARITIES = "singularities"
    RELATIVITY = "relativity"
    COSMOLOGY = "cosmology"
    SOLAR_PHYSICS = "solar_physics"
    GRAVITATIONAL_WAVES = "gravitational_waves"
    
    # Quantum Sciences
    QUANTUM_COMPUTING = "quantum_computing"
    QUANTUM_INFORMATION = "quantum_information"
    QUBIT_ENGINEERING = "qubit_engineering"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    QUANTUM_CRYPTOGRAPHY = "quantum_cryptography"
    QUANTUM_ALGORITHMS = "quantum_algorithms"
    
    # Particle Physics
    PARTICLE_PHYSICS = "particle_physics"
    HIGGS_BOSON = "higgs_boson"
    CERN_EXPERIMENTS = "cern_experiments"
    STANDARD_MODEL = "standard_model"
    
    # Theoretical Physics
    STRING_THEORY = "string_theory"
    LOOP_QUANTUM_GRAVITY = "loop_quantum_gravity"
    QUANTUM_FIELD_THEORY = "quantum_field_theory"
    UNIFIED_THEORIES = "unified_theories"
    
    # Applied Sciences
    QUANTUM_MATERIALS = "quantum_materials"
    SUPERCONDUCTIVITY = "superconductivity"
    TOPOLOGICAL_STATES = "topological_states"
    QUANTUM_SENSORS = "quantum_sensors"

class Institution(Enum):
    """Major research institutions"""
    MIT = "mit"
    CERN = "cern"
    NASA = "nasa"
    ESA = "esa"
    CALTECH = "caltech"
    PRINCETON = "princeton"
    CAMBRIDGE = "cambridge"
    OXFORD = "oxford"
    STANFORD = "stanford"
    HARVARD = "harvard"
    ETH_ZURICH = "eth_zurich"
    BERKELEY = "berkeley"
    TOKYO = "tokyo"
    MAX_PLANCK = "max_planck"
    ARXIV = "arxiv"

@dataclass
class ScientificPaper:
    """Represents a scientific paper"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[ResearchCategory]
    institutions: List[Institution]
    publication_date: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    citations: int = 0
    keywords: List[str] = field(default_factory=list)
    related_papers: List[str] = field(default_factory=list)
    breakthrough_score: float = 0.0  # 0-1 scale
    added_date: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['categories'] = [c.value for c in self.categories]
        data['institutions'] = [i.value for i in self.institutions]
        return data
    
    @staticmethod
    def from_dict(data: Dict) -> 'ScientificPaper':
        """Create from dictionary"""
        data['categories'] = [ResearchCategory(c) for c in data.get('categories', [])]
        data['institutions'] = [Institution(i) for i in data.get('institutions', [])]
        return ScientificPaper(**data)

@dataclass
class Breakthrough:
    """Represents a scientific breakthrough"""
    breakthrough_id: str
    title: str
    description: str
    date: str
    category: ResearchCategory
    institution: Institution
    key_researchers: List[str]
    related_papers: List[str]
    impact_score: float  # 0-10 scale
    media_coverage: List[str] = field(default_factory=list)
    verification_status: str = "pending"  # pending, verified, disputed
    notes: str = ""

class ResearchDatabase:
    """SQLite database for research papers and breakthroughs"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.expanduser("~/.pandora/research_database.db")
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Papers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                categories TEXT,
                institutions TEXT,
                publication_date TEXT,
                arxiv_id TEXT,
                doi TEXT,
                pdf_url TEXT,
                citations INTEGER DEFAULT 0,
                keywords TEXT,
                related_papers TEXT,
                breakthrough_score REAL DEFAULT 0.0,
                added_date TEXT,
                notes TEXT
            )
        ''')
        
        # Breakthroughs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS breakthroughs (
                breakthrough_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                date TEXT,
                category TEXT,
                institution TEXT,
                key_researchers TEXT,
                related_papers TEXT,
                impact_score REAL,
                media_coverage TEXT,
                verification_status TEXT,
                notes TEXT
            )
        ''')
        
        # Research sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT,
                institution TEXT,
                category TEXT,
                last_checked TEXT,
                papers_found INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(publication_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(categories)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_breakthroughs_date ON breakthroughs(date)')
        
        self.conn.commit()
    
    def add_paper(self, paper: ScientificPaper):
        """Add paper to database"""
        cursor = self.conn.cursor()
        
        data = paper.to_dict()
        cursor.execute('''
            INSERT OR REPLACE INTO papers VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            data['paper_id'],
            data['title'],
            json.dumps(data['authors']),
            data['abstract'],
            json.dumps(data['categories']),
            json.dumps(data['institutions']),
            data['publication_date'],
            data.get('arxiv_id'),
            data.get('doi'),
            data.get('pdf_url'),
            data['citations'],
            json.dumps(data['keywords']),
            json.dumps(data['related_papers']),
            data['breakthrough_score'],
            data['added_date'],
            data['notes']
        ))
        
        self.conn.commit()
    
    def add_breakthrough(self, breakthrough: Breakthrough):
        """Add breakthrough to database"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO breakthroughs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            breakthrough.breakthrough_id,
            breakthrough.title,
            breakthrough.description,
            breakthrough.date,
            breakthrough.category.value,
            breakthrough.institution.value,
            json.dumps(breakthrough.key_researchers),
            json.dumps(breakthrough.related_papers),
            breakthrough.impact_score,
            json.dumps(breakthrough.media_coverage),
            breakthrough.verification_status,
            breakthrough.notes
        ))
        
        self.conn.commit()
    
    def search_papers(self, 
                     categories: List[ResearchCategory] = None,
                     institutions: List[Institution] = None,
                     keywords: List[str] = None,
                     min_breakthrough_score: float = 0.0,
                     limit: int = 100) -> List[ScientificPaper]:
        """Search papers with filters"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM papers WHERE breakthrough_score >= ?"
        params = [min_breakthrough_score]
        
        if categories:
            cat_filter = ' OR '.join(['categories LIKE ?' for _ in categories])
            query += f" AND ({cat_filter})"
            params.extend([f'%{cat.value}%' for cat in categories])
        
        if keywords:
            kw_filter = ' OR '.join(['keywords LIKE ? OR title LIKE ? OR abstract LIKE ?' for _ in keywords])
            query += f" AND ({kw_filter})"
            for kw in keywords:
                params.extend([f'%{kw}%', f'%{kw}%', f'%{kw}%'])
        
        query += " ORDER BY publication_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        papers = []
        for row in cursor.fetchall():
            paper_dict = {
                'paper_id': row[0],
                'title': row[1],
                'authors': json.loads(row[2]) if row[2] else [],
                'abstract': row[3],
                'categories': json.loads(row[4]) if row[4] else [],
                'institutions': json.loads(row[5]) if row[5] else [],
                'publication_date': row[6],
                'arxiv_id': row[7],
                'doi': row[8],
                'pdf_url': row[9],
                'citations': row[10],
                'keywords': json.loads(row[11]) if row[11] else [],
                'related_papers': json.loads(row[12]) if row[12] else [],
                'breakthrough_score': row[13],
                'added_date': row[14],
                'notes': row[15]
            }
            papers.append(ScientificPaper.from_dict(paper_dict))
        
        return papers
    
    def get_recent_breakthroughs(self, days: int = 30, limit: int = 50) -> List[Breakthrough]:
        """Get recent breakthroughs"""
        cursor = self.conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM breakthroughs 
            WHERE date >= ? 
            ORDER BY impact_score DESC, date DESC 
            LIMIT ?
        ''', (cutoff_date, limit))
        
        breakthroughs = []
        for row in cursor.fetchall():
            breakthrough = Breakthrough(
                breakthrough_id=row[0],
                title=row[1],
                description=row[2],
                date=row[3],
                category=ResearchCategory(row[4]),
                institution=Institution(row[5]),
                key_researchers=json.loads(row[6]) if row[6] else [],
                related_papers=json.loads(row[7]) if row[7] else [],
                impact_score=row[8],
                media_coverage=json.loads(row[9]) if row[9] else [],
                verification_status=row[10],
                notes=row[11]
            )
            breakthroughs.append(breakthrough)
        
        return breakthroughs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM papers")
        total_papers = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM breakthroughs")
        total_breakthroughs = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(breakthrough_score) FROM papers")
        avg_breakthrough_score = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT COUNT(*) FROM papers WHERE breakthrough_score >= 0.7")
        high_impact_papers = cursor.fetchone()[0]
        
        return {
            'total_papers': total_papers,
            'total_breakthroughs': total_breakthroughs,
            'avg_breakthrough_score': avg_breakthrough_score,
            'high_impact_papers': high_impact_papers
        }


class ScientificSourceManager:
    """Manages scientific literature sources"""
    
    def __init__(self):
        self.sources = self._initialize_sources()
    
    def _initialize_sources(self) -> Dict[str, Dict]:
        """Initialize major scientific literature sources"""
        return {
            # arXiv Categories
            'arxiv_gr_qc': {
                'name': 'arXiv General Relativity and Quantum Cosmology',
                'url': 'https://arxiv.org/list/gr-qc/recent',
                'institution': Institution.ARXIV,
                'categories': [ResearchCategory.RELATIVITY, ResearchCategory.BLACK_HOLES, 
                             ResearchCategory.COSMOLOGY, ResearchCategory.GRAVITATIONAL_WAVES]
            },
            'arxiv_quant_ph': {
                'name': 'arXiv Quantum Physics',
                'url': 'https://arxiv.org/list/quant-ph/recent',
                'institution': Institution.ARXIV,
                'categories': [ResearchCategory.QUANTUM_COMPUTING, ResearchCategory.QUANTUM_INFORMATION,
                             ResearchCategory.QUANTUM_ENTANGLEMENT, ResearchCategory.QUANTUM_ALGORITHMS]
            },
            'arxiv_hep_th': {
                'name': 'arXiv High Energy Physics - Theory',
                'url': 'https://arxiv.org/list/hep-th/recent',
                'institution': Institution.ARXIV,
                'categories': [ResearchCategory.STRING_THEORY, ResearchCategory.QUANTUM_FIELD_THEORY,
                             ResearchCategory.UNIFIED_THEORIES]
            },
            'arxiv_astro_ph': {
                'name': 'arXiv Astrophysics',
                'url': 'https://arxiv.org/list/astro-ph/recent',
                'institution': Institution.ARXIV,
                'categories': [ResearchCategory.SOLAR_PHYSICS, ResearchCategory.COSMOLOGY,
                             ResearchCategory.BLACK_HOLES]
            },
            
            # MIT Sources
            'mit_quantum': {
                'name': 'MIT Center for Quantum Engineering',
                'url': 'https://cqe.mit.edu/research/',
                'institution': Institution.MIT,
                'categories': [ResearchCategory.QUANTUM_COMPUTING, ResearchCategory.QUBIT_ENGINEERING,
                             ResearchCategory.QUANTUM_SENSORS]
            },
            'mit_ligo': {
                'name': 'MIT LIGO Laboratory',
                'url': 'https://space.mit.edu/LIGO/',
                'institution': Institution.MIT,
                'categories': [ResearchCategory.GRAVITATIONAL_WAVES, ResearchCategory.BLACK_HOLES,
                             ResearchCategory.RELATIVITY]
            },
            
            # CERN Sources
            'cern_main': {
                'name': 'CERN Document Server',
                'url': 'https://cds.cern.ch/',
                'institution': Institution.CERN,
                'categories': [ResearchCategory.PARTICLE_PHYSICS, ResearchCategory.HIGGS_BOSON,
                             ResearchCategory.CERN_EXPERIMENTS, ResearchCategory.STANDARD_MODEL]
            },
            'cern_lhc': {
                'name': 'CERN Large Hadron Collider',
                'url': 'https://home.cern/science/accelerators/large-hadron-collider',
                'institution': Institution.CERN,
                'categories': [ResearchCategory.PARTICLE_PHYSICS, ResearchCategory.HIGGS_BOSON]
            },
            
            # NASA Sources
            'nasa_solar': {
                'name': 'NASA Solar System Exploration',
                'url': 'https://solarsystem.nasa.gov/solar-system/sun/',
                'institution': Institution.NASA,
                'categories': [ResearchCategory.SOLAR_PHYSICS]
            },
            'nasa_black_holes': {
                'name': 'NASA Black Holes',
                'url': 'https://science.nasa.gov/astrophysics/focus-areas/black-holes/',
                'institution': Institution.NASA,
                'categories': [ResearchCategory.BLACK_HOLES, ResearchCategory.SINGULARITIES,
                             ResearchCategory.RELATIVITY]
            },
            
            # Other Major Institutions
            'caltech_theoretical': {
                'name': 'Caltech Theoretical Astrophysics',
                'url': 'https://www.tapir.caltech.edu/',
                'institution': Institution.CALTECH,
                'categories': [ResearchCategory.BLACK_HOLES, ResearchCategory.COSMOLOGY,
                             ResearchCategory.RELATIVITY]
            },
            'princeton_quantum': {
                'name': 'Princeton Quantum Initiative',
                'url': 'https://quantum.princeton.edu/',
                'institution': Institution.PRINCETON,
                'categories': [ResearchCategory.QUANTUM_COMPUTING, ResearchCategory.QUANTUM_MATERIALS]
            },
            'cambridge_damtp': {
                'name': 'Cambridge DAMTP (Hawking Institute)',
                'url': 'https://www.damtp.cam.ac.uk/',
                'institution': Institution.CAMBRIDGE,
                'categories': [ResearchCategory.BLACK_HOLES, ResearchCategory.COSMOLOGY,
                             ResearchCategory.RELATIVITY, ResearchCategory.STRING_THEORY]
            },
        }
    
    def get_source_urls(self, categories: List[ResearchCategory] = None,
                       institutions: List[Institution] = None) -> List[Dict]:
        """Get filtered list of source URLs"""
        filtered_sources = []
        
        for source_id, source_info in self.sources.items():
            # Filter by category
            if categories:
                if not any(cat in source_info['categories'] for cat in categories):
                    continue
            
            # Filter by institution
            if institutions:
                if source_info['institution'] not in institutions:
                    continue
            
            filtered_sources.append({
                'id': source_id,
                'name': source_info['name'],
                'url': source_info['url'],
                'institution': source_info['institution'].value,
                'categories': [cat.value for cat in source_info['categories']]
            })
        
        return filtered_sources


class BreakthroughDetector:
    """Detects potential breakthroughs in scientific literature"""
    
    BREAKTHROUGH_KEYWORDS = {
        # Black Holes & Singularities
        ResearchCategory.BLACK_HOLES: [
            'event horizon', 'schwarzschild', 'kerr', 'hawking radiation',
            'black hole merger', 'gravitational collapse', 'penrose', 
            'information paradox', 'holographic principle'
        ],
        ResearchCategory.SINGULARITIES: [
            'singularity', 'naked singularity', 'cosmic censorship',
            'ring singularity', 'spacelike singularity', 'timelike singularity'
        ],
        
        # Relativity
        ResearchCategory.RELATIVITY: [
            'general relativity', 'special relativity', 'einstein field equations',
            'spacetime curvature', 'geodesic', 'equivalence principle',
            'lorentz transformation', 'time dilation', 'length contraction'
        ],
        ResearchCategory.GRAVITATIONAL_WAVES: [
            'gravitational wave', 'ligo', 'virgo', 'lisa', 'gw detection',
            'binary merger', 'chirp mass', 'strain sensitivity'
        ],
        
        # Solar Physics
        ResearchCategory.SOLAR_PHYSICS: [
            'solar flare', 'coronal mass ejection', 'solar wind', 'sunspot',
            'solar corona', 'heliosphere', 'solar cycle', 'magnetic reconnection',
            'parker solar probe', 'solar dynamics observatory'
        ],
        
        # Quantum Computing
        ResearchCategory.QUANTUM_COMPUTING: [
            'quantum computer', 'quantum supremacy', 'quantum advantage',
            'quantum error correction', 'logical qubit', 'quantum gate',
            'quantum circuit', 'variational quantum', 'qaoa', 'vqe'
        ],
        ResearchCategory.QUBIT_ENGINEERING: [
            'superconducting qubit', 'transmon', 'ion trap', 'quantum dot',
            'topological qubit', 'photonic qubit', 'neutral atom', 'rydberg',
            'qubit coherence', 'qubit readout', 'qubit fidelity'
        ],
        ResearchCategory.QUANTUM_ENTANGLEMENT: [
            'entanglement', 'bell state', 'epr', 'quantum correlation',
            'entanglement swapping', 'entanglement distillation', 'monogamy'
        ],
        ResearchCategory.QUANTUM_ALGORITHMS: [
            'shor algorithm', 'grover algorithm', 'quantum fourier transform',
            'quantum simulation', 'quantum chemistry', 'quantum optimization',
            'quantum machine learning', 'quantum sampling'
        ],
        
        # Particle Physics
        ResearchCategory.HIGGS_BOSON: [
            'higgs', 'higgs boson', 'higgs field', 'electroweak symmetry',
            'mass generation', 'higgs decay', 'higgs coupling'
        ],
        ResearchCategory.CERN_EXPERIMENTS: [
            'lhc', 'atlas', 'cms', 'alice', 'lhcb', 'cern', 'proton collision',
            'particle detector', 'luminosity', 'cross section'
        ],
    }
    
    HIGH_IMPACT_TERMS = [
        'breakthrough', 'first observation', 'first detection', 'first measurement',
        'discovery', 'novel', 'unprecedented', 'paradigm shift', 'revolutionary',
        'experimental confirmation', 'theoretical prediction confirmed'
    ]
    
    def calculate_breakthrough_score(self, paper: ScientificPaper) -> float:
        """Calculate likelihood of paper being a breakthrough"""
        score = 0.0
        
        text = (paper.title + ' ' + paper.abstract).lower()
        
        # Check for category-specific keywords
        for category in paper.categories:
            if category in self.BREAKTHROUGH_KEYWORDS:
                keywords = self.BREAKTHROUGH_KEYWORDS[category]
                matches = sum(1 for kw in keywords if kw.lower() in text)
                score += matches * 0.02
        
        # Check for high-impact terms
        high_impact_matches = sum(1 for term in self.HIGH_IMPACT_TERMS if term.lower() in text)
        score += high_impact_matches * 0.15
        
        # Citation bonus (if available)
        if paper.citations > 100:
            score += 0.2
        elif paper.citations > 50:
            score += 0.1
        elif paper.citations > 20:
            score += 0.05
        
        # Institution bonus
        prestigious = [Institution.MIT, Institution.CERN, Institution.CALTECH,
                      Institution.PRINCETON, Institution.CAMBRIDGE]
        if any(inst in prestigious for inst in paper.institutions):
            score += 0.1
        
        # Normalize to 0-1
        return min(1.0, score)


def populate_sample_data(db: ResearchDatabase):
    """Populate database with sample papers and breakthroughs"""
    
    # Sample Papers
    papers = [
        ScientificPaper(
            paper_id="hawking1974",
            title="Black hole explosions?",
            authors=["Stephen W. Hawking"],
            abstract="Quantum mechanics and general relativity predict that black holes emit thermal radiation. This represents a fundamental connection between gravity, thermodynamics, and quantum mechanics.",
            categories=[ResearchCategory.BLACK_HOLES, ResearchCategory.QUANTUM_FIELD_THEORY],
            institutions=[Institution.CAMBRIDGE],
            publication_date="1974-03-01",
            arxiv_id=None,
            doi="10.1038/248030a0",
            citations=15000,
            keywords=["hawking radiation", "black hole thermodynamics", "quantum effects"],
            breakthrough_score=1.0,
            notes="Foundational paper on Hawking radiation"
        ),
        ScientificPaper(
            paper_id="einstein1915",
            title="The Field Equations of Gravitation",
            authors=["Albert Einstein"],
            abstract="Field equations of general relativity relating spacetime curvature to energy-momentum tensor.",
            categories=[ResearchCategory.RELATIVITY],
            institutions=[Institution.PRINCETON],
            publication_date="1915-11-25",
            citations=50000,
            keywords=["general relativity", "einstein field equations", "gravity"],
            breakthrough_score=1.0,
            notes="Foundation of general relativity"
        ),
        ScientificPaper(
            paper_id="shor1994",
            title="Algorithms for quantum computation: discrete logarithms and factoring",
            authors=["Peter W. Shor"],
            abstract="Quantum algorithm for integer factorization providing exponential speedup over classical algorithms.",
            categories=[ResearchCategory.QUANTUM_ALGORITHMS, ResearchCategory.QUANTUM_COMPUTING],
            institutions=[Institution.MIT],
            publication_date="1994-11-20",
            citations=20000,
            keywords=["shor algorithm", "quantum factoring", "quantum speedup"],
            breakthrough_score=1.0,
            notes="Proved quantum advantage for factoring"
        ),
    ]
    
    for paper in papers:
        db.add_paper(paper)
    
    # Sample Breakthroughs
    breakthroughs = [
        Breakthrough(
            breakthrough_id="gw150914",
            title="First Direct Detection of Gravitational Waves",
            description="LIGO detected gravitational waves from merging black holes, confirming Einstein's century-old prediction.",
            date="2016-02-11",
            category=ResearchCategory.GRAVITATIONAL_WAVES,
            institution=Institution.MIT,
            key_researchers=["Rainer Weiss", "Kip Thorne", "Barry Barish"],
            related_papers=["ligo2016"],
            impact_score=10.0,
            verification_status="verified",
            notes="Nobel Prize 2017"
        ),
        Breakthrough(
            breakthrough_id="higgs2012",
            title="Discovery of Higgs Boson",
            description="CERN's LHC discovered the Higgs boson, completing the Standard Model of particle physics.",
            date="2012-07-04",
            category=ResearchCategory.HIGGS_BOSON,
            institution=Institution.CERN,
            key_researchers=["Peter Higgs", "François Englert"],
            related_papers=["atlas2012", "cms2012"],
            impact_score=10.0,
            verification_status="verified",
            notes="Nobel Prize 2013"
        ),
        Breakthrough(
            breakthrough_id="google_supremacy2019",
            title="Quantum Supremacy Demonstration",
            description="Google's Sycamore processor performed computation infeasible for classical computers.",
            date="2019-10-23",
            category=ResearchCategory.QUANTUM_COMPUTING,
            institution=Institution.MIT,  # Multiple institutions
            key_researchers=["John Martinis", "Sergio Boixo"],
            related_papers=["arora2019"],
            impact_score=8.5,
            verification_status="disputed",
            notes="Disputed by IBM"
        ),
    ]
    
    for breakthrough in breakthroughs:
        db.add_breakthrough(breakthrough)


def generate_research_prompt_for_gemini() -> str:
    """Generate comprehensive prompt for Gemini to track research"""
    
    prompt = """
# SCIENTIFIC RESEARCH TRACKING PROMPT FOR GEMINI

## Mission
You are tasked with monitoring, analyzing, and summarizing cutting-edge scientific research in the following domains:

### Primary Research Areas

#### 1. RELATIVISTIC ASTROPHYSICS & COSMOLOGY
**Key Figures**: Albert Einstein, Stephen Hawking, Roger Penrose, Kip Thorne
**Focus Areas**:
- Black hole physics and thermodynamics
- Singularity theorems and cosmic censorship
- Hawking radiation and information paradox
- Event horizons and ergospheres
- Gravitational wave astronomy (LIGO, Virgo, LISA)
- Spacetime geometry and curvature
- Wormholes and exotic spacetimes
- Dark matter and dark energy
- Big Bang cosmology and inflation
- Cosmic microwave background

**Key Institutions**: MIT, Caltech, LIGO, Cambridge (DAMTP), Princeton (IAS)

#### 2. SOLAR PHYSICS & STELLAR RESEARCH
**Focus Areas**:
- Solar flares and coronal mass ejections
- Solar wind and heliosphere
- Magnetic reconnection
- Solar cycles and sunspot dynamics
- Parker Solar Probe missions
- Solar corona heating problem
- Space weather prediction
- Stellar evolution and nucleosynthesis

**Key Institutions**: NASA, ESA, MIT Kavli Institute

#### 3. QUANTUM PHYSICS & COMPUTING
**Key Figures**: Niels Bohr, Werner Heisenberg, Richard Feynman, Peter Shor
**Focus Areas**:
- Quantum computers and processors
- Qubit engineering (superconducting, ion trap, photonic, topological)
- Quantum algorithms (Shor, Grover, VQE, QAOA)
- Quantum error correction and fault tolerance
- Quantum supremacy/advantage demonstrations
- Quantum entanglement and non-locality
- Quantum cryptography and QKD
- Quantum simulation
- Quantum machine learning
- Quantum sensors and metrology

**Key Institutions**: MIT CQE, IBM Quantum, Google Quantum AI, Caltech IQIM, Princeton

#### 4. QUANTUM INFORMATION THEORY
**Focus Areas**:
- Quantum channel capacity
- Quantum entropy and information
- Quantum teleportation
- Entanglement distillation
- Quantum communication protocols
- Quantum complexity theory
- Quantum Shannon theory

#### 5. PARTICLE PHYSICS & CERN
**Focus Areas**:
- Large Hadron Collider experiments
- Higgs boson properties and couplings
- Beyond Standard Model physics
- Supersymmetry searches
- Dark matter candidates
- Neutrino physics
- Quark-gluon plasma

**Key Institutions**: CERN (ATLAS, CMS, ALICE, LHCb), Fermilab

#### 6. THEORETICAL PHYSICS
**Focus Areas**:
- String theory and M-theory
- Loop quantum gravity
- Quantum field theory
- AdS/CFT correspondence
- Holographic principle
- Quantum foundations
- Unified theories (Theory of Everything)

**Key Institutions**: Princeton IAS, Perimeter Institute, Cambridge DAMTP

## Data Sources to Monitor

### 1. arXiv Preprint Server
- **gr-qc** (General Relativity and Quantum Cosmology)
- **quant-ph** (Quantum Physics)
- **hep-th** (High Energy Physics - Theory)
- **hep-ex** (High Energy Physics - Experiment)
- **astro-ph** (Astrophysics)
- **cond-mat** (Condensed Matter - for quantum materials)

### 2. Institution Websites
- MIT: CQE, LIGO Lab, Kavli Institute
- CERN: Document Server, LHC reports
- NASA: Astrophysics Division, Solar System Exploration
- Caltech: TAPIR, IQIM
- Princeton: Quantum Initiative, IAS
- Cambridge: DAMTP, Stephen Hawking Centre

### 3. Major Journals
- Nature
- Science
- Physical Review Letters
- Physical Review D
- Physical Review X
- Nature Physics
- Nature Astronomy
- Astrophysical Journal
- Monthly Notices RAS

## Tasks to Perform

### 1. Daily Monitoring
- Check arXiv new submissions in target categories
- Monitor institution news and press releases
- Track major journal publications
- Identify papers with high breakthrough potential

### 2. Breakthrough Detection
Flag papers containing:
- First observations or detections
- Experimental confirmations of theoretical predictions
- Novel phenomena or unexpected results
- Record-breaking measurements
- New technologies or methodologies
- Paradigm-shifting ideas

### 3. Categorization
For each paper/breakthrough:
- Assign research categories
- Extract key concepts and keywords
- Identify related work
- Assess potential impact (0-10 scale)
- Note institutional affiliations
- Extract author information

### 4. Literature Database Maintenance
- Add new papers to database with full metadata
- Update citation counts periodically
- Link related papers
- Track research trends over time
- Identify emerging subfields

### 5. Breakthrough Verification
- Cross-reference multiple sources
- Check for independent confirmation
- Monitor community response
- Track media coverage
- Update verification status

### 6. Research Summaries
Generate periodic summaries:
- Weekly highlights (top 10 papers)
- Monthly breakthrough reports
- Quarterly trend analysis
- Annual reviews by field

## Output Format

### For Individual Papers
```
PAPER_ID: unique_identifier
TITLE: full title
AUTHORS: author list
CATEGORIES: [category1, category2, ...]
INSTITUTIONS: [inst1, inst2, ...]
DATE: YYYY-MM-DD
ARXIV_ID: arXiv:XXXX.XXXXX
DOI: doi_if_available
ABSTRACT: full abstract
KEYWORDS: [keyword1, keyword2, ...]
BREAKTHROUGH_SCORE: 0.0-1.0
CITATIONS: number
PDF_URL: direct link
RELATED: [related_paper_ids]
NOTES: significance and context
```

### For Breakthroughs
```
BREAKTHROUGH_ID: unique_identifier
TITLE: breakthrough title
DESCRIPTION: detailed description
DATE: YYYY-MM-DD
CATEGORY: primary category
INSTITUTION: lead institution
RESEARCHERS: [key people]
IMPACT_SCORE: 0-10
VERIFICATION: pending/verified/disputed
MEDIA_COVERAGE: [links to coverage]
RELATED_PAPERS: [paper_ids]
SIGNIFICANCE: why this matters
```

## Special Emphasis Areas

### Black Holes (Hawking's Legacy)
- Information paradox resolution attempts
- Hawking radiation observations
- Black hole entropy and thermodynamics
- Quantum corrections to black holes
- Primordial black holes
- Supermassive black hole dynamics
- Black hole mergers and gravitational waves

### Quantum Computing (Practical Implementation)
- Qubit coherence time improvements
- Error correction code advances
- New qubit architectures
- Quantum algorithm discoveries
- Hardware scaling demonstrations
- Quantum software frameworks
- Commercial quantum computer deployments

### Solar Research (Understanding Our Star)
- Solar cycle predictions
- Space weather forecasting
- Solar probe measurements
- Corona heating mechanisms
- Solar interior dynamics

## Citation Analysis
Track and report:
- Most cited papers by category
- Citation networks and clusters
- Influential researchers
- Emerging research groups
- Cross-disciplinary connections

## Trend Detection
Identify:
- Rapidly growing subfields
- Declining research areas
- Controversial topics
- Consensus formations
- Paradigm shifts
- Technology transitions

## Quality Metrics
Assess papers on:
- Theoretical rigor
- Experimental precision
- Reproducibility
- Novelty
- Impact potential
- Clarity of presentation

## Alert Triggers
Generate immediate alerts for:
- Papers with breakthrough_score > 0.8
- First observations of predicted phenomena
- Major experimental results from LIGO, CERN, etc.
- Nobel-worthy discoveries
- Contradictions to established theory
- Quantum computing milestones

## Integration with Pandora AIOS
- Feed high-impact quantum computing papers to quantum overlay system
- Use latest black hole research to inform wormhole overlay logic
- Apply solar physics insights to energy management
- Incorporate quantum algorithm advances into system optimization
- Use cosmology research to inspire system architecture

## Remember
- Prioritize peer-reviewed work but don't ignore important preprints
- Consider interdisciplinary connections
- Maintain healthy skepticism of extraordinary claims
- Track replication attempts
- Note funding sources and potential conflicts
- Preserve scientific integrity above all

---

**"We stand on the shoulders of giants."** - Isaac Newton
**"The important thing is not to stop questioning."** - Albert Einstein
**"Remember to look up at the stars and not down at your feet."** - Stephen Hawking
"""
    
    return prompt


def main():
    """Main entry point for research tracker"""
    print("="*70)
    print("Pandora AIOS Scientific Research Tracker & Literature Database")
    print("="*70)
    print()
    
    # Initialize database
    db = ResearchDatabase()
    print(f"[INFO] Database initialized at: {db.db_path}")
    
    # Populate with sample data
    print("[INFO] Populating sample data...")
    populate_sample_data(db)
    
    # Show statistics
    stats = db.get_statistics()
    print(f"\n[INFO] Database Statistics:")
    print(f"  Total Papers: {stats['total_papers']}")
    print(f"  Total Breakthroughs: {stats['total_breakthroughs']}")
    print(f"  High Impact Papers: {stats['high_impact_papers']}")
    print(f"  Avg Breakthrough Score: {stats['avg_breakthrough_score']:.3f}")
    
    # Show research sources
    print(f"\n[INFO] Available Research Sources:")
    source_manager = ScientificSourceManager()
    
    for category in [ResearchCategory.BLACK_HOLES, ResearchCategory.QUANTUM_COMPUTING]:
        sources = source_manager.get_source_urls(categories=[category])
        print(f"\n{category.value.upper()}: {len(sources)} sources")
        for source in sources[:3]:
            print(f"  - {source['name']}")
            print(f"    {source['url']}")
    
    # Generate Gemini prompt
    print(f"\n[INFO] Generating Gemini research tracking prompt...")
    prompt = generate_research_prompt_for_gemini()
    
    prompt_path = os.path.expanduser("~/.pandora/gemini_research_prompt.txt")
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    with open(prompt_path, 'w') as f:
        f.write(prompt)
    
    print(f"[SUCCESS] Gemini prompt saved to: {prompt_path}")
    print(f"[INFO] Prompt length: {len(prompt)} characters")
    
    # Show recent breakthroughs
    print(f"\n[INFO] Recent Breakthroughs:")
    breakthroughs = db.get_recent_breakthroughs(days=3650, limit=5)  # Last 10 years
    for bt in breakthroughs:
        print(f"\n  {bt.title}")
        print(f"  Date: {bt.date} | Impact: {bt.impact_score}/10")
        print(f"  Institution: {bt.institution.value.upper()}")
        print(f"  Researchers: {', '.join(bt.key_researchers[:3])}")
    
    print("\n" + "="*70)
    print("Scientific Research Tracker Ready")
    print("="*70)


if __name__ == "__main__":
    main()
