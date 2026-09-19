"""
Benchmark dataset for RAG quality evaluation.

Based on `reference_database_for_hazard_identification.pdf` (CFIA, 2008).

30 questions covering:
    - Factual accuracy (8)
    - Retrieval quality (8)
    - Citation accuracy (4)
    - Edge cases (6)
    - Arabic support (4)

Each question includes:
    - expected_answer: reference answer (for EM/F1)
    - expected_keywords: critical terms (for relevance)
    - expected_pages: pages where the answer appears (for retrieval/citation)
    - category: factual | retrieval | citation | edge | arabic
    - difficulty: easy | medium | hard
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with reference answer."""

    id: str
    question: str
    expected_answer: str
    expected_keywords: list[str]
    expected_pages: list[int]
    category: str  # factual | retrieval | citation | edge | arabic
    difficulty: str  # easy | medium | hard
    notes: str = ""

    def __repr__(self) -> str:
        return f"<Q{self.id}: {self.question[:60]}...>"


# ============================================================
# BENCHMARK QUESTIONS
# ============================================================

BENCHMARK_QUESTIONS: list[BenchmarkQuestion] = [

    # ------------------------------------------------------------
    # CATEGORY: FACTUAL (8 questions)
    # ------------------------------------------------------------
    BenchmarkQuestion(
        id="fact_01",
        question="What is the temperature range for growth of Vibrio cholerae?",
        expected_answer="10 to 45°C",
        expected_keywords=["10", "45", "vibrio cholerae", "temperature"],
        expected_pages=[201],
        category="factual",
        difficulty="easy",
        notes="Direct fact from Vibrio cholerae fact sheet (page 201)",
    ),
    BenchmarkQuestion(
        id="fact_02",
        question="What is the minimum water activity (Aw) for Clostridium perfringens growth?",
        expected_answer="0.93-0.95",
        expected_keywords=["0.93", "0.95", "water activity", "clostridium perfringens"],
        expected_pages=[194],
        category="factual",
        difficulty="easy",
        notes="Fact sheet for C. perfringens",
    ),
    BenchmarkQuestion(
        id="fact_03",
        question="What is the pH range for growth of Yersinia enterocolitica?",
        expected_answer="4.2 minimum, 10 maximum",
        expected_keywords=["4.2", "10", "ph", "yersinia enterocolitica"],
        expected_pages=[204],
        category="factual",
        difficulty="easy",
        notes="Yersinia fact sheet",
    ),
    BenchmarkQuestion(
        id="fact_04",
        question="What are the nine priority food allergens identified by Health Canada and CFIA?",
        expected_answer=(
            "Peanuts, tree nuts, sesame seeds, milk, eggs, seafood "
            "(fish, crustaceans, shellfish), soy, wheat, and sulphites"
        ),
        expected_keywords=[
            "peanuts", "tree nuts", "sesame", "milk", "eggs",
            "seafood", "soy", "wheat", "sulphites",
        ],
        expected_pages=[221],
        category="factual",
        difficulty="medium",
        notes="Health Canada/CFIA priority allergen list (page 221)",
    ),
    BenchmarkQuestion(
        id="fact_05",
        question="What is the threshold size at which Health Canada considers extraneous material a health risk?",
        expected_answer="2.0 mm or greater",
        expected_keywords=["2.0", "mm", "extraneous", "health risk"],
        expected_pages=[288],
        category="factual",
        difficulty="easy",
        notes="Physical Hazards section (page 288)",
    ),
    BenchmarkQuestion(
        id="fact_06",
        question="What is the source of Histamine (Scombroid) poisoning?",
        expected_answer=(
            "Foods contaminated with histamine, mainly fish. Bacteria convert "
            "histidine into histamine during temperature abuse and spoilage. "
            "Cooking does not destroy histamine."
        ),
        expected_keywords=["histamine", "fish", "histidine", "temperature abuse", "spoilage"],
        expected_pages=[217],
        category="factual",
        difficulty="medium",
        notes="Scombroid poisoning fact sheet (page 217)",
    ),
    BenchmarkQuestion(
        id="fact_07",
        question="What is Bovine Spongiform Encephalopathy (BSE) and which tissues are considered Specified Risk Material (SRM)?",
        expected_answer=(
            "BSE (Mad Cow Disease) is caused by a proteinaceous infectious particle (prion). "
            "In cattle over 30 months, SRM includes the skull, brain, spinal cord, and a portion "
            "of the small intestine. Removing SRM at slaughter is the most important public health measure."
        ),
        expected_keywords=["bse", "prion", "srm", "skull", "brain", "spinal cord", "small intestine"],
        expected_pages=[216],
        category="factual",
        difficulty="hard",
        notes="BSE fact sheet",
    ),
    BenchmarkQuestion(
        id="fact_08",
        question="According to the document, what are the three broad categories of food safety hazards?",
        expected_answer="Biological, Chemical, and Physical hazards",
        expected_keywords=["biological", "chemical", "physical"],
        expected_pages=[5, 7],
        category="factual",
        difficulty="easy",
        notes="Introduction (pages 5-7)",
    ),

    # ------------------------------------------------------------
    # CATEGORY: RETRIEVAL (8 questions)
    # ------------------------------------------------------------
    BenchmarkQuestion(
        id="ret_01",
        question="What are the control measures for preventing Listeria monocytogenes contamination?",
        expected_answer=(
            "Control measures include: proper sanitation of food contact surfaces, "
            "preventing condensate from overhead structures onto product, "
            "maintaining appropriate refrigeration temperatures, "
            "and preventing cross-contamination from raw products."
        ),
        expected_keywords=["listeria", "sanitation", "condensate", "refrigeration", "cross-contamination"],
        expected_pages=[92, 95, 99, 100, 106, 127, 133],
        category="retrieval",
        difficulty="medium",
        notes="Appears in multiple sections: cooling, holding, sanitation",
    ),
    BenchmarkQuestion(
        id="ret_02",
        question="How can allergen cross-contamination be prevented in food processing?",
        expected_answer=(
            "Prevention strategies include: dedicated storage areas for allergenic ingredients, "
            "storing allergens below non-allergenic foods, color-coding, "
            "dedicated production lines, minimizing shared equipment, "
            "production scheduling (allergen products at the end), "
            "thorough cleaning between runs, and employee training."
        ),
        expected_keywords=["dedicated", "storage", "allergen", "clean", "scheduling", "training"],
        expected_pages=[221, 238, 239],
        category="retrieval",
        difficulty="medium",
        notes="From Allergen Prevention Plan section",
    ),
    BenchmarkQuestion(
        id="ret_03",
        question="What are the key factors in determining the risk of physical hazards?",
        expected_answer=(
            "Factors include: target audience, type of product, method of consumption, "
            "size of material, hardness, sharpness, shape, type of material, "
            "and ease of discovery."
        ),
        expected_keywords=["target audience", "size", "hardness", "sharpness", "shape"],
        expected_pages=[288],
        category="retrieval",
        difficulty="medium",
        notes="Physical hazards risk factors",
    ),
    BenchmarkQuestion(
        id="ret_04",
        question="What is the recommended approach for cooling food products to prevent pathogen growth?",
        expected_answer=(
            "Cool food rapidly to prevent growth. Use adequate air flow, "
            "avoid overloading cooling areas, ensure proper time/temperature application, "
            "and monitor for condensate dripping."
        ),
        expected_keywords=["cool", "rapid", "time", "temperature", "air flow", "pathogen"],
        expected_pages=[92, 98, 99, 106],
        category="retrieval",
        difficulty="hard",
        notes="From cooling sections",
    ),
    BenchmarkQuestion(
        id="ret_05",
        question="What are the main biological hazards in fish and seafood products?",
        expected_answer=(
            "Pathogens (Salmonella, Shigella, Vibrio parahaemolyticus, E. coli, "
            "Listeria monocytogenes, Staphylococcus aureus, Clostridium botulinum), "
            "parasites (Anisakis simplex, Pseudoterranova decipiens), "
            "Histamine, Ciguatera Toxin, Paralytic Shellfish Poisoning, "
            "Amnesic Shellfish Poisoning, Diarrhetic Shellfish Poisoning, "
            "and environmental contaminants."
        ),
        expected_keywords=[
            "salmonella", "vibrio", "listeria", "parasites", "anisakis",
            "histamine", "shellfish poisoning",
        ],
        expected_pages=[25, 35],
        category="retrieval",
        difficulty="hard",
        notes="Fish / Marine mammals section (pages 25, 35)",
    ),
    BenchmarkQuestion(
        id="ret_06",
        question="What are good manufacturing practices (GMPs) for allergen control?",
        expected_answer=(
            "GMPs include: appropriate standards for manufacturing and handling, "
            "design of premises, transportation and storage, maintenance of equipment, "
            "sanitation, personnel hygiene, and training. "
            "Controlling food allergens is an essential part of GMPs."
        ),
        expected_keywords=["gmp", "allergen", "hygiene", "sanitation", "training"],
        expected_pages=[235, 236],
        category="retrieval",
        difficulty="medium",
        notes="GMPs section",
    ),
    BenchmarkQuestion(
        id="ret_07",
        question="How should allergenic foods be stored to prevent cross-contamination?",
        expected_answer=(
            "Store allergenic foods and ingredients in dedicated areas. "
            "If dedicated areas aren't possible, store them below non-allergenic foods "
            "(e.g., on bottom shelves) to prevent falling. "
            "Clearly identify using signs or color codes."
        ),
        expected_keywords=["dedicated", "storage", "below", "bottom shelf", "signs", "color codes"],
        expected_pages=[221, 238],
        category="retrieval",
        difficulty="easy",
        notes="From Allergen Prevention Plan",
    ),
    BenchmarkQuestion(
        id="ret_08",
        question="What are the signs and symptoms of foodborne illness caused by bacterial toxins?",
        expected_answer=(
            "Symptoms include nausea, vomiting, abdominal cramps, chills, sweats, "
            "shock, shallow respiration, dizziness, headache, dryness of mouth & throat, "
            "muscle paralysis, and breathing difficulties. "
            "Onset is 30 minutes to 72 hours; recovery 1-10 days."
        ),
        expected_keywords=[
            "nausea", "vomiting", "abdominal", "30 minutes", "72 hours", "recovery",
        ],
        expected_pages=[190],
        category="retrieval",
        difficulty="medium",
        notes="Bacterial agents introduction (page 190)",
    ),

    # ------------------------------------------------------------
    # CATEGORY: CITATION (4 questions)
    # ------------------------------------------------------------
    BenchmarkQuestion(
        id="cite_01",
        question="What is the pH range for Clostridium perfringens growth?",
        expected_answer="5 minimum, 9 maximum",
        expected_keywords=["5", "9", "ph", "clostridium perfringens"],
        expected_pages=[194],
        category="citation",
        difficulty="easy",
        notes="Tests exact page citation",
    ),
    BenchmarkQuestion(
        id="cite_02",
        question="What are the symptoms of Vitamin A deficiency?",
        expected_answer=(
            "Night blindness, Bitot's Spots, Xerophthalmia (blindness), Keratinization, "
            "rough skin, susceptibility to infection, impaired bone growth, "
            "abnormal tooth and jaw alignment, and anemia."
        ),
        expected_keywords=[
            "night blindness", "bitot", "xerophthalmia", "keratinization",
        ],
        expected_pages=[244],
        category="citation",
        difficulty="medium",
        notes="Vitamin A fact sheet (page 244)",
    ),
    BenchmarkQuestion(
        id="cite_03",
        question="What are the priority allergen sources according to Health Canada?",
        expected_answer=(
            "Nine priority allergens: Peanuts, Tree nuts, Sesame Seeds, Milk, Eggs, "
            "Seafood (fish, crustaceans, shellfish), Soy, Wheat, Sulphites."
        ),
        expected_keywords=["peanuts", "tree nuts", "milk", "eggs", "soy", "wheat"],
        expected_pages=[221],
        category="citation",
        difficulty="easy",
        notes="Table 1 - Priority Allergen List",
    ),
    BenchmarkQuestion(
        id="cite_04",
        question="What are the biological agents categorized in the document?",
        expected_answer=(
            "Bacteria (Bacillus cereus, Campylobacter jejuni, Clostridium botulinum, "
            "Clostridium perfringens, E. coli, E. coli O157:H7, Listeria monocytogenes, "
            "Salmonella, Shigella, Staphylococcus aureus, Vibrio cholerae, Vibrio "
            "parahaemolyticus, Vibrio vulnificus, Yersinia enterocolitica); "
            "Viruses (Bacteriophage, Enteric Virus, Hepatitis A, Norovirus); "
            "Fungi and Moulds (Mycotoxigenic fungi); "
            "Parasites (Cryptosporidium, Giardia, Taenia, Toxoplasma, Trichinella); "
            "Other Biologics (Prions, Scombroid)."
        ),
        expected_keywords=[
            "bacteria", "viruses", "fungi", "parasites", "prions",
            "salmonella", "listeria", "hepatitis",
        ],
        expected_pages=[188, 189],
        category="citation",
        difficulty="hard",
        notes="Biological Hazards Section overview",
    ),

    # ------------------------------------------------------------
    # CATEGORY: EDGE (6 questions)
    # ------------------------------------------------------------
    BenchmarkQuestion(
        id="edge_01",
        question="What is the exact recipe for making chocolate cake?",
        expected_answer="NOT IN DOCUMENT - The document does not contain recipes",
        expected_keywords=["not available", "not found", "no information"],
        expected_pages=[],
        category="edge",
        difficulty="easy",
        notes="Negative test - should reject",
    ),
    BenchmarkQuestion(
        id="edge_02",
        question="What are the cosmetic regulations in France?",
        expected_answer="NOT IN DOCUMENT - The document covers food safety, not cosmetics",
        expected_keywords=["not available", "not found", "no information"],
        expected_pages=[],
        category="edge",
        difficulty="easy",
        notes="Negative test - off-topic",
    ),
    BenchmarkQuestion(
        id="edge_03",
        question=(
            "If a food processing plant needs to prevent both Listeria contamination "
            "AND allergen cross-contamination, what comprehensive control strategy "
            "would you recommend?"
        ),
        expected_answer=(
            "A comprehensive strategy requires: (1) For Listeria: "
            "proper sanitation of food contact surfaces, prevent condensate dripping, "
            "maintain proper refrigeration, prevent cross-contamination from raw products. "
            "(2) For allergens: dedicated storage areas, color-coding, production scheduling, "
            "thorough cleaning between runs, employee training. "
            "(3) Both: solid HACCP plan, GMPs, monitoring and verification."
        ),
        expected_keywords=[
            "listeria", "allergen", "sanitation", "dedicated", "haccp",
            "gmp", "training", "monitoring",
        ],
        expected_pages=[92, 95, 221, 235, 238],
        category="edge",
        difficulty="hard",
        notes="Multi-hop question combining two topics",
    ),
    BenchmarkQuestion(
        id="edge_04",
        question="What is the temperature range for growth of a specific bacterium?",
        expected_answer=(
            "Ambiguous - requires specifying which bacterium. "
            "The document contains fact sheets for multiple bacteria (Vibrio cholerae, "
            "Yersinia enterocolitica, Clostridium perfringens, etc.) each with different ranges."
        ),
        expected_keywords=["specific", "which", "bacterium", "clarify"],
        expected_pages=[194, 201, 204],
        category="edge",
        difficulty="medium",
        notes="Ambiguous question - should ask for clarification or list options",
    ),
    BenchmarkQuestion(
        id="edge_05",
        question="What year was the CFIA Reference Database for Hazard Identification published?",
        expected_answer="2008-03-01 (visible on most pages)",
        expected_keywords=["2008"],
        expected_pages=[2, 5, 7],
        category="edge",
        difficulty="easy",
        notes="Metadata question",
    ),
    BenchmarkQuestion(
        id="edge_06",
        question="What is the difference between food allergy and food intolerance?",
        expected_answer=(
            "Food allergy is an abnormal IMMUNE response to proteins found in food "
            "(involves antibodies). Symptoms include skin rash, itching, migraine, "
            "drop in blood pressure, anaphylaxis. "
            "Food intolerance is an abnormal PHYSIOLOGICAL response that does NOT "
            "involve the immune system (e.g., lactose intolerance, MSG headache, "
            "sulphite reaction). Symptoms include diarrhea and bloating."
        ),
        expected_keywords=[
            "immune", "allergy", "intolerance", "antibodies", "anaphylaxis",
            "lactose", "physiological",
        ],
        expected_pages=[221, 222],
        category="edge",
        difficulty="medium",
        notes="Comparison question",
    ),

    # ------------------------------------------------------------
    # CATEGORY: ARABIC (4 questions)
    # ------------------------------------------------------------
    BenchmarkQuestion(
        id="ar_01",
        question="ما هي المدة الزمنية والظروف اللازمة لمنع نمو البكتيريا في الأغذية؟",
        expected_answer=(
            "التحكم في درجة الحرارة والوقت: تبريد سريع للأغذية، "
            "تجنب نطاق 10-52°C حيث تنمو البكتيريا، الحفاظ على التبريد المناسب، "
            "ومنع التلوث المتبادل."
        ),
        expected_keywords=["حرارة", "وقت", "تبريد", "بكتيريا", "تلوث"],
        expected_pages=[92, 98, 99, 106, 194],
        category="arabic",
        difficulty="medium",
        notes="Tests Arabic query → Arabic answer",
    ),
    BenchmarkQuestion(
        id="ar_02",
        question="ما هي مسببات الحساسية التسعة ذات الأولوية في كندا؟",
        expected_answer=(
            "الفول السوداني، المكسرات، بذور السمسم، الحليب، البيض، "
            "المأكولات البحرية (الأسماك، القشريات، المحار)، الصويا، القمح، "
            "والكبريتيت."
        ),
        expected_keywords=[
            "الفول السوداني", "المكسرات", "السمسم", "الحليب", "البيض",
            "الصويا", "القمح", "الكبريتيت",
        ],
        expected_pages=[221],
        category="arabic",
        difficulty="medium",
        notes="Tests Arabic translation of allergen list",
    ),
    BenchmarkQuestion(
        id="ar_03",
        question="ما هي درجات الحرارة الموصى بها لطهي اللحوم للقضاء على البكتيريا الممرضة؟",
        expected_answer=(
            "طهي اللحوم حتى درجة حرارة داخلية 70-75°C للحم الخنزير، "
            "و71°C للحوم الطرائد. للأسماك: 90°C لمدة 1.5 دقيقة. "
            "تسخين السالمونيلا يتطلب درجة حرارة أعلى من 74°C."
        ),
        expected_keywords=["70", "75", "71", "طهي", "حرارة", "بكتيريا"],
        expected_pages=[215, 207, 208],
        category="arabic",
        difficulty="hard",
        notes="Multi-page Arabic answer",
    ),
    BenchmarkQuestion(
        id="ar_04",
        question="كيف يمكن منع التلوث المتبادل في مصانع الأغذية؟",
        expected_answer=(
            "منع التلوث المتبادل يشمل: تنظيف وتعقيم الأسطح والأدوات، "
            "فصل مناطق الأغذية النيئة عن المطهوة، "
            "تدريب الموظفين، جدولة الإنتاج بشكل صحيح، "
            "استخدام معدات مخصصة، ومنع الحركة غير الصحيحة للموظفين."
        ),
        expected_keywords=[
            "تنظيف", "تعقيم", "فصل", "تدريب", "جدولة", "معدات",
        ],
        expected_pages=[92, 100, 221, 238, 239],
        category="arabic",
        difficulty="medium",
        notes="Arabic multi-hop question",
    ),
]


# ============================================================
# HELPERS
# ============================================================

def get_questions_by_category(category: str) -> list[BenchmarkQuestion]:
    """Return questions filtered by category."""
    return [q for q in BENCHMARK_QUESTIONS if q.category == category]


def get_questions_by_difficulty(difficulty: str) -> list[BenchmarkQuestion]:
    """Return questions filtered by difficulty."""
    return [q for q in BENCHMARK_QUESTIONS if q.difficulty == difficulty]


def get_question_by_id(qid: str) -> BenchmarkQuestion | None:
    """Return a single question by ID."""
    for q in BENCHMARK_QUESTIONS:
        if q.id == qid:
            return q
    return None


__all__ = [
    "BenchmarkQuestion",
    "BENCHMARK_QUESTIONS",
    "get_questions_by_category",
    "get_questions_by_difficulty",
    "get_question_by_id",
]
