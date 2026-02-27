"""
Italian Morphology Analysis API
Uses spaCy Italian pipeline for morphological structure analysis
Extracts roots, prefixes, suffixes, word formation types

Deployment: Docker container or any Python hosting
Cost: $0 (spaCy is free, open source)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

app = FastAPI(
    title="Italian Morphology API",
    description="Morphological analysis using spaCy Italian pipeline",
    version="1.0.0"
)

# Load Italian model (download once: python -m spacy download it_core_news_lg)
try:
    nlp = spacy.load("it_core_news_lg")
except OSError:
    # Fallback to smaller model if large not available
    try:
        nlp = spacy.load("it_core_news_sm")
    except OSError:
        nlp = None

# Italian prefix patterns with meanings
PREFIXES = {
    "in": "negation", "im": "negation", "il": "negation", "ir": "negation",
    "dis": "negation/reversal", "s": "negation/removal",
    "ri": "repetition", "re": "repetition",
    "pre": "before", "post": "after",
    "sotto": "under", "sopra": "above", "sovra": "above",
    "anti": "against", "contro": "against",
    "co": "together", "con": "together",
    "stra": "intensive", "arci": "intensive",
    "auto": "self", "multi": "many", "uni": "one",
    "inter": "between", "trans": "across", "extra": "beyond",
    "super": "above/excessive", "ultra": "beyond",
    "bi": "two", "tri": "three",
}

# Italian suffix patterns with meanings
SUFFIXES = {
    # Noun-forming
    "ità": "abstract noun (quality)", "tà": "abstract noun",
    "zione": "action/process noun", "sione": "action noun",
    "mento": "process/result noun", "ezza": "quality noun",
    "ura": "result/collective noun", "anza": "state noun", "enza": "state noun",
    "tore": "agent (male)", "trice": "agent (female)",
    "ista": "practitioner/believer", "ismo": "doctrine/system",
    "eria": "place/collection", "aio": "container/seller",
    # Adjective-forming
    "oso": "full of", "osa": "full of",
    "abile": "capable of being", "ibile": "capable of being",
    "evole": "tending to", "ale": "relating to", "ile": "relating to",
    "ico": "relating to", "ica": "relating to",
    "ivo": "tending to", "iva": "tending to",
    # Adverb-forming
    "mente": "manner adverb",
    # Diminutive/Augmentative
    "ino": "diminutive", "ina": "diminutive",
    "etto": "diminutive", "etta": "diminutive",
    "ello": "diminutive", "ella": "diminutive",
    "one": "augmentative", "ona": "augmentative",
    "accio": "pejorative", "accia": "pejorative",
    # Verb infinitives
    "are": "verb (1st conjugation)", "ere": "verb (2nd conjugation)", "ire": "verb (3rd conjugation)",
}

# Common Italian compound word starters
COMPOUND_MARKERS = [
    "capo", "porta", "lava", "apri", "asciuga", "para",
    "salva", "rompi", "copri", "taglia", "spazza", "gratta",
    "passa", "conta", "caccia", "guarda", "batti", "ferma",
]


class AnalyzeRequest(BaseModel):
    word: str
    translation: str = ""


class MorphologyResponse(BaseModel):
    word: str
    lemma: str
    pos: str
    morphFeatures: dict
    rootMorpheme: str
    wordFormationType: str  # SIMPLE | DERIVED | COMPOUND | PARASYNTHETIC | CONVERSION
    inflectionalClass: str | None
    isCompound: bool
    compoundComponents: list[str]
    derivationalMorphemes: list[str]
    prefixes: list[dict]  # [{prefix, meaning}]
    suffixes: list[dict]  # [{suffix, meaning}]
    derivationPath: str
    confidence: float
    source: str


def detect_prefixes(word: str, lemma: str) -> list[dict]:
    """Detect prefixes by comparing word to lemma and known patterns."""
    prefixes = []
    word_lower = word.lower()

    # Sort by length (longest first) to match longer prefixes first
    sorted_prefixes = sorted(PREFIXES.items(), key=lambda x: len(x[0]), reverse=True)

    for prefix, meaning in sorted_prefixes:
        if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
            # Check if the remaining part could be a root
            remainder = word_lower[len(prefix):]
            # Avoid matching if lemma also starts with this prefix (it's part of the root)
            if not lemma.lower().startswith(prefix) or len(prefixes) == 0:
                prefixes.append({"prefix": f"{prefix}-", "meaning": meaning})
                break  # Usually one prefix in Italian

    return prefixes


def detect_suffixes(word: str, lemma: str, pos: str) -> list[dict]:
    """Detect derivational suffixes."""
    suffixes = []
    word_lower = word.lower()

    # Sort by length (longest first) to match longer suffixes first
    sorted_suffixes = sorted(SUFFIXES.items(), key=lambda x: len(x[0]), reverse=True)

    for suffix, meaning in sorted_suffixes:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            # Skip verb infinitive endings as they're inflectional, not derivational
            if suffix in ["are", "ere", "ire"] and pos == "VERB":
                continue
            suffixes.append({"suffix": f"-{suffix}", "meaning": meaning})
            break  # Usually one main derivational suffix

    return suffixes


def extract_root(word: str, prefixes: list, suffixes: list) -> str:
    """Extract root by removing detected affixes."""
    root = word.lower()

    for p in prefixes:
        prefix = p["prefix"].rstrip("-")
        if root.startswith(prefix):
            root = root[len(prefix):]

    for s in suffixes:
        suffix = s["suffix"].lstrip("-")
        if root.endswith(suffix):
            root = root[:-len(suffix)]

    # Clean up: ensure root has at least 2 characters
    if len(root) < 2:
        return word.lower()

    return root


def detect_compound(word: str) -> tuple[bool, list[str]]:
    """Detect if word is a compound and extract components."""
    word_lower = word.lower()

    for marker in COMPOUND_MARKERS:
        if word_lower.startswith(marker) and len(word_lower) > len(marker) + 2:
            component2 = word_lower[len(marker):]
            # Verify second component is substantial
            if len(component2) >= 3:
                return True, [marker, component2]

    return False, []


def classify_word_formation(
    word: str,
    lemma: str,
    prefixes: list,
    suffixes: list,
    is_compound: bool,
    pos: str
) -> str:
    """Classify word formation type."""
    if is_compound:
        return "COMPOUND"

    has_prefix = len(prefixes) > 0
    has_suffix = len(suffixes) > 0

    if has_prefix and has_suffix:
        # Parasynthetic: prefix + root + suffix where neither prefix+root nor root+suffix exists alone
        # Common in Italian: im-bottigli-are, s-barc-are, in-vecchi-are
        if word.lower().endswith(("are", "ere", "ire")) and pos == "VERB":
            return "PARASYNTHETIC"
        return "DERIVED"
    elif has_prefix or has_suffix:
        return "DERIVED"
    elif lemma != word.lower() and pos != lemma:
        # Conversion: same form, different POS (e.g., "il bello" noun from "bello" adj)
        return "CONVERSION"
    else:
        return "SIMPLE"


def get_inflectional_class(pos: str, morph_features: dict, word: str) -> str | None:
    """Determine inflectional class."""
    word_lower = word.lower()

    if pos == "VERB":
        if word_lower.endswith("are"):
            return "1st conjugation (-are)"
        elif word_lower.endswith("ere"):
            return "2nd conjugation (-ere)"
        elif word_lower.endswith("ire"):
            return "3rd conjugation (-ire)"
        else:
            return "conjugated form"
    elif pos == "NOUN":
        gender = morph_features.get("Gender", "")
        number = morph_features.get("Number", "Sing")
        if gender == "Masc":
            return f"masculine noun ({number.lower()})"
        elif gender == "Fem":
            return f"feminine noun ({number.lower()})"
        else:
            return "noun"
    elif pos == "ADJ":
        return "adjective"
    elif pos == "ADV":
        return "adverb"

    return None


def build_derivation_path(root: str, prefixes: list, suffixes: list, word: str, is_compound: bool, components: list) -> str:
    """Build the derivation path showing word formation steps."""
    if is_compound and components:
        return " + ".join(components) + " → " + word

    if not prefixes and not suffixes:
        return word

    steps = [root]
    current = root

    # Add prefixes
    for p in prefixes:
        prefix = p["prefix"].rstrip("-")
        current = prefix + current
        steps.append(current)

    # Add suffixes
    for s in suffixes:
        suffix = s["suffix"].lstrip("-")
        current = current + suffix
        if current != steps[-1]:
            steps.append(current)

    # If final step doesn't match word, add the word
    if steps[-1].lower() != word.lower():
        steps.append(word)

    return " → ".join(steps)


@app.post("/analyze", response_model=MorphologyResponse)
async def analyze_word(request: AnalyzeRequest):
    """Analyze morphological structure of an Italian word."""
    if nlp is None:
        raise HTTPException(
            status_code=503,
            detail="spaCy Italian model not loaded. Run: python -m spacy download it_core_news_lg"
        )

    word = request.word.strip()

    if not word:
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    # SECURITY: Validate word length (max 100 characters)
    if len(word) > 100:
        raise HTTPException(
            status_code=400,
            detail="Word exceeds maximum length of 100 characters"
        )

    # SECURITY: Validate word contains only printable characters
    # Allow letters, accented characters, hyphens, apostrophes (common in Italian)
    if not all(c.isprintable() for c in word):
        raise HTTPException(
            status_code=400,
            detail="Word contains invalid characters (non-printable)"
        )

    # Process with spaCy
    doc = nlp(word)

    if len(doc) == 0:
        raise HTTPException(status_code=400, detail="Could not process word")

    token = doc[0]

    # Extract spaCy features
    lemma = token.lemma_
    pos = token.pos_
    morph_dict = token.morph.to_dict()

    # Detect compound
    is_compound, compound_components = detect_compound(word)

    # Detect affixes
    prefixes = detect_prefixes(word, lemma)
    suffixes = detect_suffixes(word, lemma, pos)

    # Extract root
    root = extract_root(word, prefixes, suffixes)

    # Classify formation type
    word_formation = classify_word_formation(word, lemma, prefixes, suffixes, is_compound, pos)

    # Get inflectional class
    inflectional_class = get_inflectional_class(pos, morph_dict, word)

    # Build derivational morphemes list
    derivational_morphemes = []
    for p in prefixes:
        derivational_morphemes.append(p["prefix"])
    for s in suffixes:
        derivational_morphemes.append(s["suffix"])

    # Build derivation path
    derivation_path = build_derivation_path(root, prefixes, suffixes, word, is_compound, compound_components)

    # Calculate confidence based on analysis depth
    confidence = 0.95
    if word_formation == "SIMPLE":
        confidence = 0.98
    elif is_compound:
        confidence = 0.90  # Compounds can be tricky
    elif word_formation == "PARASYNTHETIC":
        confidence = 0.85  # Parasynthetic is harder to detect accurately

    return MorphologyResponse(
        word=word,
        lemma=lemma,
        pos=pos,
        morphFeatures=morph_dict,
        rootMorpheme=root,
        wordFormationType=word_formation,
        inflectionalClass=inflectional_class,
        isCompound=is_compound,
        compoundComponents=compound_components,
        derivationalMorphemes=derivational_morphemes,
        prefixes=prefixes,
        suffixes=suffixes,
        derivationPath=derivation_path,
        confidence=confidence,
        source="spaCy-it_core_news_lg"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = nlp is not None
    model_name = nlp.meta.get("name", "unknown") if nlp else "not loaded"
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model": model_name,
        "model_loaded": model_loaded
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Italian Morphology API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze word morphology",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
