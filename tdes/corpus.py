"""Deterministic synthetic corpus with provenance, plus cleaning and exact deduplication.

The corpus is deliberately small but carries every hazard the data system must handle:
exact duplicates, a source with an unknown license, a web page that leaked a benchmark item,
validation and test splits, a trusted OPUS proxy set and an anneal reserve.
"""
import inspect
import random
import re
import unicodedata

from .util import seed_int, sha256_json

# ----------------------------------------------------------------------------- sources
SOURCES = {
    "src_web_edu_crawl":       {"lane": "general_web", "license": "cc-by-4.0", "provenance_tier": "B", "language": "en", "script": "Latn"},
    "src_web_india_news":      {"lane": "general_web", "license": "licensed-news", "provenance_tier": "A", "language": "en", "script": "Latn"},
    "src_code_permissive":     {"lane": "code", "license": "mit/apache-2.0", "provenance_tier": "A", "language": "python", "script": "Latn"},
    "src_code_scraped_unknown": {"lane": "code", "license": "unknown", "provenance_tier": "C", "language": "python", "script": "Latn"},
    "src_math_textbooks":      {"lane": "math_science", "license": "cc-by-sa-4.0", "provenance_tier": "A", "language": "en", "script": "Latn"},
    "src_indic_hi_verified":   {"lane": "indic", "license": "cc-by-4.0", "provenance_tier": "A", "language": "hi", "script": "Deva"},
    "src_indic_ta_verified":   {"lane": "indic", "license": "cc-by-4.0", "provenance_tier": "A", "language": "ta", "script": "Taml"},
    "src_reasoning_banded":    {"lane": "reasoning", "license": "internal-generated", "provenance_tier": "A", "language": "en", "script": "Latn"},
    "src_agentic_tierA":       {"lane": "agentic", "license": "internal-generated", "provenance_tier": "A", "language": "en", "script": "Latn"},
    "src_anneal_curated":      {"lane": "anneal_reserve", "license": "internal-curated", "provenance_tier": "A", "language": "en+hi", "script": "Latn+Deva"},
    "bench_mini_arith_v1":     {"lane": "eval", "license": "benchmark", "provenance_tier": "A", "language": "en", "script": "Latn"},
    "bench_mini_hiqa_v1":      {"lane": "eval", "license": "benchmark", "provenance_tier": "A", "language": "hi", "script": "Deva"},
}

ALLOWED_LICENSES = {"cc-by-4.0", "cc-by-sa-4.0", "licensed-news", "mit/apache-2.0",
                    "internal-generated", "internal-curated", "benchmark"}

# ----------------------------------------------------------------------------- vocab
NOUNS = ["river", "market", "library", "farmer", "engineer", "monsoon", "village", "teacher", "bridge",
         "festival", "harbor", "railway", "forest", "museum", "school", "garden", "factory", "temple",
         "valley", "clinic", "orchard", "stadium", "canal", "bazaar"]
PLACES = ["Pune", "Chennai", "Kolkata", "Jaipur", "Kochi", "Mysuru", "Delhi", "Guwahati", "Bhopal",
          "Surat", "Nagpur", "Madurai", "Indore", "Shillong"]
VERBS = ["improved", "described", "visited", "changed", "protected", "rebuilt", "celebrated",
         "measured", "studied", "connected", "cleaned", "expanded"]
ADJS = ["old", "busy", "quiet", "new", "famous", "small", "green", "crowded", "historic", "modern",
        "bright", "narrow"]
TIMES = ["morning", "winter", "harvest season", "last decade", "festival week", "summer", "evening",
         "monsoon months"]

HI_NOUNS = ["गाँव", "नदी", "किसान", "शहर", "बाज़ार", "स्कूल", "पुस्तक", "बारिश", "त्योहार", "सड़क",
            "पानी", "खेत", "मंदिर", "पुल", "जंगल"]
HI_PLACES = ["दिल्ली", "जयपुर", "पुणे", "भोपाल", "लखनऊ", "पटना", "इंदौर"]
HI_ADJS = ["सुंदर", "बड़ा", "पुराना", "नया", "शांत", "व्यस्त", "हरा"]
HI_VERBS = ["है", "था", "दिखता है", "बदल गया", "मशहूर है", "साफ़ है"]

TA_NOUNS = ["ஊர்", "நதி", "விவசாயி", "நகரம்", "சந்தை", "பள்ளி", "புத்தகம்", "மழை", "திருவிழா", "சாலை",
            "தண்ணீர்", "கோவில்"]
TA_PLACES = ["சென்னையில்", "மதுரையில்", "கோவையில்", "திருச்சியில்", "சேலத்தில்"]
TA_ADJS = ["அழகான", "பெரிய", "பழைய", "புதிய", "அமைதியான", "பசுமையான"]
TA_VERBS = ["உள்ளது", "இருந்தது", "பிரபலமானது", "மாறியது"]

CODE_FUNCS = [
    "def {name}(values):\n    total = 0\n    for v in values:\n        total += v\n    return total\n",
    "def {name}(values):\n    if not values:\n        return 0.0\n    return sum(values) / len(values)\n",
    "def {name}(values, factor={k}):\n    return [v * factor for v in values]\n",
    "def {name}(values, low={a}, high={b}):\n    out = []\n    for v in values:\n        out.append(min(max(v, low), high))\n    return out\n",
    "def {name}(values):\n    return sum(1 for v in values if v % {k} == 0)\n",
    "def {name}(values):\n    best = 0\n    for a, b in zip(values, values[1:]):\n        best = max(best, abs(b - a))\n    return best\n",
    "def {name}(values):\n    acc = []\n    running = 0\n    for v in values:\n        running += v\n        acc.append(running)\n    return acc\n",
    "class {cls}:\n    def __init__(self, capacity={k}):\n        self.capacity = capacity\n        self.items = []\n\n    def push(self, item):\n        if len(self.items) >= self.capacity:\n            self.items.pop(0)\n        self.items.append(item)\n",
    "def {name}(text):\n    words = text.split()\n    counts = {{}}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n",
]
CODE_VERBS = ["total", "average", "scale", "clip", "count_multiples", "max_gap", "running_sum", "word_counts"]
CODE_OBJS = ["values", "scores", "prices", "temps", "readings", "votes", "rainfall"]

TOOLS = ["get_weather", "convert_currency", "search_trains"]

# Benchmark items use a vocabulary that never appears in training templates, so n-gram
# fingerprints identify true leakage instead of shared template phrases.
BENCH_NAMES = ["Zorvath", "Quenby", "Myrtelle", "Oskarine", "Ilsabet", "Thaddeon", "Verushka", "Calloway"]
BENCH_PLACES = ["Quillmere", "Vandrake Hollow", "Esterbrook Isle", "Morrowgate", "Pellinore Reach"]
BENCH_THINGS = ["brass astrolabes", "vellum charts", "glass prisms", "copper orreries", "ivory quadrants"]
HI_BENCH_NAMES = ["ज़ोरवाथ", "क्वेनबी", "मर्टेल", "ओस्कारीन"]
HI_BENCH_THINGS = ["पीतल के यंत्र", "काँच के प्रिज़्म", "ताँबे के दीपक"]


def _web_doc(r):
    noun, place = r.choice(NOUNS), r.choice(PLACES)
    templates = [
        "The {adj} {noun} in {place} {verb} the {noun2} during the {time}.",
        "Many people in {place} say the {noun} is {adj} and {adj2}.",
        "Reports from {place} show that the {noun2} was {verb} by local {noun}s.",
        "In the {time}, the {adj} {noun} near {place} attracts visitors from {place2}.",
        "A {adj} {noun2} was {verb} after the {time} in {place}.",
    ]
    sents = []
    for _ in range(r.randint(4, 11)):
        sents.append(r.choice(templates).format(
            adj=r.choice(ADJS), adj2=r.choice(ADJS), noun=noun, noun2=r.choice(NOUNS), verb=r.choice(VERBS),
            place=place, place2=r.choice(PLACES), time=r.choice(TIMES)))
    return f"Notes on the {noun} of {place}\n" + " ".join(sents)


def _math_doc(r):
    parts = []
    for _ in range(r.randint(2, 5)):
        kind = r.randint(0, 3)
        a, b = r.randint(2, 99), r.randint(2, 30)
        if kind == 0:
            parts.append(f"Problem: A train travels {a * b} km in {b} hours. What is its speed? "
                         f"Solution: speed = {a * b} / {b} = {a} km per hour.")
        elif kind == 1:
            parts.append(f"Problem: Find the area of a rectangle with sides {a} and {b}. "
                         f"Solution: area = {a} x {b} = {a * b} square units.")
        elif kind == 2:
            parts.append(f"Fact: {a} percent of {b * 100} is {a * b}. "
                         f"This follows because {b * 100} x {a} / 100 = {a * b}.")
        else:
            parts.append(f"Science: water boils at 100 degrees Celsius at sea level, and a sample of "
                         f"{a} grams warmed by {b} degrees needs about {a * b * 4} joules.")
    return "\n".join(parts)


def _code_doc(r):
    mod = f"{r.choice(CODE_OBJS)}_{r.choice(['utils', 'stats', 'tools', 'ops'])}"
    funcs = []
    for _ in range(r.randint(1, 5)):
        tpl = r.choice(CODE_FUNCS)
        funcs.append(tpl.format(name=f"{r.choice(CODE_VERBS)}_{r.choice(CODE_OBJS)}",
                                cls=f"{r.choice(['Rolling', 'Bounded', 'Recent'])}{r.choice(['Buffer', 'Window', 'Queue'])}",
                                k=r.randint(2, 9), a=r.randint(-5, 0), b=r.randint(10, 50)))
    return f"# module: {mod}.py\nimport math\n\n\n" + "\n\n".join(funcs)


def _hi_doc(r):
    sents = [f"{r.choice(HI_PLACES)} में {r.choice(HI_ADJS)} {r.choice(HI_NOUNS)} {r.choice(HI_VERBS)}।"
             for _ in range(r.randint(4, 10))]
    return " ".join(sents)


def _ta_doc(r):
    sents = [f"{r.choice(TA_PLACES)} {r.choice(TA_ADJS)} {r.choice(TA_NOUNS)} {r.choice(TA_VERBS)}."
             for _ in range(r.randint(4, 10))]
    return " ".join(sents)


def _reasoning_doc(r):
    n, k = r.randint(3, 12), r.randint(2, 9)
    g = r.randint(1, n * k - 1)
    total, res = n * k, n * k - g
    name = r.choice(["Ravi", "Meera", "Arjun", "Fatima", "Kavya", "John"])
    item = r.choice(["pens", "mangoes", "books", "tickets", "marbles"])
    problem = f"A shop packs {n} {item} per box. {name} buys {k} boxes and gives away {g} {item}. How many {item} remain?"
    think = (f"Each box holds {n} {item}.\nTotal = {n} x {k} = {total}.\n"
             f"After giving away {g}: {total} - {g} = {res}.\nCheck: {res} + {g} = {total}, which is correct.")
    return [{"role": "user", "text": problem}, {"role": "think", "text": think}, {"role": "answer", "text": str(res)}]


def _agentic_doc(r):
    tool = r.choice(TOOLS)
    city = r.choice(PLACES)
    if tool == "get_weather":
        t, cond = r.randint(18, 41), r.choice(["sunny", "cloudy", "raining", "humid"])
        turns = [("user", f"What is the weather in {city} right now?"),
                 ("assistant", "I will look up the current weather."),
                 ("tool_call", f'{{"tool": "get_weather", "city": "{city}"}}'),
                 ("tool_obs", f'{{"temp_c": {t}, "condition": "{cond}"}}'),
                 ("assistant", f"It is {t} degrees Celsius and {cond} in {city}.")]
    elif tool == "convert_currency":
        amt, rate = r.randint(5, 500), r.choice([83, 90, 1])
        turns = [("user", f"Convert {amt} USD to INR."),
                 ("assistant", "I will call the currency converter."),
                 ("tool_call", f'{{"tool": "convert_currency", "amount": {amt}, "from": "USD", "to": "INR"}}'),
                 ("tool_obs", f'{{"rate": {rate}, "result": {amt * rate}}}'),
                 ("assistant", f"{amt} USD is about {amt * rate} INR at a rate of {rate}.")]
    else:
        dst = r.choice(PLACES)
        num, hh = r.randint(12000, 22999), r.randint(5, 22)
        turns = [("user", f"Find a train from {city} to {dst} tomorrow."),
                 ("assistant", "I will search the train schedule."),
                 ("tool_call", f'{{"tool": "search_trains", "from": "{city}", "to": "{dst}"}}'),
                 ("tool_obs", f'{{"train": {num}, "departs": "{hh:02d}:15"}}'),
                 ("assistant", f"Train {num} leaves {city} for {dst} at {hh:02d}:15 tomorrow.")]
    return [{"role": role, "text": text} for role, text in turns]


def _anneal_doc(r):
    if r.random() < 0.6:
        a, b = r.randint(11, 99), r.randint(11, 99)
        return [{"role": "user", "text": f"What is {a} times {b}? Show the work."},
                {"role": "assistant", "text": f"{a} x {b} = {a} x {b // 10 * 10} + {a} x {b % 10} = "
                                              f"{a * (b // 10 * 10)} + {a * (b % 10)} = {a * b}."}]
    place = r.choice(HI_PLACES)
    return [{"role": "user", "text": f"{place} के बारे में एक वाक्य लिखिए।"},
            {"role": "assistant", "text": f"{place} एक {r.choice(HI_ADJS)} शहर है जहाँ {r.choice(HI_NOUNS)} {r.choice(HI_VERBS)}।"}]


def _bench_arith(i, r, canary):
    name, place, thing = r.choice(BENCH_NAMES), r.choice(BENCH_PLACES), r.choice(BENCH_THINGS)
    n, k = r.randint(300, 9999), r.randint(10, 290)
    return (f"Benchmark item {i}. In the archive of {place}, curator {name} catalogued {n} {thing}. "
            f"After lending {k} of them to a travelling exhibition, how many {thing} remain? "
            f"Answer: {n - k}. {canary}")


def _bench_hiqa(i, r, canary):
    name, thing = r.choice(HI_BENCH_NAMES), r.choice(HI_BENCH_THINGS)
    n, k = r.randint(300, 9999), r.randint(10, 290)
    return (f"मूल्यांकन प्रश्न {i}: संग्रहालय में {name} ने {n} {thing} गिने। {k} उधार देने के बाद कितने बचे? "
            f"उत्तर: {n - k}। {canary}")


GENERATORS = {
    "src_web_edu_crawl": ("plain", _web_doc), "src_web_india_news": ("plain", _web_doc),
    "src_code_permissive": ("code", _code_doc), "src_code_scraped_unknown": ("code", _code_doc),
    "src_math_textbooks": ("plain", _math_doc),
    "src_indic_hi_verified": ("plain", _hi_doc), "src_indic_ta_verified": ("plain", _ta_doc),
    "src_reasoning_banded": ("reasoning", _reasoning_doc),
    "src_agentic_tierA": ("agentic", _agentic_doc),
    "src_anneal_curated": ("chat", _anneal_doc),
}

# documents per source at scale 1.0: (train, validation)
COUNTS = {
    "src_web_edu_crawl": (110, 4), "src_web_india_news": (60, 3), "src_code_permissive": (90, 5),
    "src_code_scraped_unknown": (10, 0), "src_math_textbooks": (90, 5),
    "src_indic_hi_verified": (55, 3), "src_indic_ta_verified": (55, 3),
    "src_reasoning_banded": (80, 4), "src_agentic_tierA": (80, 4), "src_anneal_curated": (70, 3),
}
PROXY_SOURCES = {"src_web_edu_crawl": 4, "src_code_permissive": 3, "src_math_textbooks": 3}


def canary_string(seed, benchmark_id) -> str:
    return f"BENCHMARK-CANARY-{sha256_json([seed, benchmark_id])[:24]}"


def generate_corpus(cfg: dict) -> list:
    """Return raw documents (dicts) for every split. Pure function of the config seed."""
    seed, scale = cfg["seed"], cfg["corpus"]["scale"]
    docs = []

    def add(source_id, split, kind, content, extra=None):
        src = SOURCES[source_id]
        doc = {"doc_id": f"{source_id}/{split}/{len([d for d in docs if d['source_id'] == source_id and d['split'] == split]):05d}",
               "source_id": source_id, "split": split, "lane": src["lane"], "kind": kind,
               "language": src["language"], "script": src["script"], "license": src["license"],
               "provenance_tier": src["provenance_tier"]}
        if isinstance(content, str):
            doc["text"] = content
        else:
            doc["turns"] = content
        if extra:
            doc.update(extra)
        docs.append(doc)
        return doc

    for source_id, (n_train, n_val) in COUNTS.items():
        kind, gen = GENERATORS[source_id]
        for split, n in (("train", n_train), ("validation", n_val)):
            r = random.Random(seed_int(seed, "corpus", source_id, split))
            for _ in range(max(1 if n else 0, int(round(n * scale)))):
                add(source_id, split, kind, gen(r))

    for source_id, n in PROXY_SOURCES.items():
        kind, gen = GENERATORS[source_id]
        r = random.Random(seed_int(seed, "corpus", source_id, "proxy"))
        for _ in range(max(2, int(round(n * max(scale, 0.5))))):
            d = add(source_id, "proxy", kind, gen(r))
            d["lane"] = "opus_proxy"

    bench_docs = []
    for bench_id, fn, n in (("bench_mini_arith_v1", _bench_arith, 24), ("bench_mini_hiqa_v1", _bench_hiqa, 12)):
        r = random.Random(seed_int(seed, "corpus", bench_id))
        canary = canary_string(seed, bench_id)
        for i in range(max(4, int(round(n * max(scale, 0.5))))):
            bench_docs.append(add(bench_id, "test", "plain", fn(i, r, canary),
                                  {"benchmark_id": bench_id, "benchmark_version": "v1"}))

    # Hazard 1: exact duplicates inside the web crawl (removed by dedup).
    web = [d for d in docs if d["source_id"] == "src_web_edu_crawl" and d["split"] == "train"]
    for d in web[:4]:
        add("src_web_edu_crawl", "train", "plain", d["text"])
    # Hazard 2: a forum page that copied a benchmark item verbatim (caught by the eval firewall).
    leak = bench_docs[3]["text"]
    add("src_web_edu_crawl", "train", "plain",
        f"Forum thread: someone posted this quiz question, can anyone solve it?\n{leak}\nThanks in advance.")
    return docs


# ----------------------------------------------------------------------------- cleaning
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_TRAILING = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    """NFC normalisation (Indic-safe: composes matras/nukta consistently), strip control
    characters, strip trailing whitespace. Indentation is preserved for code."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = _TRAILING.sub("\n", text)
    return text.strip("\n")


CLEANING_PIPELINE = {
    "name": "tdes-clean",
    "version": "1.0",
    "steps": ["unicode_nfc", "strip_control_chars", "strip_trailing_ws", "exact_dedup_per_split"],
    "source_sha": sha256_json(inspect.getsource(clean_text)),
}
CLEANING_PIPELINE_HASH = sha256_json(CLEANING_PIPELINE)


def doc_content(doc) -> str:
    if "text" in doc:
        return doc["text"]
    return "\n".join(f"<{t['role']}>{t['text']}" for t in doc["turns"])


def doc_hash(doc) -> str:
    return sha256_json({"text": doc.get("text"), "turns": doc.get("turns")})


def clean_and_dedup(docs: list):
    """Clean every document and drop exact duplicates within each split.
    Returns (kept_docs, report)."""
    kept, seen, dropped = [], {}, []
    for d in docs:
        d = dict(d)
        if "text" in d:
            d["text"] = clean_text(d["text"])
        else:
            d["turns"] = [{"role": t["role"], "text": clean_text(t["text"])} for t in d["turns"]]
        d["doc_hash"] = doc_hash(d)
        key = (d["split"], d["doc_hash"])
        if key in seen:
            dropped.append({"doc_id": d["doc_id"], "duplicate_of": seen[key], "split": d["split"]})
            continue
        seen[key] = d["doc_id"]
        d["dedup_status"] = "exact_dedup_v1:unique"
        kept.append(d)
    report = {"cleaning_pipeline": CLEANING_PIPELINE, "cleaning_pipeline_hash": CLEANING_PIPELINE_HASH,
              "input_docs": len(docs), "kept_docs": len(kept), "dropped_duplicates": dropped}
    return kept, report
