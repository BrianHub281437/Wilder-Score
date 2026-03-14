import os
import json
from typing import Any
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
SEED = 42

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
VALID_LANGUAGES = {"auto", "python", "javascript", "typescript", "java", "cpp", "csharp", "sql"}
VALID_INTENSITIES = {"gentle", "medium", "savage"}
MAX_CODE_LENGTH = 10000

# ---------------------------------------------------------------------------
# Corrected code truncation
# ---------------------------------------------------------------------------
MAX_CORRECTED_CODE_LENGTH = 3000


def truncate_corrected_code(corrected_code: str) -> str:
    """
    Truncates corrected code if it exceeds MAX_CORRECTED_CODE_LENGTH characters.
    Cuts at last newline to avoid mid-line truncation.
    Appends a note explaining the truncation.
    """
    if len(corrected_code) <= MAX_CORRECTED_CODE_LENGTH:
        return corrected_code

    truncated = corrected_code[:MAX_CORRECTED_CODE_LENGTH]
    last_newline = truncated.rfind("\n")
    if last_newline > 0:
        truncated = truncated[:last_newline]

    return (
        truncated
        + "\n\n-- Output truncated. "
        "File too large for full rewrite. "
        "Apply the patterns shown above "
        "to the rest of the file. --"
    )


# ---------------------------------------------------------------------------
# Code size detection
# ---------------------------------------------------------------------------
def count_non_empty_lines(code: str) -> int:
    return sum(1 for line in code.split("\n") if line.strip())


def get_size_bucket(line_count: int) -> str:
    if line_count <= 20:
        return f"Snippet ({line_count} lines)"
    elif line_count <= 50:
        return f"Short ({line_count} lines)"
    elif line_count <= 150:
        return f"Medium ({line_count} lines)"
    elif line_count <= 300:
        return f"Long ({line_count} lines)"
    else:
        return f"Very Long ({line_count} lines)"


def get_size_key(size_bucket: str) -> str:
    bucket = size_bucket.lower()
    if "very long" in bucket:
        return "very_long"
    if "snippet" in bucket:
        return "snippet"
    if "short" in bucket:
        return "short"
    if "medium" in bucket:
        return "medium"
    if "long" in bucket:
        return "long"
    return "medium"


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------
STANDARD_LANGUAGES = {"Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "SQL"}

LANGUAGE_DISPLAY_NAMES = {
    "auto":       None,
    "python":     "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java":       "Java",
    "cpp":        "C++",
    "csharp":     "C#",
    "sql":        "SQL",
}

AI_LANGUAGE_NORMALIZER = {
    "c++":         "C++",
    "cpp":         "C++",
    "c plus plus": "C++",
    "c":           "C",
    "c#":          "C#",
    "csharp":      "C#",
    "c sharp":     "C#",
    "cs":          "C#",
    "js":          "JavaScript",
    "javascript":  "JavaScript",
    "ts":          "TypeScript",
    "typescript":  "TypeScript",
    "py":          "Python",
    "python":      "Python",
    "java":        "Java",
    "sql":         "SQL",
}


def resolve_language(user_language: str, ai_language: str) -> tuple:
    """
    Returns (display_name, is_approximate).
    is_approximate is True only when user selected auto AND resolved language
    is not in STANDARD_LANGUAGES.
    C maps to "C" honestly, not "C++" incorrectly.
    User explicit selections are always trusted.
    """
    if user_language != "auto":
        display = LANGUAGE_DISPLAY_NAMES.get(user_language, user_language.upper())
        return display, False

    normalized = ai_language.lower().strip()
    display = AI_LANGUAGE_NORMALIZER.get(normalized, ai_language)
    is_approximate = display not in STANDARD_LANGUAGES
    return display, is_approximate


# ---------------------------------------------------------------------------
# Mistakes validation
# ---------------------------------------------------------------------------
VALID_SEVERITIES = {"small", "medium", "big", "critical"}

SEVERITY_FALLBACK = {
    "minor":    "small",
    "trivial":  "small",
    "style":    "small",
    "warning":  "small",
    "moderate": "medium",
    "major":    "big",
    "large":    "big",
    "serious":  "big",
    "severe":   "critical",
    "fatal":    "critical",
    "error":    "critical",
    "blocker":  "critical",
}


def validate_and_clean_mistakes(mistakes: Any) -> list:
    if not isinstance(mistakes, list):
        return []
    clean = []
    for item in mistakes:
        if not isinstance(item, dict):
            continue
        issue = item.get("issue")
        if not issue or not isinstance(issue, str):
            continue
        severity = item.get("severity")
        if not severity or not isinstance(severity, str):
            continue
        severity = severity.lower().strip()
        if severity not in VALID_SEVERITIES:
            severity = SEVERITY_FALLBACK.get(severity, "small")
        clean.append({"issue": issue.strip(), "severity": severity})
    return clean


def get_fallback_mistakes() -> list:
    """
    Only used when AI response is malformed or unusable.
    Never used for genuinely empty mistakes list.
    Perfect code should score 100.
    """
    return [
        {
            "issue": "Code could not be fully analyzed. Results may be incomplete.",
            "severity": "small",
        }
    ]


# ---------------------------------------------------------------------------
# Scoring tables
# ---------------------------------------------------------------------------
DEDUCTION_TABLES = {
    "snippet": {
        "small":    [-5,   -4,   -3,   -2,   -1  ],
        "medium":   [-14,  -11,  -8,   -6,   -4  ],
        "big":      [-28,  -22,  -17,  -13,  -9  ],
        "critical": [-45,  -35,  -27,  -20,  -14 ],
    },
    "short": {
        "small":    [-4,   -3,   -2,   -2,   -1  ],
        "medium":   [-11,  -8,   -6,   -5,   -3  ],
        "big":      [-22,  -17,  -13,  -10,  -7  ],
        "critical": [-38,  -29,  -22,  -17,  -12 ],
    },
    "medium": {
        "small":    [-3,   -2,   -2,   -1,   -1  ],
        "medium":   [-9,   -7,   -5,   -4,   -3  ],
        "big":      [-18,  -14,  -11,  -8,   -6  ],
        "critical": [-35,  -27,  -20,  -15,  -11 ],
    },
    "long": {
        "small":    [-2,   -2,   -1,   -1,   -0.5],
        "medium":   [-7,   -5,   -4,   -3,   -2  ],
        "big":      [-14,  -11,  -8,   -6,   -4  ],
        "critical": [-28,  -21,  -16,  -12,  -8  ],
    },
    "very_long": {
        "small":    [-1,   -1,   -0.5, -0.5, -0.5],
        "medium":   [-4,   -3,   -2,   -2,   -1  ],
        "big":      [-9,   -7,   -5,   -4,   -3  ],
        "critical": [-18,  -14,  -10,  -8,   -6  ],
    },
}

VERDICT_MAP = [
    (91, 100, "Jason Wilder Approved"),
    (76, 90,  "Almost Wilder-Worthy"),
    (51, 75,  "Wilder Would Sigh"),
    (26, 50,  "Wilder Is Disappointed"),
    (0,  25,  "Jason Wilder Has Left The Building"),
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def get_ordinal(n: int) -> str:
    if n == 1:
        return "1st"
    if n == 2:
        return "2nd"
    if n == 3:
        return "3rd"
    return f"{n}th"


def get_deduction(table: dict, severity: str, count: int) -> float:
    rates = table.get(severity, [-1])
    index = min(count, len(rates) - 1)
    return rates[index]


def format_deduction(deduction: float) -> str:
    """
    Preserves half points instead of truncating.
    -2   -> "-2"
    -0.5 -> "-0.5"
    -1.5 -> "-1.5"
    """
    if deduction == int(deduction):
        return str(int(deduction))
    return str(deduction)


def calculate_score(mistakes: list, size_key: str) -> tuple:
    """
    Returns (score, deductions).
    deductions is a list of dicts: [{"text": "...", "points": -4.0}]
    text is the human-readable description. points is the float value.
    Uses int() not round() so half-point deductions are never rounded away.
    100 - 0.5 = 99.5 -> int() -> 99 not 100.
    max(0) applied before int() so we never call int() on a negative number.
    Empty mistakes list returns 100. Perfect code gets Jason Wilder Approved.
    """
    table = DEDUCTION_TABLES[size_key]
    score = 100.0
    counts = {"small": 0, "medium": 0, "big": 0, "critical": 0}
    deductions = []

    for mistake in mistakes:
        severity = mistake.get("severity", "small")
        if severity not in counts:
            severity = "small"
        issue = mistake.get("issue", "Unknown issue")
        count = counts[severity]
        deduction = get_deduction(table, severity, count)
        counts[severity] += 1
        score += deduction
        ordinal = get_ordinal(counts[severity])
        formatted = format_deduction(deduction)

        deductions.append({
            "text": (
                f"{issue}: {formatted} points "
                f"({ordinal} {severity} mistake)"
            ),
            "points": deduction,
        })

    score = int(max(0.0, min(100.0, score)))
    return score, deductions


def get_verdict(score: int) -> str:
    for low, high, label in VERDICT_MAP:
        if low <= score <= high:
            return label
    return "Jason Wilder Has Left The Building"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
def build_system_prompt(intensity: str, language: str, size_bucket: str, line_count: int) -> str:
    intensity_tone = {
        "gentle": (
            "Jason Wilder on a good day, encouraging but still notices everything, "
            "light sarcasm"
        ),
        "medium": (
            "Jason Wilder on a normal day, sarcastic and exacting, calls out every mistake"
        ),
        "savage": (
            "Jason Wilder on his worst day, merciless, makes developer feel like they failed, "
            "still educational"
        ),
    }.get(intensity, "Jason Wilder on a normal day, sarcastic and exacting, calls out every mistake")

    language_specific = ""
    if language == "java":
        language_specific = """
If Java apply Jason Wilder's exact BCIT standards on top of the general mistakes above:
- Missing JavaDoc on public methods (small)
- Parameters not final (small each)
- No explicit units in names e.g. price not priceUsd (small each)
- Method does more than one thing (medium each)
- Class does more than one thing (big each)
- Concrete collection type e.g. ArrayList not List (medium each)
- Variables not declared then initialized (small each)
- System.out.print in final code (small each)
- Missing braces on single line conditions (small each)
- Constants not CAPITALIZED_WITH_UNDERSCORES (small each)
- Method name not camelCase verb (small each)
- Instance variables not private final (medium each)
- Missing class level JavaDoc with @author and @version (small)"""
    elif language == "python":
        language_specific = """
If Python:
- Missing type hints (small each)
- Missing docstrings (small each)
- Not snake_case (small each)
- PEP 8 violations (small each)
- Constants not UPPERCASE (small each)
- Bare except clauses (medium each)"""
    elif language in ("javascript", "typescript"):
        language_specific = """
If JavaScript or TypeScript:
- var instead of const or let (small each)
- console.log left in code (small each)
- Missing TypeScript type annotations (small each)
- Raw promises instead of async/await (small each)"""
    elif language == "cpp":
        language_specific = """
If C++:
- Raw pointers instead of smart pointers (big each)
- Missing const correctness (small each)
- Memory leaks (critical each)
- Missing RAII principles (big each)"""
    elif language == "csharp":
        language_specific = """
If C#:
- Missing XML documentation (small each)
- Public fields instead of properties (medium each)
- Missing readonly (small each)
- Interface not prefixed with I (small each)"""
    elif language == "sql":
        language_specific = """
If SQL:
- Lowercase keywords (small each)
- SELECT * usage (medium each)
- String concatenation in queries (critical each)
- Missing meaningful aliases (small each)
- No indexing consideration (medium each)"""

    # Corrected code rules depend on line count
    if line_count <= 50:
        corrected_code_rules = """CORRECTED CODE RULES (Snippet / Short — <= 50 lines):
- Full rewrite at senior level
- Inline comments on every change
- Comments in Jason Wilder's voice
- No emojis"""
    elif line_count <= 150:
        corrected_code_rules = """CORRECTED CODE RULES (Medium — 51-150 lines):
- Full rewrite at senior level
- Inline comments on major changes only
- Comments in Jason Wilder's voice
- No emojis"""
    elif line_count <= 300:
        corrected_code_rules = """CORRECTED CODE RULES (Long — 151-300 lines):
- Partial rewrite of the 3 worst sections only
- Start with: "Showing rewrite of the 3 most problematic sections. Jason Wilder expects you to apply these patterns to the rest of the file."
- Inline comments in his voice
- No emojis"""
    else:
        corrected_code_rules = """CORRECTED CODE RULES (Very Long — 300+ lines):
- No rewrite
- Improvement roadmap instead. Start with: "This file is too long for a full rewrite. Here is what Jason Wilder would put on your code review:"
- List all issues in priority order: CRITICAL first with line numbers if possible, BIG second, MEDIUM third, SMALL last
- End with: "Fix these in order. Jason Wilder is watching."
- No emojis"""

    return f"""You are a code reviewer channeling the spirit of Jason Wilder, a strict but brilliant BCIT professor known for his exacting coding standards. Analyze the submitted code and return ONLY a valid JSON object with no extra text, no markdown, no backticks, just raw JSON.

The code has been pre-analyzed and determined to be: {size_bucket}
You must use this exact string for code_size. Do not count lines yourself or invent a different size.

The JSON must contain exactly these fields:
{{
  "roast": "string - funny sarcastic analysis in the voice of a strict professor. No emojis. Start with: Detected: {size_bucket}. Then continue your analysis.",
  "code_size": "{size_bucket}",
  "detected_language": "string - use full proper names only: Python, JavaScript, TypeScript, Java, C++, C#, SQL, C, or other if truly unknown",
  "mistakes": [
    {{
      "issue": "string - clear description of the specific mistake found",
      "severity": "string - must be exactly one of: small, medium, big, critical"
    }}
  ],
  "corrected_code": "string - see rules below"
}}

If no mistakes are found return: "mistakes": []
This means the code is perfect and scores 100. Do not invent mistakes just to fill the list. Only report real issues you actually found.

MISTAKES TO LOOK FOR IN PRIORITY ORDER:

CRITICAL — flag every one found:
1. SQL injection — string concatenation in queries
2. Hardcoded credentials or API keys in code
3. No input sanitization on user data
4. Catching all exceptions and ignoring silently
5. Memory management issues — leaks in C or C++
6. Storing sensitive data insecurely

BIG — flag every one found:
7.  No error handling at all
8.  Entire logic crammed into one function
9.  No separation of concerns
10. Global variables used instead of parameters
11. Infinite loop risks with no exit condition
12. No type hints or type safety where expected

MEDIUM — flag every one found:
13. Function does more than one thing
14. No input validation
15. Missing edge case handling — null, empty, zero division
16. Repeated code that should be its own function
17. Wrong data types used
18. Hardcoded values that should be configurable

SMALL — flag every one found:
19. Bad variable names — x, temp, data, single letters
20. Magic numbers instead of named constants
21. Missing docstrings or comments
22. Debug print statements left in code
23. Abbreviations — addr, qty, usr, wt
24. Inconsistent formatting and spacing
25. Unclear or misleading function names
26. Missing blank lines between logical sections
{language_specific}

COMPLETENESS NOTE:
If snippet or short code is missing things a complete senior implementation would have, add encouragingly at end of roast field only: "This looks like a snippet. Jason Wilder would expect a complete implementation to also include error handling, type hints, and proper documentation."
Never add to mistakes array for things only expected in larger files.

INTENSITY TONE — affects roast text only:
{intensity_tone}

{corrected_code_rules}

FORMATTING RULES:
- Never use emojis anywhere
- Plain text only
- No markdown inside JSON string values
- Return raw JSON only, nothing else"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index() -> Any:
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    data = request.get_json(silent=True) or {}

    code: str = data.get("code", "")
    language: str = data.get("language", "auto")
    intensity: str = data.get("intensity", "medium")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if not code or not code.strip():
        return jsonify({"error": "Code cannot be empty"}), 400

    if language not in VALID_LANGUAGES:
        return jsonify({"error": "Invalid language selection"}), 400

    if intensity not in VALID_INTENSITIES:
        return jsonify({"error": "Invalid intensity selection"}), 400

    if len(code) > MAX_CODE_LENGTH:
        return jsonify({"error": "Code exceeds maximum length of 10000 characters"}), 400

    # ------------------------------------------------------------------
    # Code size detection (Python-side, before AI call)
    # ------------------------------------------------------------------
    line_count = count_non_empty_lines(code)
    size_bucket = get_size_bucket(line_count)
    size_key = get_size_key(size_bucket)

    # ------------------------------------------------------------------
    # Build system prompt and call Groq
    # ------------------------------------------------------------------
    system_prompt = build_system_prompt(intensity, language, size_bucket, line_count)

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            seed=SEED,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code},
            ],
        )
        raw_content = completion.choices[0].message.content
    except Exception:
        return jsonify({"error": "Analysis failed. Please try again."}), 500

    # ------------------------------------------------------------------
    # Parse JSON response
    # ------------------------------------------------------------------
    try:
        ai_response = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return jsonify({"error": "Failed to parse response. Please try again."}), 500

    # ------------------------------------------------------------------
    # Verify required fields
    # ------------------------------------------------------------------
    required_fields = {"roast", "code_size", "detected_language", "mistakes", "corrected_code"}
    if not required_fields.issubset(ai_response.keys()):
        return jsonify({"error": "Incomplete response. Please try again."}), 500

    # ------------------------------------------------------------------
    # Override code_size with Python-computed value
    # ------------------------------------------------------------------
    ai_response["code_size"] = size_bucket

    # ------------------------------------------------------------------
    # Validate mistakes
    # ------------------------------------------------------------------
    raw_mistakes = ai_response.get("mistakes")

    if not isinstance(raw_mistakes, list):
        # Case 1 — raw_mistakes is not a list at all: AI response is malformed
        clean_mistakes = get_fallback_mistakes()
    else:
        # Case 2 — raw_mistakes is a list
        clean_mistakes = validate_and_clean_mistakes(raw_mistakes)
        # Case 2a — had items but ALL were invalid after cleaning
        if len(raw_mistakes) > 0 and len(clean_mistakes) == 0:
            clean_mistakes = get_fallback_mistakes()
        # Case 2b — genuinely empty: perfect code, clean_mistakes stays []

    # ------------------------------------------------------------------
    # Language resolution
    # ------------------------------------------------------------------
    ai_detected: str = ai_response.get("detected_language", "")
    resolved_language, is_approximate = resolve_language(language, ai_detected)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    score, deductions = calculate_score(clean_mistakes, size_key)
    verdict = get_verdict(score)

    # ------------------------------------------------------------------
    # Corrected code truncation
    # ------------------------------------------------------------------
    corrected_code = truncate_corrected_code(ai_response.get("corrected_code", ""))

    # ------------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------------
    return jsonify({
        "roast":               ai_response.get("roast", ""),
        "score":               score,
        "verdict":             verdict,
        "code_size":           size_bucket,
        "detected_language":   resolved_language,
        "language_approximate": is_approximate,
        "deductions":          deductions,
        "corrected_code":      corrected_code,
    })


if __name__ == "__main__":
    app.run(debug=True)
