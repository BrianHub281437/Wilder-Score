import os
import re
import json
import sys
import argparse
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

from app import (
    count_non_empty_lines,
    get_size_bucket,
    get_size_key,
    build_system_prompt,
    validate_and_clean_mistakes,
    enforce_java_special_cases,
    filter_roast_for_super,
    try_add_line_number,
    calculate_score,
    get_verdict,
    resolve_language,
    client,
    MODEL,
    TEMPERATURE,
    SEED,
)


def main():
    parser = argparse.ArgumentParser(
        description="Am I Wilder Yet? — CLI Code Reviewer"
    )
    parser.add_argument(
        "file",
        help="Path to the code file to review"
    )
    parser.add_argument(
        "--intensity",
        choices=["gentle", "medium", "savage"],
        default="medium",
        help="Review intensity (default: medium)"
    )
    parser.add_argument(
        "--language",
        choices=["auto", "java", "python",
                 "javascript", "typescript",
                 "cpp", "csharp", "sql"],
        default="auto",
        help="Language (default: auto)"
    )
    args = parser.parse_args()

    try:
        with open(args.file, "r") as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    intensity = args.intensity
    language = args.language

    print(f"\nAnalyzing {args.file}...")
    print(f"Intensity: {intensity.capitalize()}")

    line_count = count_non_empty_lines(code)
    size_bucket = get_size_bucket(line_count)
    size_key = get_size_key(size_bucket)

    system_prompt = build_system_prompt(
        intensity, language, size_bucket, line_count, code
    )

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
        raw = completion.choices[0].message.content
    except Exception as e:
        print(f"Error: API call failed — {e}")
        sys.exit(1)

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()
        ai_response = json.loads(cleaned)
    except Exception:
        print("Error: Failed to parse response. Please try again.")
        sys.exit(1)

    raw_mistakes = ai_response.get("mistakes", [])
    clean_mistakes = validate_and_clean_mistakes(raw_mistakes)
    clean_mistakes = [
        {
            "issue": try_add_line_number(code, m["issue"]),
            "severity": m["severity"]
        }
        for m in clean_mistakes
    ]

    ai_detected = ai_response.get("detected_language", "")
    resolved_language, _ = resolve_language(
        language, ai_detected
    )
    clean_mistakes = enforce_java_special_cases(
        clean_mistakes, language, code, resolved_language
    )

    score, deductions = calculate_score(clean_mistakes, size_key)
    verdict = get_verdict(score)
    roast = filter_roast_for_super(
        ai_response.get("roast", ""),
        language, code, resolved_language
    )

    print(f"\n{'=' * 50}")
    print(f"  {resolved_language} — {size_bucket}")
    print(f"{'=' * 50}")

    print(f"\nWILDER'S VERDICT")
    print(f"{'-' * 50}")
    print(roast)

    print(f"\nWILDER SCORE: {score}/100 — {verdict}")
    print(f"{'-' * 50}")

    if deductions:
        print("\nWHAT WILDER FOUND:")
        for d in deductions:
            print(f"  - {d['text']}")
        total = sum(d["points"] for d in deductions)
        fmt_total = int(total) if total == int(total) else total
        print(f"\n  Total deducted: {fmt_total} points")
    else:
        print("\nWilder found nothing wrong. Suspicious.")

    print(f"\n{'=' * 50}\n")


if __name__ == "__main__":
    main()
