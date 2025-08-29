#!/usr/bin/env python3
"""
r1_data_processor.py

Lightweight dataset skimmer for R1-style JSONL datasets.

Features:
- Stream-reads JSONL to avoid loading large files
- Extracts assistant message content and looks for <think>...</think> and <answer>...</answer>
- Performs basic validations and heuristics to flag malformed or suspicious (poison) records
- Writes a JSON report and prints a short summary

Usage:
  python3 r1_data_processor.py --file deepseek_classified_math_science.jsonl --sample 200

This module is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple


RE_THINK = re.compile(r"<think>(.*?)</think>", re.I | re.S)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
RE_IGNORES = re.compile(r"ignore previous|ignore.*instruction|disregard previous", re.I)
RE_URL = re.compile(r"https?://|www\.", re.I)


@dataclass
class RecordIssue:
    line_no: int
    score: float
    issues: List[str]
    excerpt: str
    raw: Dict[str, Any]


def extract_assistant_text(record: Dict[str, Any]) -> List[str]:
    """Return a list of assistant message contents from common dataset schemas.

    Supports these patterns (heuristic):
    - record['messages'] is a list of {'role':..., 'content': ...}
    - record['assistant'] or record['completion'] or record['response'] string fields
    - record.get('answer') or record.get('output')
    """
    texts: List[str] = []
    if isinstance(record.get("messages"), list):
        for m in record.get("messages", []):
            try:
                role = (m.get("role") or "").lower()
            except Exception:
                role = ""
            if role in ("assistant", "system", "final"):
                c = m.get("content") or m.get("text")
                if isinstance(c, str):
                    texts.append(c)
    # common single-field completions
    for k in ("assistant", "completion", "response", "output", "answer_content"):
        v = record.get(k)
        if isinstance(v, str):
            texts.append(v)
    # fallback: flatten any string values
    if not texts:
        for k, v in record.items():
            if isinstance(v, str) and len(v) > 30:
                texts.append(v)
    return texts


def find_think_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    think_m = RE_THINK.search(text)
    ans_m = RE_ANSWER.search(text)
    think = think_m.group(1).strip() if think_m else None
    answer = ans_m.group(1).strip() if ans_m else None
    return think, answer


def fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_record(record: Dict[str, Any], line_no: int) -> Optional[RecordIssue]:
    """Run a set of heuristics and return a RecordIssue if suspicious/malformed.

    Scoring heuristic returns a positive score for suspicious records.
    """
    texts = extract_assistant_text(record)
    issues: List[str] = []
    score = 0.0
    excerpt = ""

    # If there are no assistant texts at all, it's malformed
    if not texts:
        issues.append("no_assistant_text")
        score += 3.0
        return RecordIssue(line_no, score, issues, excerpt, record)

    # analyze each assistant chunk and keep the worst
    for txt in texts:
        t, a = find_think_answer(txt)
        excerpt = (txt[:300].replace('\n', '\\n'))
        # missing tags
        if t is None and a is None:
            issues.append("missing_think_and_answer_tags")
            score += 2.0
        elif t is None:
            issues.append("missing_think_tag")
            score += 1.0
        elif a is None:
            issues.append("missing_answer_tag")
            score += 1.5
        else:
            # both present: check order and emptiness
            if txt.find("<answer>") < txt.find("<think>"):
                issues.append("answer_before_think")
                score += 2.0
            if len(t) < 10:
                issues.append("think_too_short")
                score += 0.8
            if len(a) < 1:
                issues.append("empty_answer")
                score += 2.0

            # numeric vs textual checks
            if a.isdigit() and len(a) > 50:
                issues.append("suspicious_long_digits_in_answer")
                score += 1.0

            # repetition / garbage
            if len(set(a.split())) <= 2 and len(a.split()) > 0:
                issues.append("answer_repetitive")
                score += 1.0

            # unsafe content (prompt injections)
            if RE_IGNORES.search(t or "") or RE_IGNORES.search(a or ""):
                issues.append("prompt_injection_ignore_instructions")
                score += 4.0

            if RE_URL.search(t or "") or RE_URL.search(a or ""):
                issues.append("external_url_in_think_or_answer")
                score += 1.0

        # compare with reference answer when available
        ref = None
        for meta_key in ("reference_answer", "reference", "answer", "label"):
            if meta_key in record and isinstance(record[meta_key], str):
                ref = record[meta_key]
                break
        if ref and a:
            r = fuzzy_ratio(ref.strip().lower(), a.strip().lower())
            # if very different but both short answers, suspicious
            if r < 0.25:
                issues.append("answer_mismatch_with_reference")
                score += 1.2 * (1.0 - r)

    if not issues:
        return None

    return RecordIssue(line_no=line_no, score=score, issues=issues, excerpt=excerpt, raw=record)


def skim_jsonl(path: str, max_lines: Optional[int] = None) -> Dict[str, Any]:
    """Stream the JSONL file and collect a short report of suspicious records."""
    totals = Counter()
    suspicious: List[RecordIssue] = []
    line_no = 0

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line_no += 1
            if max_lines and line_no > max_lines:
                break
            raw_strip = raw.strip()
            if not raw_strip:
                totals['empty_lines'] += 1
                continue
            try:
                rec = json.loads(raw_strip)
            except json.JSONDecodeError:
                totals['bad_json'] += 1
                suspicious.append(RecordIssue(line_no, 10.0, ["json_decode_error"], raw_strip[:200], {"raw": raw_strip}))
                continue

            # quick structural checks
            if not isinstance(rec, dict):
                totals['not_object'] += 1
                suspicious.append(RecordIssue(line_no, 5.0, ["record_not_object"], str(rec)[:200], {"raw": rec}))
                continue

            # run heuristics
            issue = score_record(rec, line_no)
            if issue:
                suspicious.append(issue)
                for it in issue.issues:
                    totals[f'issue:{it}'] += 1
            else:
                totals['ok'] += 1

    suspicious.sort(key=lambda r: r.score, reverse=True)
    report = {
        'file': os.path.abspath(path),
        'lines_scanned': line_no,
        'summary': dict(totals),
        'suspicious_count': len(suspicious),
        'top_suspicious': [asdict(r) for r in suspicious[:50]],
    }
    return report


def write_report(report: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="r1_data_processor.py", description="Scan JSONL for malformed R1 records and poison candidates")
    p.add_argument("--file", "-f", required=True, help="Path to JSONL dataset")
    p.add_argument("--sample", "-n", type=int, default=500, help="Max lines to scan (default: 500)")
    p.add_argument("--out", "-o", default="data_quality_report.json", help="Report output path")
    p.add_argument("--show-top", "-t", type=int, default=10, help="Number of top suspicious examples to print")
    args = p.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 2

    print(f"Scanning {args.file} (up to {args.sample} lines)...")
    report = skim_jsonl(args.file, max_lines=args.sample)
    write_report(report, args.out)

    print("--- scan summary ---")
    print(f"file: {report['file']}")
    print(f"lines scanned: {report['lines_scanned']}")
    for k, v in sorted(report['summary'].items(), key=lambda kv: -kv[1] if isinstance(kv[1], int) else 0):
        print(f"  {k}: {v}")
    print(f"suspicious_count: {report['suspicious_count']}")
    print(f"wrote report -> {os.path.abspath(args.out)}")

    top = report.get('top_suspicious', [])[: args.show_top]
    if top:
        print('\nTop suspicious examples:')
        for r in top:
            print(f"- line {r['line_no']} score={r['score']:.2f} issues={r['issues']}")
            print(f"  excerpt: {r['excerpt'][:200]}")
    else:
        print('No suspicious records found in sample.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
