#!/usr/bin/env python3
"""Build TensorTrust subsets from the raw defense corpus.

This script reads the original TensorTrust defense dataset and emits a set of
filtered JSONL subsets under ``data/TensorTrust_subsets``:

1. access code < 30
2. access code < 30 and all alphabets
3. access code < 30 and all alphabets and directly recoverable
4. access code < 30 and all alphabets and deterministically recoverable
5. access code < 30 and all alphabets and indirectly recoverable
6. access code < 30 and all alphabets and not recoverable
7. access code < 30 and all alphabets and directly recoverable or deterministically recoverable
8. access code < 30 and all alphabets and directly recoverable or deterministically recoverable or indirectly recoverable
9. access code < 30 and all alphabets and deterministically recoverable or indirectly recoverable

The recoverability labels are heuristic and conservative. The script also writes
``manifest.json`` with counts and classification notes.
"""

from __future__ import annotations

import base64
import argparse
import codecs
import bz2
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    ROOT / "experiment" / "raw_dump_defenses.jsonl.bz2",
]
OUTPUT_DIR = ROOT / "data" / "TensorTrust_subsets"
MAX_ACCESS_CODE_LENGTH = 30


@dataclass(frozen=True)
class Recoverability:
    direct: bool
    deterministic: bool
    indirect: bool
    reasons: tuple[str, ...]


def exclusive_bucket_name(recoverability: Recoverability) -> str | None:
    """Return the most specific single-label bucket for a row.

    This is used only for the mutually exclusive 3-6 buckets.
    """

    direct = recoverability.direct
    deterministic = recoverability.deterministic
    indirect = recoverability.indirect

    if direct and not deterministic and not indirect:
        return "subset_3_ac30_all_alpha_direct"
    if deterministic and not direct and not indirect:
        return "subset_4_ac30_all_alpha_deterministic"
    if indirect and not direct and not deterministic:
        return "subset_5_ac30_all_alpha_indirect"
    if not direct and not deterministic and not indirect:
        return "subset_6_ac30_all_alpha_not_recoverable"
    return None


def union_buckets(recoverability: Recoverability) -> list[str]:
    """Return all overlapping union buckets a row belongs to."""

    buckets: list[str] = []
    direct = recoverability.direct
    deterministic = recoverability.deterministic
    indirect = recoverability.indirect

    if direct or deterministic:
        buckets.append("subset_7_ac30_all_alpha_direct_or_deterministic")
    if direct or deterministic or indirect:
        buckets.append("subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect")
    if deterministic or indirect:
        buckets.append("subset_9_ac30_all_alpha_deterministic_or_indirect")
    return buckets


def load_rows(paths: Iterable[Path]) -> Iterator[tuple[dict, str]]:
    for path in paths:
        if not path.exists():
            continue
        opener = bz2.open if path.suffix == ".bz2" else open
        with opener(path, "rt", encoding="utf-8") as fp:
            for _, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row, path.name


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def alpha_fold(text: str) -> str:
    return re.sub(r"[^a-z]", "", strip_accents(text).casefold())


def surface_text(row: dict) -> str:
    return "\n".join([
        str(row.get("opening_defense") or ""),
        str(row.get("closing_defense") or ""),
    ])


def is_short_access_code(code: str) -> bool:
    return len(str(code or "")) < MAX_ACCESS_CODE_LENGTH


def is_all_alpha_access_code(code: str) -> bool:
    code = str(code or "")
    code = re.sub(r"\s+", "", code)
    return bool(code) and code.isalpha()


def direct_recoverable(access_code: str, surface: str) -> bool:
    code_folded = alpha_fold(access_code)
    surface_folded = alpha_fold(surface)
    if not code_folded or not surface_folded:
        return False
    return code_folded in surface_folded


def _search_candidates(surface: str, candidates: Iterable[str]) -> tuple[bool, str]:
    raw = surface
    folded = surface.casefold()
    folded_no_space = re.sub(r"\s+", "", folded)
    for candidate in candidates:
        if not candidate:
            continue
        cand = str(candidate)
        cand_fold = cand.casefold()
        cand_no_space = re.sub(r"\s+", "", cand_fold)
        if cand in raw:
            return True, cand
        if cand_fold in folded:
            return True, cand
        if cand_no_space and cand_no_space in folded_no_space:
            return True, cand
    return False, ""


def extract_code_blocks(surface: str) -> list[str]:
    blocks = []
    for match in re.finditer(r"```(?:([a-zA-Z0-9_+-]+))?\s*([\s\S]*?)```", surface, re.IGNORECASE):
        blocks.append(match.group(2).strip())
    return blocks


def _eval_python_string_expr(node: Any, env: dict[str, str]) -> str | None:
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_python_string_expr(node.left, env)
        right = _eval_python_string_expr(node.right, env)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                inner = _eval_python_string_expr(value.value, env)
                if inner is None:
                    return None
                pieces.append(inner)
            else:
                return None
        return "".join(pieces)
    if isinstance(node, ast.Subscript):
        base = _eval_python_string_expr(node.value, env)
        if base is None:
            return None
        slc = node.slice
        if isinstance(slc, ast.Slice):
            start = _eval_python_int_expr(slc.lower, env) if slc.lower else None
            stop = _eval_python_int_expr(slc.upper, env) if slc.upper else None
            step = _eval_python_int_expr(slc.step, env) if slc.step else None
            return base[slice(start, stop, step)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if attr == "join" and len(node.args) == 1:
            sep = _eval_python_string_expr(node.func.value, env)
            values = _eval_python_list_expr(node.args[0], env)
            if sep is not None and values is not None and all(v is not None for v in values):
                return sep.join(values)  # type: ignore[arg-type]
        if attr in {"strip", "lower", "upper", "casefold", "replace"}:
            base = _eval_python_string_expr(node.func.value, env)
            if base is None:
                return None
            if attr == "strip":
                return base.strip()
            if attr == "lower":
                return base.lower()
            if attr == "upper":
                return base.upper()
            if attr == "casefold":
                return base.casefold()
            if attr == "replace" and len(node.args) >= 2:
                old = _eval_python_string_expr(node.args[0], env)
                new = _eval_python_string_expr(node.args[1], env)
                if old is None or new is None:
                    return None
                return base.replace(old, new)
    return None


def _eval_python_int_expr(node: Any, env: dict[str, str]) -> int | None:
    import ast

    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_python_int_expr(node.operand, env)
        return -value if value is not None else None
    return None


def _eval_python_list_expr(node: Any, env: dict[str, str]) -> list[str | None] | None:
    import ast

    if isinstance(node, ast.List):
        return [_eval_python_string_expr(elt, env) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return [_eval_python_string_expr(elt, env) for elt in node.elts]
    return None


def _eval_shell_string_expr(expr: str, env: dict[str, str]) -> str | None:
    expr = expr.strip()
    if not expr:
        return ""
    if expr.startswith('"') and expr.endswith('"'):
        expr = expr[1:-1]
        expr = re.sub(r"\\([\\\"'nrt])", lambda m: {"n": "\n", "r": "\r", "t": "\t"}.get(m.group(1), m.group(1)), expr)
        return expr
    if expr.startswith("'") and expr.endswith("'"):
        return expr[1:-1]
    if expr.startswith("$(") and expr.endswith(")"):
        inner = expr[2:-1]
        return _eval_shell_string_expr(inner, env)
    if expr.startswith("$"):
        return env.get(expr[1:])
    parts = re.split(r"(\$\w+)", expr)
    if len(parts) > 1:
        built = []
        for part in parts:
            if part.startswith("$") and len(part) > 1:
                built.append(env.get(part[1:], ""))
            else:
                built.append(part)
        return "".join(built)
    return expr


def code_block_shell_recoverable(access_code: str, surface: str) -> tuple[bool, tuple[str, ...]]:
    code_target = alpha_fold(access_code)
    hits = []
    for block in extract_code_blocks(surface):
        if not re.search(r"\b(?:sh|bash|shell)\b|^\s*(?:#!/bin/(?:ba)?sh|echo|printf|export|set)\b", block, re.IGNORECASE | re.MULTILINE):
            continue
        env: dict[str, str] = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            assign = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
            if assign:
                name, expr = assign.groups()
                value = _eval_shell_string_expr(expr, env)
                if value is not None:
                    env[name] = value
                continue
            if line.startswith("export "):
                assign = re.match(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
                if assign:
                    name, expr = assign.groups()
                    value = _eval_shell_string_expr(expr, env)
                    if value is not None:
                        env[name] = value
                continue
            echo = re.match(r"^(?:echo|printf)\s+(.*)$", line)
            if echo:
                expr = echo.group(1)
                value = _eval_shell_string_expr(expr, env)
                if value is not None and alpha_fold(value) == code_target:
                    hits.append("code_output:shell_echo")
                    break
        if hits:
            break
    return bool(hits), tuple(hits)


def code_block_structured_output_recoverable(access_code: str, surface: str) -> tuple[bool, tuple[str, ...]]:
    code_target = alpha_fold(access_code)
    hits = []
    for block in extract_code_blocks(surface):
        compact = re.sub(r"\s+", "", block)
        if not compact:
            continue

        # JSON-style output
        jsonish = re.search(r'"(?:answer|password|code|token|secret)"\s*:\s*"(.*?)"', block, re.IGNORECASE | re.DOTALL)
        if jsonish and alpha_fold(jsonish.group(1)) == code_target:
            hits.append("code_output:json_value")
            break

        # XML-style output
        xmlish = re.search(r"<(?:answer|password|code|token|secret)>(.*?)</(?:answer|password|code|token|secret)>", block, re.IGNORECASE | re.DOTALL)
        if xmlish and alpha_fold(xmlish.group(1)) == code_target:
            hits.append("code_output:xml_value")
            break

        # Pseudo-code / step lists
        if re.search(r"\b(?:step|steps|then|output|print|return|derive|assemble|concatenate|combine)\b", block, re.IGNORECASE):
            assignments = re.findall(r"(?:set|assign|let|make)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:to|=)\s+([^\n;]+)", block, re.IGNORECASE)
            env: dict[str, str] = {}
            for name, expr in assignments:
                value = _eval_shell_string_expr(expr, env) or _eval_python_string_expr(_safe_parse_expr(expr), env)
                if value is not None:
                    env[name] = value
            out = re.search(r"(?:output|print|return)\s+([A-Za-z_][A-Za-z0-9_]*|\".*?\"|'.*?')", block, re.IGNORECASE)
            if out:
                expr = out.group(1)
                value = env.get(expr, None) or _eval_shell_string_expr(expr, env)
                if value is not None and alpha_fold(value) == code_target:
                    hits.append("code_output:pseudocode")
                    break
    return bool(hits), tuple(hits)


def _safe_parse_expr(expr: str) -> Any:
    import ast

    try:
        return ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None


def code_block_deterministic_recoverable(access_code: str, surface: str) -> tuple[bool, tuple[str, ...]]:
    import ast

    code_target = alpha_fold(access_code)
    hits = []
    for block in extract_code_blocks(surface):
        if len(block) > 5000:
            continue
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue

        env: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                value = _eval_python_string_expr(node.value, env)
                if value is not None:
                    env[node.targets[0].id] = value
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == "print" and call.args:
                    value = _eval_python_string_expr(call.args[0], env)
                    if value is not None and alpha_fold(value) == code_target:
                        hits.append("code_output:python_print")
                elif isinstance(call.func, ast.Attribute) and call.func.attr in {"join", "strip", "lower", "upper", "casefold", "replace"}:
                    value = _eval_python_string_expr(call, env)
                    if value is not None and alpha_fold(value) == code_target:
                        hits.append(f"code_output:python_{call.func.attr}")

        if hits:
            break

    if not hits:
        shell_hits, shell_reasons = code_block_shell_recoverable(access_code, surface)
        if shell_hits:
            hits.extend(shell_reasons)
    if not hits:
        structured_hits, structured_reasons = code_block_structured_output_recoverable(access_code, surface)
        if structured_hits:
            hits.extend(structured_reasons)

    return bool(hits), tuple(hits)


def deterministic_recoverable(access_code: str, surface: str) -> tuple[bool, tuple[str, ...]]:
    code = alpha_fold(access_code)
    if not code:
        return False, ()

    candidates = []
    candidates.append(code[::-1])
    candidates.append(codecs.decode(code, "rot_13"))
    candidates.append(base64.b64encode(code.encode("utf-8")).decode("ascii"))
    candidates.append(code.encode("utf-8").hex())

    matches = []
    found, candidate = _search_candidates(surface, candidates)
    if found and candidate:
        matches.append(candidate)

    code_hits, code_reasons = code_block_deterministic_recoverable(access_code, surface)
    if code_hits:
        matches.extend(code_reasons)

    cue_words = [
        "base64",
        "rot13",
        "hex",
        "reverse",
        "encode",
        "decode",
        "unicode",
        "transformed",
        "cipher",
        "python",
        "print(",
        "return ",
        "def ",
    ]
    if not matches:
        if any(word in surface.casefold() for word in cue_words):
            # Cues alone are not enough; require a concrete transformed candidate.
            pass

    return bool(matches), tuple(matches)


INDIRECT_PATTERNS = [
    r"\[secrets?\]",
    r"\[input\]",
    r"\[user input\]",
    r"\$securitykey",
    r"\$input",
    r"\bsecret(?:s| key| phrase| token| code)?\b",
    r"\bpassword\b",
    r"\baccess code\b",
    r"\bcredential\b",
    r"\bthe password is\b",
    r"\bif .*? is exactly\b",
    r"\bcorrect password\b",
    r"\bprevious(?:ly)?\b",
    r"\bearlier\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bhidden\b",
    r"\bverbatim\b",
    r"\brepeat\b",
    r"\bdecode\b",
    r"\btranslate\b",
]


def indirect_recoverable(surface: str) -> tuple[bool, tuple[str, ...]]:
    hits = []
    folded = surface.casefold()
    for pattern in INDIRECT_PATTERNS:
        if re.search(pattern, folded, re.IGNORECASE | re.DOTALL):
            hits.append(pattern)
    return bool(hits), tuple(hits)


def classify_row(row: dict) -> Recoverability:
    access_code = str(row.get("access_code") or "")
    surface = surface_text(row)

    direct = direct_recoverable(access_code, surface)
    deterministic, det_reasons = deterministic_recoverable(access_code, surface)
    indirect, ind_reasons = indirect_recoverable(surface)

    reasons = []
    if direct:
        reasons.append("direct:normalized_surface")
    if deterministic:
        reasons.extend(f"deterministic:{reason}" for reason in det_reasons)
    if indirect:
        reasons.extend(f"indirect:{reason}" for reason in ind_reasons[:5])

    return Recoverability(
        direct=direct,
        deterministic=deterministic,
        indirect=indirect,
        reasons=tuple(reasons),
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TensorTrust subset datasets")
    parser.add_argument(
        "--input",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Input TensorTrust JSONL/JSONL.BZ2 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for subset files",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("subset_*_ac30*.jsonl"):
        stale_path.unlink()

    output_paths = {
        "subset_1_ac30": output_dir / "subset_1_ac30.jsonl",
        "subset_2_ac30_all_alpha": output_dir / "subset_2_ac30_all_alpha.jsonl",
        "subset_3_ac30_all_alpha_direct": output_dir / "subset_3_ac30_all_alpha_direct.jsonl",
        "subset_4_ac30_all_alpha_deterministic": output_dir / "subset_4_ac30_all_alpha_deterministic.jsonl",
        "subset_5_ac30_all_alpha_indirect": output_dir / "subset_5_ac30_all_alpha_indirect.jsonl",
        "subset_6_ac30_all_alpha_not_recoverable": output_dir / "subset_6_ac30_all_alpha_not_recoverable.jsonl",
        "subset_7_ac30_all_alpha_direct_or_deterministic": output_dir / "subset_7_ac30_all_alpha_direct_or_deterministic.jsonl",
        "subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect": output_dir / "subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl",
        "subset_9_ac30_all_alpha_deterministic_or_indirect": output_dir / "subset_9_ac30_all_alpha_deterministic_or_indirect.jsonl",
    }

    bucket_rows: dict[str, list[dict]] = {name: [] for name in output_paths}
    stats = {
        "total_rows": 0,
        "ac30_rows": 0,
        "all_alpha_rows": 0,
        "direct_rows": 0,
        "deterministic_rows": 0,
        "indirect_rows": 0,
        "not_recoverable_rows": 0,
        "excluded_rows": 0,
    }
    recoverability_reasons: dict[str, int] = {}
    excluded_reasons: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for row, source_file in load_rows(args.input):
        stats["total_rows"] += 1
        source_counts[source_file] = source_counts.get(source_file, 0) + 1

        access_code = str(row.get("access_code") or "")
        if not is_short_access_code(access_code):
            continue

        stats["ac30_rows"] += 1
        bucket_rows["subset_1_ac30"].append(row)

        if not is_all_alpha_access_code(access_code):
            continue

        stats["all_alpha_rows"] += 1
        bucket_rows["subset_2_ac30_all_alpha"].append(row)

        recoverability = classify_row(row)
        for reason in recoverability.reasons:
            recoverability_reasons[reason] = recoverability_reasons.get(reason, 0) + 1

        if recoverability.direct:
            stats["direct_rows"] += 1
        if recoverability.deterministic:
            stats["deterministic_rows"] += 1
        if recoverability.indirect:
            stats["indirect_rows"] += 1
        if not (recoverability.direct or recoverability.deterministic or recoverability.indirect):
            stats["not_recoverable_rows"] += 1

        bucket_name = exclusive_bucket_name(recoverability)
        if bucket_name is not None:
            bucket_rows[bucket_name].append(row)

        union_names = union_buckets(recoverability)
        for name in union_names:
            bucket_rows[name].append(row)

        if not union_names and bucket_name is None:
            stats["excluded_rows"] += 1
            parts = [
                label
                for label, flag in (
                    ("direct", recoverability.direct),
                    ("deterministic", recoverability.deterministic),
                    ("indirect", recoverability.indirect),
                )
                if flag
            ]
            key = "+".join(parts) if parts else "unknown"
            excluded_reasons[key] = excluded_reasons.get(key, 0) + 1

    counts = {}
    for name, path in output_paths.items():
        counts[name] = write_jsonl(path, bucket_rows[name])

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files": [str(path) for path in args.input],
        "output_dir": str(output_dir),
        "rules": {
            "access_code_length_lt": MAX_ACCESS_CODE_LENGTH,
            "all_alphabetic": True,
            "direct": "access code appears in the visible defense text after normalization",
            "deterministic": "a reversible transform of the access code appears in the visible defense text",
            "indirect": "the visible defense text contains structural or referential clues that point to the hidden secret",
            "subset_7": "inclusive union of rows that are direct or deterministic recoverable",
            "subset_8": "inclusive union of rows that are direct or deterministic or indirect recoverable",
            "subset_9": "inclusive union of rows that are deterministic or indirect recoverable",
        },
        "subset_mode": {
            "subset_1_to_6": "exclusive",
            "subset_7_to_9": "inclusive_union",
        },
        "subset_counts": counts,
        "input_stats": stats,
        "recoverability_signal_counts": recoverability_reasons,
        "excluded_reason_counts": excluded_reasons,
        "source_counts": source_counts,
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    readme = output_dir / "README.md"
    readme.write_text(
        "# TensorTrust Subsets\n\n"
        "This directory was generated from `data/defense_classifier_dataset-Part1.jsonl` "
        "and `data/defense_classifier_dataset-Part2.jsonl`.\n\n"
        "Recoverability labels are heuristic and conservative:\n"
        "- direct: code is visible in the prompt surface after normalization\n"
        "- deterministic: a reversible transform of the code is visible\n"
        "- indirect: the prompt contains structural or referential clues to the secret\n"
        "- not recoverable: none of the above\n"
        "\nRows that match a combination not named in the 1-9 taxonomy are excluded\n"
        "from the subset files and recorded in manifest.json.\n\n"
        "Subset semantics:\n"
        "- subset_1 and subset_2 remain broad filters.\n"
        "- subset_3 through subset_6 remain exclusive single-label buckets.\n"
        "- subset_7 through subset_9 are inclusive union buckets and can overlap.\n",
        encoding="utf-8",
    )

    print("TensorTrust subset build complete")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
