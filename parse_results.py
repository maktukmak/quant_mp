import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Tasks and human-friendly short names for table headers
TASKS: List[str] = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "piqa",
    "social_iqa",
    "hellaswag",
    "openbookqa",
    "winogrande",
]
TASK_SHORT: Dict[str, str] = {
    "arc_easy": "arc_e",
    "arc_challenge": "arc_c",
    "boolq": "boolq",
    "piqa": "piqa",
    "social_iqa": "siqa",
    "hellaswag": "hella",
    "openbookqa": "obqa",
    "winogrande": "wino",
}

# Standardized sort keys
NUMERIC_SORT_KEYS = {
    "arc_e",
    "arc_c",
    "boolq",
    "piqa",
    "siqa",
    "hella",
    "obqa",
    "wino",
    "avg",
    "wiki2",
}
STRING_SORT_KEYS = {
    "model",
    "label",
    "w_dtype",
    "w_block",
    "w_alg",
    "a_dtype",
    "a_alg",
    "a_block",
}
CANONICAL_SORT_KEYS = STRING_SORT_KEYS | NUMERIC_SORT_KEYS

SORT_ALIASES: Dict[str, str] = {
    # Weights
    "dtype": "w_dtype",
    "format": "w_dtype",
    "w_format": "w_dtype",
    "block": "w_block",
    "block_size": "w_block",
    "alg": "w_alg",
    # Back-compat for old keys
    "w_dformat": "w_dtype",
    # Activations
    "act_dtype": "a_dtype",
    "act_format": "a_dtype",
    "act_alg": "a_alg",
    # Back-compat for old keys
    "a_dformat": "a_dtype",
    # Metrics
    "perplexity": "wiki2",
    "ppl": "wiki2",
    "average": "avg",
    "mean": "avg",
    # Meta
    "model_name": "model",
    "config": "label",
    "quant": "label",
    "quant_label": "label",
}
SORT_ALLOWED_DISPLAY = ",".join(sorted(CANONICAL_SORT_KEYS | set(SORT_ALIASES.keys())))

# Pretty help text for --sort, grouped by category
SORT_HELP_GROUPED = (
    "Multi-key sort. Prefix with - for desc, + for asc.\n"
    "Default direction: metrics descend; others ascend.\n\n"
    "Keys by group:\n"
    "- Model: model, label\n"
    "- Weights: w_dtype (aliases: dtype, format, w_format, w_dformat), "
    "w_alg (aliases: alg), w_block (aliases: block, block_size)\n"
    "- Activations: a_dtype (aliases: act_dtype, act_format, a_dformat), "
    "a_alg (aliases: act_alg), a_block\n"
    "- Metrics: arc_e, arc_c, boolq, piqa, siqa, hella, obqa, wino, "
    "avg (aliases: average, mean), wiki2 (aliases: perplexity, ppl)"
)

# Default non-metric columns to display (replaces showing only 'label').
META_COLUMNS_DEFAULT: List[str] = [
    "model",
    "w_dtype",
    "w_alg",
    "w_block",
    "a_dtype",
    "a_alg",
    "a_block",
]

# Hide group aliases for convenience
HIDE_GROUPS: Dict[str, List[str]] = {
    "activations": ["a_dtype", "a_alg", "a_block"],
    "activation": ["a_dtype", "a_alg", "a_block"],
    "acts": ["a_dtype", "a_alg", "a_block"],
}


def expand_hide_tokens(tokens: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in tokens:
        if not t:
            continue
        exp = HIDE_GROUPS.get(t, [t])
        for item in exp:
            if item not in seen:
                out.append(item)
                seen.add(item)
    return out


def find_model_label_dirs(output_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (model_short, label, label_dir) found under output_dir.

    Expects layout: <output_dir>/<model_short>/<label>/
    Skips the special directory name 'eval' used for aggregated eval results.
    """
    if not output_dir.exists():
        return []

    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name == "eval":
            # This is where aggregated lm-eval results live; not a model dir
            continue
        for label_dir in model_dir.iterdir():
            if not label_dir.is_dir():
                continue
            yield model_dir.name, label_dir.name, label_dir


def load_eval_results_from_eval_dir(
    output_dir: Path, model_short: str, label: str
) -> Optional[Dict]:
    """Try to load lm-eval accuracy JSON from <output_dir>/eval/*.json if present."""
    eval_dir = output_dir / "eval"
    if not eval_dir.exists():
        return None

    acc_path = eval_dir / f"{model_short}_{label}_results.json"
    if not acc_path.exists():
        return None
    try:
        return load_lm_eval_results(acc_path)
    except Exception:
        return None


def load_lm_eval_results(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_wiki_eval_perplexity(
    base_output: Path, model_short: str, label: str
) -> Optional[float]:
    wiki_path = base_output / model_short / label / "eval_results.json"
    if not wiki_path.exists():
        return None
    try:
        with wiki_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("perplexity"))
    except Exception:
        return None


def extract_task_acc_percent(results: Dict, task: str) -> Optional[float]:
    """Get accuracy (in percent) for a given task from lm-eval results file."""
    try:
        val = results["results"][task]["acc,none"]
    except Exception:
        return None
    if val is None:
        return None
    return float(val) * 100.0


def compute_avg(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def format_value(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.1f}"


def build_rows(
    output_dir: Path,
    include_columns: List[str],
) -> Tuple[List[str], List[List[str]]]:
    """
    Build table headers and rows.

    include_columns controls which metric columns are included (among task shorts, 'avg', 'wiki2').
    Non-metric columns default to individual traits: model, w_dtype, w_alg, w_block, a_dtype, a_alg, a_block.
    """
    # Headers
    headers = META_COLUMNS_DEFAULT + include_columns
    rows: List[List[str]] = []

    for model_short, label, _label_dir in find_model_label_dirs(output_dir):
        # Try to load lm-eval accuracies from <output_dir>/eval if present
        acc_json = load_eval_results_from_eval_dir(output_dir, model_short, label)
        # Prepare values for all metrics we might include
        task_values: Dict[str, Optional[float]] = {}
        for task in TASKS:
            if acc_json is None:
                task_values[TASK_SHORT[task]] = None
            else:
                task_values[TASK_SHORT[task]] = extract_task_acc_percent(acc_json, task)

        avg_val = compute_avg(list(task_values.values()))
        wiki_val = load_wiki_eval_perplexity(output_dir, model_short, label)

        metric_map: Dict[str, Optional[float]] = {
            **task_values,
            "avg": avg_val,
            "wiki2": wiki_val,
        }

        # Parse individual label fields
        fields = parse_label_fields(label)

        # Compose non-metric (meta) cells
        meta_cells: List[str] = [
            model_short,
            fields.get("w_dtype") or "-",
            fields.get("w_alg") or "-",
            fields.get("w_block") or "-",
            fields.get("a_dtype") or "-",
            fields.get("a_alg") or "-",
            fields.get("a_block") or "-",
        ]

        row: List[str] = meta_cells
        for col in include_columns:
            row.append(format_value(metric_map.get(col)))
        rows.append(row)

    return headers, rows


def parse_label_fields(label: str) -> Dict[str, Optional[str]]:
    """Parse common attributes from a label.

    Supported patterns:
      - "Baseline" or any label containing 'baseline' (case-insensitive)
      - "W-<w_dformat>-<w_block>-<w_alg>--A-<a_dformat>-<a_alg>"

    Returns a mapping with keys: w_dtype, w_block, w_alg, a_dtype, a_alg, a_block, dtype (alias of w_dtype).
    Missing/unknown fields are set to None.
    """
    low = label.lower()
    out: Dict[str, Optional[str]] = {
        "w_dtype": None,
        "w_block": None,
        "w_alg": None,
        "a_dtype": None,
        "a_alg": None,
        "a_block": None,
        "dtype": None,
    }
    if "baseline" in low:
        return out

    if label.startswith("W-") and "--A-" in label:
        try:
            w_part_full = label[len("W-") :]
            w_part, a_part = w_part_full.split("--A-", 1)
            w_bits = w_part.split("-")
            if len(w_bits) >= 3:
                out["w_dtype"] = w_bits[0] if w_bits[0] != "None" else None
                out["w_block"] = w_bits[1] if w_bits[1] != "None" else None
                out["w_alg"] = (
                    "-".join(w_bits[2:]) if "-".join(w_bits[2:]) != "None" else None
                )
            a_bits = a_part.split("-")
            if len(a_bits) >= 2:
                out["a_dtype"] = a_bits[0] if a_bits[0] != "None" else None
                out["a_alg"] = (
                    "-".join(a_bits[1:]) if "-".join(a_bits[1:]) != "None" else None
                )
        except Exception:
            pass

    out["dtype"] = out["w_dtype"]
    return out


def is_numeric_key(name: str) -> bool:
    return name in NUMERIC_SORT_KEYS


def resolve_sort_specs(sort_arg: Optional[str]) -> List[Tuple[str, bool]]:
    """Parse --sort into a list of (key, descending) specs.

    Accepted keys:
      - model, label
      - dtype (alias of w_dformat), w_dformat, w_block (alias block, block_size), w_alg (alias alg)
      - a_dformat (alias act_dtype), a_alg (alias act_alg)
      - metrics: arc_e, arc_c, boolq, piqa, siqa, hella, obqa, wino, avg, wiki2

    Prefix with '-' for descending, '+' for ascending. If no prefix: metrics default to descending, others ascending.
    """
    if not sort_arg:
        return []

    specs: List[Tuple[str, bool]] = []
    for raw in [s.strip() for s in sort_arg.split(",") if s.strip()]:
        desc = False
        name = raw
        if raw[0] in "+-":
            desc = raw[0] == "-"
            name = raw[1:]
        name = SORT_ALIASES.get(name, name)
        if name not in CANONICAL_SORT_KEYS:
            raise SystemExit(
                f"Unknown sort key '{name}'. Allowed: {SORT_ALLOWED_DISPLAY}"
            )
        if raw[0] not in "+-":
            # Defaults: metrics -> desc, others -> asc
            desc = is_numeric_key(name)
        specs.append((name, desc))
    return specs


def invert_str_key(s: str) -> str:
    return "".join(chr(0x10FFFF - ord(c)) for c in s)


def build_sort_key_funcs(specs: List[Tuple[str, bool]]):
    def key_for_row(row: Dict) -> Tuple:
        parts = []
        for name, desc in specs:
            # Map name -> value from row
            if name in {
                "model",
                "label",
                "w_dtype",
                "w_block",
                "w_alg",
                "a_dtype",
                "a_alg",
                "a_block",
            }:
                v: Optional[str] = row.get(name)
                if desc:
                    parts.append(
                        (v is None, invert_str_key(v) if v is not None else "")
                    )
                else:
                    parts.append((v is None, v or ""))
            elif name in {
                "avg",
                "wiki2",
                "arc_e",
                "arc_c",
                "boolq",
                "piqa",
                "siqa",
                "hella",
                "obqa",
                "wino",
            }:
                # numeric
                if name in {"avg", "wiki2"}:
                    vnum: Optional[float] = row.get(name)
                else:
                    vnum = (row.get("metrics") or {}).get(name)
                if desc:
                    parts.append((vnum is None, -(vnum if vnum is not None else 0.0)))
                else:
                    parts.append((vnum is None, vnum if vnum is not None else 0.0))
            else:
                # Unknown key, treat as label
                v2: Optional[str] = row.get(name) or row.get("label")
                if desc:
                    parts.append(
                        (v2 is None, invert_str_key(v2) if v2 is not None else "")
                    )
                else:
                    parts.append((v2 is None, v2 or ""))
        return tuple(parts)

    return key_for_row


def build_row_data(output_dir: Path) -> List[Dict]:
    data: List[Dict] = []
    for model_short, label, _label_dir in find_model_label_dirs(output_dir):
        acc_json = load_eval_results_from_eval_dir(output_dir, model_short, label)
        metrics: Dict[str, Optional[float]] = {}
        for task in TASKS:
            short = TASK_SHORT[task]
            metrics[short] = (
                extract_task_acc_percent(acc_json, task)
                if acc_json is not None
                else None
            )
        avg_val = compute_avg(list(metrics.values()))
        wiki_val = load_wiki_eval_perplexity(output_dir, model_short, label)
        fields = parse_label_fields(label)
        row = {
            "model": model_short,
            "label": label,
            "metrics": metrics,
            "avg": avg_val,
            "wiki2": wiki_val,
            **fields,
        }
        data.append(row)
    return data


def compute_col_widths(headers: List[str], rows: List[List[str]]) -> List[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = compute_col_widths(headers, rows)

    def fmt_row(vals: List[str]) -> str:
        return " ".join(f"{v:>{w}}" for v, w in zip(vals, widths))

    print(fmt_row(headers))
    for row in rows:
        print(fmt_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unify parsing of lm-eval accuracies and WikiText-2 perplexity into a single table."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help=(
            "Base output directory that contains <model>/<label>/ and optionally an 'eval' folder."
        ),
    )

    parser.add_argument(
        "--hide",
        action="append",
        default=[],
        help=(
            "Comma-separated list of columns to hide. "
            "Columns: model,w_dtype,w_alg,w_block,a_dtype,a_alg,a_block,"
            "arc_e,arc_c,boolq,piqa,siqa,hella,obqa,wino,avg,wiki2. "
            "Groups: activations (hides a_dtype,a_alg,a_block)."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Show only a single metric column (plus model and label). "
            "One of: arc_e,arc_c,boolq,piqa,siqa,hella,obqa,wino,avg,wiki2"
        ),
    )

    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        help=SORT_HELP_GROUPED,
    )

    return parser.parse_args()


def resolve_columns(hide_args: List[str], only: Optional[str]) -> List[str]:
    metric_columns = [
        "arc_e",
        "arc_c",
        "boolq",
        "piqa",
        "siqa",
        "hella",
        "obqa",
        "wino",
        "avg",
        "wiki2",
    ]

    if only is not None:
        only = only.strip()
        if only not in metric_columns:
            raise SystemExit(
                f"--only must be one of {','.join(metric_columns)}; got '{only}'."
            )
        return [only]

    # Expand hide lists (support group aliases)
    hide_raw: List[str] = []
    for h in hide_args:
        if not h:
            continue
        hide_raw.extend([p.strip() for p in h.split(",") if p.strip()])
    hide = expand_hide_tokens(hide_raw)

    # Support hiding model/label as well
    # Build full column order, then filter
    cols = metric_columns.copy()
    for col in ["model", "label"]:
        if col in hide:
            pass  # handled in header composition
    # Only metric columns are controlled here; model/label handled in build_rows
    cols = [c for c in cols if c not in hide]
    return cols


def main() -> None:
    args = parse_args()

    include_columns = resolve_columns(args.hide, args.only)

    # Build row data and apply sorting if requested
    row_data = build_row_data(args.output_dir)
    sort_specs = resolve_sort_specs(args.sort)
    if sort_specs:
        keyfunc = build_sort_key_funcs(sort_specs)
        row_data = sorted(row_data, key=keyfunc)

    # Compose table
    headers = META_COLUMNS_DEFAULT + include_columns
    rows: List[List[str]] = []
    for rd in row_data:
        # Non-metric columns: show individual traits instead of raw label
        row_out: List[str] = [
            rd.get("model", ""),
            rd.get("w_dtype") or "-",
            rd.get("w_alg") or "-",
            rd.get("w_block") or "-",
            rd.get("a_dtype") or "-",
            rd.get("a_alg") or "-",
            rd.get("a_block") or "-",
        ]
        for col in include_columns:
            if col in {"avg", "wiki2"}:
                row_out.append(format_value(rd.get(col)))
            else:
                row_out.append(format_value((rd.get("metrics") or {}).get(col)))
        rows.append(row_out)

    # Drop any columns the user asked to hide (meta or metrics)
    hide_expanded_raw: List[str] = []
    for h in args.hide:
        if h:
            hide_expanded_raw.extend([p.strip() for p in h.split(",") if p.strip()])
    hide_expanded = expand_hide_tokens(hide_expanded_raw)
    if hide_expanded:
        # Determine indices to drop from headers present in the table
        to_drop_indices: List[int] = [
            i for i, name in enumerate(headers) if name in set(hide_expanded)
        ]
        if to_drop_indices:
            to_drop_indices = sorted(to_drop_indices, reverse=True)
            for idx in to_drop_indices:
                headers.pop(idx)
                for r in rows:
                    r.pop(idx)

    print_table(headers, rows)


if __name__ == "__main__":
    main()
