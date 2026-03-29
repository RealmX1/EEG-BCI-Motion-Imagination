"""
HPO study inspection & reporting utilities.

Provides:
- ``collect_study_report``  – gather trial-level statistics from an Optuna study
- ``render_study_report``   – render the report as a coloured terminal string
- ``generate_hpo_report_plot`` – save a matplotlib dashboard PNG

Extracted from ``scripts/run_hpo.py`` to keep the entry-point lean.
"""

import statistics as _statistics_mod
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna

# ---------------------------------------------------------------------------
# Lazy-loaded ANSI colour helpers (from src.utils.timing)
# ---------------------------------------------------------------------------

_colors_loaded = False
_colored = None
_Colors = None


def _ensure_report_colors():
    """Lazy-load ANSI color helpers for inspect-study rendering."""
    global _colors_loaded, _colored, _Colors
    if not _colors_loaded:
        from src.utils.timing import Colors, colored
        _Colors = Colors
        _colored = colored
        _colors_loaded = True


def _supports_color(use_color: Optional[bool]) -> bool:
    """Whether report rendering should use ANSI colors."""
    if use_color is not None:
        return use_color
    return sys.stdout.isatty()


def _style(text: str, color: str, *, bold: bool = False, use_color: bool) -> str:
    """Apply ANSI style when enabled."""
    if not use_color:
        return text
    _ensure_report_colors()
    return _colored(text, color, bold=bold)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _trial_last_intermediate_value(
    trial: optuna.trial.FrozenTrial,
) -> Optional[float]:
    """返回 trial 最后一次 report 的中间值。"""
    if not trial.intermediate_values:
        return None
    last_step = max(trial.intermediate_values)
    return trial.intermediate_values[last_step]


def _format_metric(value: Optional[float]) -> str:
    """格式化 trial 指标。"""
    if value is None:
        return 'N/A'
    return f"{value:.4f}"


def _format_duration(seconds: Optional[float]) -> str:
    """格式化耗时。"""
    if seconds is None:
        return 'N/A'

    total_seconds = int(seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_seconds_per_epoch(value: Optional[float]) -> str:
    """格式化每 epoch 耗时。"""
    if value is None:
        return 'N/A'
    return f"{value:.2f}s/epoch"


def _format_percent(value: float) -> str:
    """Format percentage with one decimal place."""
    return f"{value * 100:.1f}%"


def _format_params(params: Dict[str, Any]) -> str:
    """紧凑格式化超参数。"""
    if not params:
        return '(no params)'

    parts = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    return ', '.join(parts)


# ---------------------------------------------------------------------------
# Study-name parsing & inference
# ---------------------------------------------------------------------------

def _parse_study_name(study_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """从 study 名称反解 (model, paradigm, task)。"""
    parts = study_name.split('_')
    if len(parts) < 3:
        return None, None, None
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


def _infer_subject_count(
    trials: List[optuna.trial.FrozenTrial],
    *,
    paradigm: Optional[str],
    explicit_subjects: Optional[List[str]] = None,
) -> Optional[int]:
    """为 within/transfer 推断被试数。"""
    if explicit_subjects:
        return len(explicit_subjects)
    if paradigm not in {'within_subject', 'transfer'}:
        return None

    observed_counts = [
        len(trial.intermediate_values)
        for trial in trials
        if trial.intermediate_values
    ]
    return max(observed_counts) if observed_counts else None


# ---------------------------------------------------------------------------
# Epoch / speed estimation
# ---------------------------------------------------------------------------

def _configured_epoch_ceiling(
    *,
    model: Optional[str],
    paradigm: Optional[str],
    task: Optional[str],
    n_channels: int,
    trial_params: Dict[str, Any],
) -> Optional[int]:
    """读取该 category 下单个训练单元的配置 epoch ceiling。"""
    if model is None or paradigm is None or task is None:
        return None

    effective_n_channels = n_channels if n_channels in {8, 32, 61} else None

    if paradigm == 'within_subject':
        from src.config.training import get_default_config
        config = get_default_config(model, task, n_channels=effective_n_channels)
        return int(config['training']['epochs'])

    if paradigm == 'cross_subject':
        from src.config.training import get_cross_subject_config
        config = get_cross_subject_config(model, task, n_channels=effective_n_channels)
        return int(config['training']['epochs'])

    if paradigm == 'transfer':
        if 'finetune_epochs' in trial_params:
            return int(trial_params['finetune_epochs'])
        # Use within-subject config (transfer now shares the same defaults)
        from src.config.training import get_default_config
        config = get_default_config(model, task, n_channels=effective_n_channels)
        return int(config['training']['epochs'])

    return None


def _estimate_trial_epoch_metrics(
    trial: optuna.trial.FrozenTrial,
    *,
    model: Optional[str],
    paradigm: Optional[str],
    task: Optional[str],
    n_channels: int,
    subject_count_estimate: Optional[int],
) -> Dict[str, Any]:
    """估算 trial 的 configured epochs 与每 epoch 速度。"""
    epoch_ceiling = _configured_epoch_ceiling(
        model=model,
        paradigm=paradigm,
        task=task,
        n_channels=n_channels,
        trial_params=trial.params,
    )
    if epoch_ceiling is None:
        return {
            'reported_steps': len(trial.intermediate_values),
            'configured_epochs_estimate': None,
            'estimated_seconds_per_epoch': None,
        }

    if paradigm == 'cross_subject':
        completed_units = 1
    elif paradigm in {'within_subject', 'transfer'}:
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and subject_count_estimate is not None
        ):
            completed_units = subject_count_estimate
        else:
            completed_units = len(trial.intermediate_values)
        if completed_units == 0:
            return {
                'reported_steps': 0,
                'configured_epochs_estimate': None,
                'estimated_seconds_per_epoch': None,
            }
    else:
        completed_units = 1

    configured_epochs_estimate = epoch_ceiling * completed_units
    duration_seconds = None
    if trial.datetime_start and trial.datetime_complete:
        duration_seconds = (
            trial.datetime_complete - trial.datetime_start
        ).total_seconds()

    estimated_seconds_per_epoch = None
    if duration_seconds is not None and configured_epochs_estimate > 0:
        estimated_seconds_per_epoch = duration_seconds / configured_epochs_estimate

    return {
        'reported_steps': len(trial.intermediate_values),
        'configured_epochs_estimate': configured_epochs_estimate,
        'estimated_seconds_per_epoch': estimated_seconds_per_epoch,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _summarize_numeric(values: List[float]) -> Optional[Dict[str, float]]:
    """汇总数值统计。"""
    if not values:
        return None
    return {
        'count': len(values),
        'mean': float(sum(values) / len(values)),
        'median': float(_statistics_mod.median(values)),
        'min': float(min(values)),
        'max': float(max(values)),
    }


def _state_color(state: str) -> str:
    """ANSI color for Optuna trial state."""
    _ensure_report_colors()
    mapping = {
        'COMPLETE': _Colors.BRIGHT_GREEN,
        'RUNNING': _Colors.BRIGHT_CYAN,
        'WAITING': _Colors.BRIGHT_MAGENTA,
        'PRUNED': _Colors.BRIGHT_YELLOW,
        'FAIL': _Colors.BRIGHT_RED,
    }
    return mapping.get(state, _Colors.WHITE)


def _metric_color(
    value: Optional[float],
    *,
    lower_is_better: bool,
    lo: Optional[float],
    hi: Optional[float],
) -> str:
    """Color a numeric metric by its relative quality."""
    _ensure_report_colors()
    if value is None:
        return _Colors.DIM
    if lo is None or hi is None or hi <= lo:
        return _Colors.BRIGHT_CYAN

    ratio = (value - lo) / (hi - lo)
    if lower_is_better:
        ratio = 1.0 - ratio

    if ratio >= 0.67:
        return _Colors.BRIGHT_GREEN
    if ratio >= 0.34:
        return _Colors.BRIGHT_YELLOW
    return _Colors.BRIGHT_RED


def _render_bar(
    ratio: float,
    *,
    width: int = 18,
    color: Optional[str] = None,
    use_color: bool,
) -> str:
    """Render a simple proportional ASCII bar."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    bar = f"[{'#' * filled}{'.' * (width - filled)}]"
    if color is None:
        return bar
    return _style(bar, color, use_color=use_color)


def _state_badge(state: str, *, use_color: bool) -> str:
    """Render a colored state badge."""
    label = f"[{state:^8}]"
    return _style(label, _state_color(state), bold=True, use_color=use_color)


def _render_metric_block(
    title: str,
    stats: Optional[Dict[str, float]],
    *,
    formatter,
    lower_is_better: bool,
    use_color: bool,
) -> List[str]:
    """Render min/median/mean/max as a compact mini-chart."""
    lines = [title]
    if not stats:
        lines.append("  no timing data")
        return lines

    scale_max = max(stats['mean'], stats['median'], stats['min'], stats['max'])
    items = [
        ('mean', stats['mean'], _Colors.BRIGHT_CYAN),
        ('median', stats['median'], _Colors.BRIGHT_BLUE),
        ('min', stats['min'], _Colors.BRIGHT_GREEN if lower_is_better else _Colors.BRIGHT_RED),
        ('max', stats['max'], _Colors.BRIGHT_RED if lower_is_better else _Colors.BRIGHT_GREEN),
    ]
    for label, value, base_color in items:
        ratio = value / scale_max if scale_max > 0 else 0.0
        metric_text = formatter(value)
        label_text = _style(f"{label:<6}", base_color, bold=(label in {'min', 'max'}), use_color=use_color)
        value_text = _style(metric_text, base_color, bold=(label in {'min', 'max'}), use_color=use_color)
        lines.append(
            f"  {label_text} {value_text:<14} "
            f"{_render_bar(ratio, width=20, color=base_color, use_color=use_color)}"
        )
    lines.append(f"  samples {int(stats['count'])}")
    return lines


def _serialize_trial(
    trial: optuna.trial.FrozenTrial,
    *,
    model: Optional[str],
    paradigm: Optional[str],
    task: Optional[str],
    n_channels: int,
    subject_count_estimate: Optional[int],
) -> Dict[str, Any]:
    """提取 trial 摘要，供打印与测试复用。"""
    last_intermediate = _trial_last_intermediate_value(trial)
    display_value = trial.value if trial.value is not None else last_intermediate

    duration_seconds = None
    if trial.datetime_start and trial.datetime_complete:
        duration_seconds = (
            trial.datetime_complete - trial.datetime_start
        ).total_seconds()
    epoch_metrics = _estimate_trial_epoch_metrics(
        trial,
        model=model,
        paradigm=paradigm,
        task=task,
        n_channels=n_channels,
        subject_count_estimate=subject_count_estimate,
    )

    return {
        'number': trial.number,
        'state': trial.state.name,
        'final_value': trial.value,
        'display_value': display_value,
        'display_value_source': (
            'final' if trial.value is not None
            else 'intermediate' if last_intermediate is not None
            else None
        ),
        'params': dict(sorted(trial.params.items())),
        'datetime_start': (
            trial.datetime_start.isoformat(sep=' ', timespec='seconds')
            if trial.datetime_start else None
        ),
        'duration_seconds': duration_seconds,
        'reported_steps': epoch_metrics['reported_steps'],
        'configured_epochs_estimate': epoch_metrics['configured_epochs_estimate'],
        'estimated_seconds_per_epoch': epoch_metrics['estimated_seconds_per_epoch'],
    }


# ===================================================================
# Public API
# ===================================================================

def collect_study_report(
    study: optuna.Study,
    *,
    top_k: int = 3,
    model: Optional[str] = None,
    paradigm: Optional[str] = None,
    task: Optional[str] = None,
    n_channels: int = 128,
    explicit_subjects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """收集某个 study 的 trial 统计与 Top-K 摘要。"""
    inferred_model, inferred_paradigm, inferred_task = _parse_study_name(study.study_name)
    model = model or inferred_model
    paradigm = paradigm or inferred_paradigm
    task = task or inferred_task

    raw_counts = Counter(trial.state.name for trial in study.trials)
    state_order = ['COMPLETE', 'RUNNING', 'WAITING', 'PRUNED', 'FAIL']
    ordered_counts = {
        state: raw_counts.get(state, 0)
        for state in state_order
        if raw_counts.get(state, 0) > 0
    }
    subject_count_estimate = _infer_subject_count(
        study.trials,
        paradigm=paradigm,
        explicit_subjects=explicit_subjects,
    )

    complete_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    complete_trials = sorted(
        complete_trials,
        key=lambda trial: trial.value,
        reverse=reverse,
    )
    serialized_trials = [
        _serialize_trial(
            trial,
            model=model,
            paradigm=paradigm,
            task=task,
            n_channels=n_channels,
            subject_count_estimate=subject_count_estimate,
        )
        for trial in sorted(study.trials, key=lambda trial: trial.number)
    ]
    duration_values = [
        trial['duration_seconds']
        for trial in serialized_trials
        if trial['duration_seconds'] is not None
    ]
    per_epoch_values = [
        trial['estimated_seconds_per_epoch']
        for trial in serialized_trials
        if trial['estimated_seconds_per_epoch'] is not None
    ]
    duration_stats = _summarize_numeric(duration_values)
    per_epoch_stats = _summarize_numeric(per_epoch_values)

    return {
        'study_name': study.study_name,
        'direction': study.direction.name,
        'category': {
            'model': model,
            'paradigm': paradigm,
            'task': task,
            'n_channels': n_channels,
            'subject_count_estimate': subject_count_estimate,
        },
        'n_trials': len(study.trials),
        'aggregate_counts': {
            'complete': raw_counts.get('COMPLETE', 0),
            'incomplete': raw_counts.get('RUNNING', 0) + raw_counts.get('WAITING', 0),
            'pruned': raw_counts.get('PRUNED', 0),
            'aborted': raw_counts.get('FAIL', 0),
        },
        'raw_counts': ordered_counts,
        'speed_stats': {
            'per_trial_duration_seconds': duration_stats,
            'trial_throughput_per_hour': (
                3600.0 / duration_stats['mean']
                if duration_stats is not None and duration_stats['mean'] > 0
                else None
            ),
            'estimated_seconds_per_epoch': per_epoch_stats,
        },
        'top_trials': [
            _serialize_trial(
                trial,
                model=model,
                paradigm=paradigm,
                task=task,
                n_channels=n_channels,
                subject_count_estimate=subject_count_estimate,
            )
            for trial in complete_trials[:top_k]
        ],
        'all_trials': serialized_trials,
    }


def render_study_report(
    report: Dict[str, Any],
    *,
    model: str,
    paradigm: str,
    task: str,
    storage_url: str,
    use_color: Optional[bool] = None,
) -> str:
    """将 study 摘要渲染为更适合终端阅读的彩色报表。"""
    use_color = _supports_color(use_color)
    _ensure_report_colors()
    category = report['category']
    separator = _style("=" * 96, _Colors.BRIGHT_BLUE, use_color=use_color)
    thin_sep = _style("-" * 96, _Colors.BRIGHT_BLUE, use_color=use_color)
    dim = _Colors.DIM

    # -- Header with prominent category display ----------------------------
    category_label = (
        f"{_style(model.upper(), _Colors.BRIGHT_CYAN, bold=True, use_color=use_color)}"
        f"  {_style('|', dim, use_color=use_color)}  "
        f"{_style(paradigm, _Colors.BRIGHT_CYAN, use_color=use_color)}"
        f"  {_style('|', dim, use_color=use_color)}  "
        f"{_style(task, _Colors.BRIGHT_CYAN, use_color=use_color)}"
    )

    lines = [
        "",
        separator,
        _style(f" HPO Study Dashboard | {report['study_name']} ", _Colors.BRIGHT_BLUE, bold=True, use_color=use_color),
        f"  {category_label}",
        separator,
        _style(" Overview ", _Colors.BRIGHT_MAGENTA, bold=True, use_color=use_color),
        (
            f"  category   {_style(model, _Colors.BRIGHT_CYAN, bold=True, use_color=use_color)}"
            f" / {_style(paradigm, _Colors.BRIGHT_CYAN, use_color=use_color)}"
            f" / {_style(task, _Colors.BRIGHT_CYAN, use_color=use_color)}"
        ),
        (
            f"  direction  {_style(report['direction'], _Colors.BRIGHT_YELLOW, bold=True, use_color=use_color)}"
            f"    total trials  {_style(str(report['n_trials']), _Colors.BRIGHT_GREEN, bold=True, use_color=use_color)}"
        ),
        f"  data src   {_style('Optuna study storage (trial-level HPO state/params)', dim, use_color=use_color)}",
        f"  storage    {_style(storage_url, dim, use_color=use_color)}",
    ]

    if category.get('subject_count_estimate') is not None:
        lines.append(
            "  subjects   "
            f"{_style(str(category['subject_count_estimate']), _Colors.BRIGHT_GREEN, bold=True, use_color=use_color)} "
            f"{_style('(estimated from reported steps)', dim, use_color=use_color)}"
        )

    lines.extend([
        "",
        _style(" State Mix ", _Colors.BRIGHT_MAGENTA, bold=True, use_color=use_color),
    ])

    total_trials = max(report['n_trials'], 1)
    for state, count in report['raw_counts'].items():
        ratio = count / total_trials
        badge = _state_badge(state, use_color=use_color)
        pct_text = _format_percent(ratio)
        color = _state_color(state)
        lines.append(
            f"  {badge}  {count:>3}  {pct_text:>6}  "
            f"{_render_bar(ratio, width=28, color=color, use_color=use_color)}"
        )

    lines.append(
        "  summary    "
        f"complete={report['aggregate_counts']['complete']}  "
        f"incomplete={report['aggregate_counts']['incomplete']}  "
        f"pruned={report['aggregate_counts']['pruned']}  "
        f"failed={report['aggregate_counts']['aborted']}"
    )

    speed_stats = report.get('speed_stats', {})
    duration_stats = speed_stats.get('per_trial_duration_seconds')
    per_epoch_stats = speed_stats.get('estimated_seconds_per_epoch')
    if duration_stats or per_epoch_stats:
        lines.extend([
            "",
            _style(" Speed ", _Colors.BRIGHT_MAGENTA, bold=True, use_color=use_color),
        ])
        if speed_stats.get('trial_throughput_per_hour') is not None:
            throughput_text = f"{speed_stats['trial_throughput_per_hour']:.2f} trials/hour"
            lines.append(
                "  throughput "
                f"{_style(throughput_text, _Colors.BRIGHT_GREEN, bold=True, use_color=use_color)}"
            )
        if duration_stats:
            lines.extend(
                f"  {line}" for line in _render_metric_block(
                    "per-trial duration",
                    duration_stats,
                    formatter=_format_duration,
                    lower_is_better=True,
                    use_color=use_color,
                )
            )
        if per_epoch_stats:
            lines.extend(
                f"  {line}" for line in _render_metric_block(
                    "per-epoch speed (estimated)",
                    per_epoch_stats,
                    formatter=_format_seconds_per_epoch,
                    lower_is_better=True,
                    use_color=use_color,
                )
            )
            basis = (
                "configured epoch ceiling x completed reported subject steps"
                if report['category']['paradigm'] in {'within_subject', 'transfer'}
                else "configured epoch ceiling per trial"
            )
            lines.append(f"  basis      {_style(basis, dim, use_color=use_color)}")

    # -- Best Result highlight section -------------------------------------
    if report['top_trials']:
        best = report['top_trials'][0]
        lines.extend([
            "",
            thin_sep,
            _style(" >>> Best Result ", _Colors.BRIGHT_GREEN, bold=True, use_color=use_color),
            thin_sep,
        ])
        best_score = _format_metric(best['final_value'])
        best_trial_num = best['number']
        best_duration = _format_duration(best['duration_seconds'])
        best_per_epoch = _format_seconds_per_epoch(best['estimated_seconds_per_epoch'])
        lines.append(
            "  "
            f"{_style('score', dim, use_color=use_color)}  "
            f"{_style(best_score, _Colors.BRIGHT_GREEN, bold=True, use_color=use_color)}"
            f"    "
            f"{_style('trial', dim, use_color=use_color)} "
            f"{_style(f'#{best_trial_num}', _Colors.BRIGHT_CYAN, bold=True, use_color=use_color)}"
            f"    "
            f"{_style('duration', dim, use_color=use_color)} {best_duration}"
            f"    "
            f"{_style('per_epoch', dim, use_color=use_color)} {best_per_epoch}"
        )
        lines.append(
            f"  {_style('params', dim, use_color=use_color)}  {_format_params(best['params'])}"
        )
        lines.append(thin_sep)

    lines.extend([
        "",
        _style(" Top Trials ", _Colors.BRIGHT_MAGENTA, bold=True, use_color=use_color),
    ])
    if report['top_trials']:
        complete_values = [
            trial['final_value']
            for trial in report['top_trials']
            if trial['final_value'] is not None
        ]
        max_value = max(complete_values) if complete_values else None
        min_value = min(complete_values) if complete_values else None
        for idx, trial in enumerate(report['top_trials'], start=1):
            trial_label = _style(
                f"#{trial['number']}",
                _Colors.BRIGHT_CYAN,
                bold=True,
                use_color=use_color,
            )
            score_color = _metric_color(
                trial['final_value'],
                lower_is_better=False,
                lo=min_value,
                hi=max_value,
            )
            value_ratio = 1.0 if max_value == min_value else (
                (trial['final_value'] - min_value) / (max_value - min_value)
                if trial['final_value'] is not None and max_value is not None and min_value is not None
                else 0.0
            )
            lines.append(
                "  "
                f"{_style(f'#{idx}', _Colors.BRIGHT_YELLOW, bold=True, use_color=use_color)} "
                f"trial {trial_label}  "
                f"score {_style(_format_metric(trial['final_value']), score_color, bold=True, use_color=use_color)}  "
                f"{_render_bar(value_ratio, width=22, color=score_color, use_color=use_color)}"
            )
            lines.append(
                "     "
                f"duration={_format_duration(trial['duration_seconds'])}  "
                f"per_epoch={_format_seconds_per_epoch(trial['estimated_seconds_per_epoch'])}"
            )
            lines.append(f"     params: {_format_params(trial['params'])}")
    else:
        lines.append("  No completed trials yet.")

    lines.extend([
        "",
        _style(" Trial Ledger ", _Colors.BRIGHT_MAGENTA, bold=True, use_color=use_color),
    ])
    if report['all_trials']:
        available_values = [
            trial['display_value']
            for trial in report['all_trials']
            if trial['display_value'] is not None
        ]
        lo = min(available_values) if available_values else None
        hi = max(available_values) if available_values else None
        lines.append(
            "  "
            f"{_style('trial', _Colors.BRIGHT_BLUE, bold=True, use_color=use_color):<9}"
            f"{_style('state', _Colors.BRIGHT_BLUE, bold=True, use_color=use_color):<18}"
            f"{_style('score', _Colors.BRIGHT_BLUE, bold=True, use_color=use_color):<12}"
            f"{_style('visual', _Colors.BRIGHT_BLUE, bold=True, use_color=use_color):<24}"
            f"{_style('duration', _Colors.BRIGHT_BLUE, bold=True, use_color=use_color):<12}"
            f"{_style('per-epoch', _Colors.BRIGHT_BLUE, bold=True, use_color=use_color)}"
        )
        for trial_idx, trial in enumerate(report['all_trials']):
            # Add visual separation every 4 trials for readability
            if trial_idx > 0 and trial_idx % 4 == 0:
                lines.append("")
            value = trial['display_value']
            score_color = _metric_color(
                value,
                lower_is_better=False,
                lo=lo,
                hi=hi,
            )
            ratio = 0.0
            if value is not None and lo is not None and hi is not None:
                ratio = 1.0 if hi == lo else (value - lo) / (hi - lo)
            trial_id = _style(f"#{trial['number']:03d}", _Colors.BRIGHT_CYAN, bold=True, use_color=use_color)
            score_text = _style(_format_metric(value), score_color, bold=(trial['state'] == 'COMPLETE'), use_color=use_color)
            lines.append(
                "  "
                f"{trial_id:<9}"
                f"{_state_badge(trial['state'], use_color=use_color):<18}"
                f"{score_text:<12}"
                f"{_render_bar(ratio, width=20, color=score_color, use_color=use_color):<24}"
                f"{_format_duration(trial['duration_seconds']):<12}"
                f"{_format_seconds_per_epoch(trial['estimated_seconds_per_epoch'])}"
            )
            lines.append(
                f"     started: {trial['datetime_start'] or 'N/A'}"
            )
            lines.append(
                f"     params:  {_format_params(trial['params'])}"
            )
    else:
        lines.append("  No trials found in this study.")

    # -- Footer summary line -----------------------------------------------
    n_complete = report['aggregate_counts']['complete']
    n_total = report['n_trials']
    throughput = speed_stats.get('trial_throughput_per_hour')
    best_trial_info = ""
    if report['top_trials']:
        best_t = report['top_trials'][0]
        best_trial_info = (
            f"Best: "
            f"{_style(_format_metric(best_t['final_value']), _Colors.BRIGHT_GREEN, bold=True, use_color=use_color)}"
            f" (trial #{best_t['number']})"
        )
    else:
        best_trial_info = "Best: N/A"
    throughput_info = f"{throughput:.2f} trials/hr" if throughput else "N/A"
    lines.extend([
        "",
        (
            f"  {best_trial_info}"
            f"  |  {n_complete}/{n_total} complete"
            f"  |  {throughput_info}"
        ),
    ])

    lines.append(separator)
    return '\n'.join(lines)


# ===================================================================
# Matplotlib dashboard
# ===================================================================

def generate_hpo_report_plot(
    report: Dict[str, Any],
    *,
    save_dir: Optional[str] = None,
) -> Optional[Path]:
    """Create a matplotlib dashboard and save to PNG.

    Parameters
    ----------
    report : dict
        Output of ``collect_study_report``.
    save_dir : str, optional
        Directory to save the PNG.  Defaults to ``results/hpo``.

    Returns
    -------
    Path or None
        Path to the saved PNG, or None if there were no trials to plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    all_trials = report.get('all_trials', [])
    if not all_trials:
        return None

    study_name = report['study_name']
    save_dir_path = Path(save_dir) if save_dir else Path('results/hpo')
    save_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = save_dir_path / f"{study_name}_dashboard.png"

    # -- Categorise trials --------------------------------------------------
    state_color_map = {
        'COMPLETE': '#2ecc71',   # green
        'PRUNED': '#f1c40f',     # yellow
        'FAIL': '#e74c3c',       # red
        'RUNNING': '#3498db',    # blue
        'WAITING': '#9b59b6',    # purple
    }

    numbers = []       # type: List[int]
    scores = []        # type: List[float]
    colors = []        # type: List[str]
    for t in all_trials:
        value = t['display_value']
        if value is None:
            continue
        numbers.append(t['number'])
        scores.append(value)
        colors.append(state_color_map.get(t['state'], '#95a5a6'))

    if not numbers:
        return None

    # Identify completed trials with numeric hyperparams for panel 2
    complete_trials = [
        t for t in all_trials
        if t['state'] == 'COMPLETE' and t['final_value'] is not None
    ]
    numeric_params = {}  # type: Dict[str, List[Tuple[float, float]]]
    for t in complete_trials:
        for k, v in t['params'].items():
            if isinstance(v, (int, float)):
                numeric_params.setdefault(k, []).append((float(v), float(t['final_value'])))

    show_param_panel = len(complete_trials) >= 5 and len(numeric_params) > 0

    # -- Layout -------------------------------------------------------------
    if show_param_panel:
        n_params = len(numeric_params)
        ncols = min(n_params, 3)
        nrows = (n_params + ncols - 1) // ncols
        fig = plt.figure(figsize=(14, 5 + 3.5 * nrows))
        gs = fig.add_gridspec(
            1 + nrows, ncols,
            height_ratios=[5] + [3.5] * nrows,
            hspace=0.35,
            wspace=0.35,
        )
        ax_top = fig.add_subplot(gs[0, :])
    else:
        fig, ax_top = plt.subplots(figsize=(14, 5))

    # -- Panel 1: Trial score progression -----------------------------------
    ax_top.scatter(numbers, scores, c=colors, s=40, alpha=0.85, edgecolors='k', linewidths=0.3, zorder=3)

    # Best score line
    best_complete = [t for t in all_trials if t['state'] == 'COMPLETE' and t['final_value'] is not None]
    if best_complete:
        best_score = max(t['final_value'] for t in best_complete)
        ax_top.axhline(best_score, color='#2ecc71', linestyle='--', linewidth=1.2, alpha=0.7, label=f'best = {best_score:.4f}')

    # Legend entries for states actually present
    from matplotlib.lines import Line2D
    present_states = sorted({t['state'] for t in all_trials if t['display_value'] is not None})
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=state_color_map.get(s, '#95a5a6'),
               markersize=7, label=s)
        for s in present_states
    ]
    if best_complete:
        legend_handles.append(
            Line2D([0], [0], color='#2ecc71', linestyle='--', linewidth=1.2, label=f'best = {best_score:.4f}')
        )
    ax_top.legend(handles=legend_handles, loc='lower right', fontsize=8, framealpha=0.85)

    ax_top.set_xlabel('Trial number')
    ax_top.set_ylabel('Score')
    ax_top.set_title(f'Trial Score Progression  —  {study_name}', fontweight='bold')
    ax_top.grid(True, alpha=0.3)

    # -- Panel 2: Hyperparameter importance (scatter per param) -------------
    if show_param_panel:
        param_names = sorted(numeric_params.keys())
        for idx, pname in enumerate(param_names):
            row = idx // ncols
            col = idx % ncols
            ax = fig.add_subplot(gs[1 + row, col])
            pairs = numeric_params[pname]
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            ax.scatter(xs, ys, c='#2ecc71', s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
            ax.set_xlabel(pname, fontsize=9)
            ax.set_ylabel('Score', fontsize=9)
            ax.set_title(pname, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Use log scale if the values span more than 1 order of magnitude
            if xs:
                x_min, x_max = min(xs), max(xs)
                if x_min > 0 and x_max / x_min > 10:
                    ax.set_xscale('log')

    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path
