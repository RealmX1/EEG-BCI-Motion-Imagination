"""Generate a self-contained HTML overview of run purposes / groups.

Queries the live ExperimentDB, aggregates runs by subgroup (parsed from notes),
and renders an HTML page with summary stats, purpose distribution chart,
and a collapsible group table.
"""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path

from src.results.experiment_db import ExperimentDB


PURPOSE_COLORS = {
    "baseline": "#2563eb",       # blue
    "final": "#16a34a",          # green
    "replication": "#0891b2",    # cyan
    "ablation": "#f59e0b",       # amber
    "hpo": "#7c3aed",            # violet
    "sweep": "#db2777",          # pink
    "sanity_check": "#94a3b8",   # slate
    "pilot": "#a78bfa",          # light violet
    "debug": "#ef4444",          # red
    "misc": "#6b7280",           # gray
}

PROVENANCE_BADGES = {
    "explicit": ("explicit", "#16a34a"),
    "backward_search": ("backward search", "#94a3b8"),
}


GROUP_RX = re.compile(r"Group:\s*([^\s(]+)\s*(\([^)]*\))?")
CONFIDENCE_RX = re.compile(r"confidence=(\w+)")
HYPOTHESIS_RX = re.compile(r"^(H:[^\n]+)")


def parse_notes(notes: str | None) -> dict:
    """Extract group name, confidence, hypothesis from notes string."""
    if not notes:
        return {"group": None, "confidence": None, "hypothesis": None}
    m_g = GROUP_RX.search(notes)
    m_c = CONFIDENCE_RX.search(notes)
    m_h = HYPOTHESIS_RX.search(notes)
    return {
        "group": m_g.group(1) if m_g else None,
        "confidence": m_c.group(1) if m_c else None,
        "hypothesis": m_h.group(1).strip() if m_h else (notes.split("\n")[0][:300] if notes else None),
    }


def main():
    db = ExperimentDB()
    with db._connection() as conn:
        rows = conn.execute("""
            SELECT run_tag, run_id, experiment_type, paradigm, task,
                   n_channels, channel_config, created_at, is_baseline,
                   purpose, purpose_provenance, superseded_by, notes
            FROM runs
            ORDER BY purpose, created_at
        """).fetchall()

    runs = [dict(r) for r in rows]
    for r in runs:
        info = parse_notes(r.get("notes"))
        r["group"] = info["group"] or (
            "AUTHORITATIVE_BASELINE" if r.get("purpose") == "baseline" else "ungrouped"
        )
        r["confidence"] = info["confidence"]
        r["hypothesis"] = info["hypothesis"]

    # Aggregate by group
    groups = defaultdict(list)
    for r in runs:
        groups[r["group"]].append(r)

    # Group metadata (one entry's hypothesis represents the group)
    group_meta = []
    for name, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        purposes = sorted({m["purpose"] for m in members if m["purpose"]})
        provenances = sorted({m["purpose_provenance"] for m in members if m["purpose_provenance"]})
        confidences = sorted({m["confidence"] for m in members if m["confidence"]})
        hypotheses = [m["hypothesis"] for m in members if m["hypothesis"]]
        hypothesis = hypotheses[0] if hypotheses else None

        # Date range
        dates = sorted([m["created_at"][:10] for m in members if m["created_at"]])
        date_range = f"{dates[0]} .. {dates[-1]}" if dates else ""

        # n_channels / channel_config summary
        ch_summary = sorted({
            f"{m['n_channels']}ch/{m['channel_config'] or '(default)'}"
            for m in members
        })
        exp_summary = sorted({m["experiment_type"] for m in members})
        task_summary = sorted({m["task"] for m in members})

        group_meta.append({
            "name": name,
            "size": len(members),
            "purposes": purposes,
            "provenances": provenances,
            "confidences": confidences,
            "hypothesis": hypothesis,
            "date_range": date_range,
            "channel_summary": ch_summary,
            "experiment_types": exp_summary,
            "tasks": task_summary,
            "members": [
                {
                    "run_tag": m["run_tag"],
                    "experiment_type": m["experiment_type"],
                    "task": m["task"],
                    "n_channels": m["n_channels"],
                    "channel_config": m["channel_config"],
                    "created_at": m["created_at"][:16],
                    "purpose": m["purpose"],
                    "provenance": m["purpose_provenance"],
                    "is_baseline": bool(m["is_baseline"]),
                    "superseded_by": m["superseded_by"],
                }
                for m in sorted(members, key=lambda x: x["created_at"] or "")
            ],
        })

    # Summary stats
    total = len(runs)
    by_purpose = defaultdict(int)
    by_provenance = defaultdict(int)
    by_purpose_provenance = defaultdict(int)
    for r in runs:
        p = r.get("purpose") or "(null)"
        pv = r.get("purpose_provenance") or "(null)"
        by_purpose[p] += 1
        by_provenance[pv] += 1
        by_purpose_provenance[(p, pv)] += 1

    n_baselines_explicit = by_purpose_provenance.get(("baseline", "explicit"), 0)
    n_explicit = by_provenance.get("explicit", 0)
    n_backward = by_provenance.get("backward_search", 0)

    stats = {
        "total": total,
        "n_explicit": n_explicit,
        "n_backward": n_backward,
        "n_baselines_explicit": n_baselines_explicit,
        "n_groups": len(group_meta),
        "by_purpose": dict(by_purpose),
    }

    db.close()

    # Generate HTML
    out_path = Path("docs/dev_log/backward_search_2026-05-13/purpose_overview.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_str = render_html(stats, group_meta)
    out_path.write_text(html_str, encoding="utf-8")
    print(f"Wrote {out_path}  ({len(html_str):,} bytes)")
    print(f"Open in browser: file:///{out_path.resolve().as_posix()}")


def render_html(stats: dict, groups: list) -> str:
    """Render the HTML page with inline JSON data + vanilla JS."""
    payload = json.dumps({
        "stats": stats,
        "groups": groups,
        "purpose_colors": PURPOSE_COLORS,
    }, ensure_ascii=False, indent=None)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>EEG-BCI Experiment Run Purposes Overview</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei",
                 "PingFang SC", Arial, sans-serif;
    max-width: 1280px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    color: #1f2937;
    background: #fafafa;
    line-height: 1.55;
  }}
  h1 {{ margin: 0 0 0.25rem; }}
  h2 {{ margin: 2.5rem 0 0.75rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.35rem; }}
  .subtitle {{ color: #6b7280; font-size: 0.95rem; margin: 0 0 1.5rem; }}
  .stat-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }}
  .stat-card {{
    background: white;
    padding: 0.9rem 1rem;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
    text-align: center;
  }}
  .stat-card .num {{ font-size: 1.8rem; font-weight: 600; color: #111827; }}
  .stat-card .lbl {{ font-size: 0.85rem; color: #6b7280; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1rem 0; }}
  .chart-box {{ background: white; padding: 1rem; border: 1px solid #e5e7eb; border-radius: 6px; }}
  .chart-box h3 {{ margin: 0 0 0.5rem; font-size: 1rem; font-weight: 600; }}
  .controls {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0; align-items: center; }}
  .controls label {{ font-size: 0.85rem; color: #4b5563; }}
  .controls input[type="text"], .controls select {{
    padding: 0.35rem 0.6rem; border: 1px solid #d1d5db; border-radius: 4px; font-size: 0.9rem;
  }}
  .group {{
    background: white;
    border: 1px solid #e5e7eb;
    border-left: 5px solid #d1d5db;
    border-radius: 6px;
    margin-bottom: 0.6rem;
    overflow: hidden;
  }}
  .group-header {{
    padding: 0.7rem 1rem;
    cursor: pointer;
    display: grid;
    grid-template-columns: minmax(0, 2fr) 100px 70px 70px 90px;
    gap: 0.5rem;
    align-items: center;
    user-select: none;
    transition: background 0.1s;
  }}
  .group-header:hover {{ background: #f9fafb; }}
  .group-name {{ font-weight: 600; font-size: 0.95rem; color: #111827; overflow: hidden;
                 text-overflow: ellipsis; white-space: nowrap; }}
  .group-hypothesis {{ font-size: 0.8rem; color: #6b7280; grid-column: 1 / 2;
                       margin-top: 0.15rem; overflow: hidden;
                       text-overflow: ellipsis; white-space: nowrap; }}
  .badge {{
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    color: white;
    font-weight: 500;
    text-align: center;
  }}
  .badge-outline {{
    background: transparent; color: #4b5563;
    border: 1px solid #d1d5db;
  }}
  .conf-high {{ color: #16a34a; font-weight: 600; }}
  .conf-medium {{ color: #f59e0b; font-weight: 600; }}
  .conf-low {{ color: #ef4444; font-weight: 600; }}
  .conf-na {{ color: #9ca3af; }}
  .group-body {{
    padding: 0 1rem 0.8rem;
    border-top: 1px solid #f3f4f6;
    display: none;
  }}
  .group.open .group-body {{ display: block; }}
  .group-meta {{ font-size: 0.85rem; color: #4b5563; margin: 0.4rem 0; }}
  .group-meta strong {{ color: #1f2937; }}
  table.members {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 0.3rem;
  }}
  table.members th, table.members td {{
    padding: 0.3rem 0.5rem;
    border-bottom: 1px solid #f3f4f6;
    text-align: left;
  }}
  table.members th {{ color: #6b7280; font-weight: 500; background: #fafafa; }}
  table.members tr:hover {{ background: #f9fafb; }}
  .run-tag {{ font-family: ui-monospace, SF Mono, Menlo, monospace; font-size: 0.78rem; }}
  .footnote {{ color: #6b7280; font-size: 0.85rem; margin-top: 2rem; padding-top: 1rem;
               border-top: 1px solid #e5e7eb; }}
  .footnote code {{ background: #f3f4f6; padding: 0.1rem 0.3rem; border-radius: 3px; }}
  details summary {{ cursor: pointer; font-weight: 500; margin: 0.5rem 0; }}
  .legend-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; font-size: 0.8rem;
                 margin-top: 0.5rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3rem; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
</style>
</head>
<body>

<h1>EEG-BCI · Experiment Run Purpose Overview</h1>
<p class="subtitle">
  ExperimentDB schema v10 — runs.purpose / purpose_provenance / superseded_by.
  Generated from live <code>results/experiments.db</code> snapshot.
</p>

<h2>📊 Summary</h2>
<div class="stat-cards" id="stat-cards"></div>

<details open>
  <summary>Schema work summary</summary>
  <ul style="font-size: 0.9rem; color: #4b5563;">
    <li><strong>v9</strong>: added <code>runs.purpose</code> (controlled vocab — 10 values) + <code>notes</code> activation</li>
    <li><strong>v10</strong>: added <code>runs.purpose_provenance</code> ('explicit' | 'backward_search') and <code>runs.superseded_by</code> (self-ref FK)</li>
    <li><strong>Helpers</strong>: <code>set_purpose()</code>, <code>mark_superseded()</code> with cycle detection</li>
    <li><strong>CLI</strong>: <code>--purpose</code> / <code>--notes</code> wired through <code>add_common_args()</code> → all 9 experiment scripts</li>
    <li><strong>Constraint (CLAUDE.md + CLI help)</strong>: <em>purpose encodes HYPOTHESIS being tested, NOT post-hoc analysis</em></li>
    <li><strong>Backward search</strong>: 3 parallel research agents inferred intent from <code>docs/dev_log</code> + <code>handoffs</code> + Claude historian for 287 non-baseline runs across 45 sub-groups</li>
    <li><strong>Audit trail</strong>: <code>docs/dev_log/backward_search_2026-05-13/*.json</code> with per-subgroup rationale + confidence</li>
  </ul>
</details>

<h2>🎯 Purpose Distribution</h2>
<div class="chart-row">
  <div class="chart-box">
    <h3>By purpose category</h3>
    <canvas id="chart-purpose" height="200"></canvas>
  </div>
  <div class="chart-box">
    <h3>By provenance</h3>
    <canvas id="chart-provenance" height="200"></canvas>
  </div>
</div>

<h2>📁 Groups ({stats['n_groups']} total) — click to expand</h2>

<div class="controls">
  <label>Filter:
    <input type="text" id="filter-text" placeholder="search name / hypothesis / run_tag">
  </label>
  <label>Purpose:
    <select id="filter-purpose">
      <option value="">(all)</option>
    </select>
  </label>
  <label>Provenance:
    <select id="filter-provenance">
      <option value="">(all)</option>
      <option value="explicit">explicit</option>
      <option value="backward_search">backward_search</option>
    </select>
  </label>
  <label>Sort:
    <select id="sort-by">
      <option value="size">By size (largest first)</option>
      <option value="name">By name</option>
      <option value="date">By earliest date</option>
      <option value="purpose">By purpose</option>
    </select>
  </label>
  <span id="count-display" style="font-size: 0.85rem; color: #4b5563; margin-left: auto;"></span>
</div>

<div class="legend-row" id="purpose-legend"></div>

<div id="groups"></div>

<div class="footnote">
  <strong>How to read this page</strong>:
  Each group is a cluster of runs that share a research intent. The colored left-border indicates
  the dominant <code>purpose</code> tag. The badge in the row shows whether the tag is
  <code>explicit</code> (CLI-set or memory-entry authoritative) or <code>backward_search</code>
  (post-hoc inferred — best-effort, not absolute truth).
  Click a group header to expand all member runs.<br><br>
  <strong>Authoritative sources</strong>: baselines (<code>is_baseline=1</code>),
  4ch FDR∩Attention memory entry (7 runs, <code>ablation</code>),
  32ch FDR-complement memory entry (2 runs, <code>debug</code>). Everything else is backward-searched.
</div>

<script>
const DATA = {payload};
const PURPOSE_COLORS = DATA.purpose_colors;

function el(tag, attrs, children) {{
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {{
    if (k === 'style') Object.assign(e.style, attrs[k]);
    else if (k === 'onclick') e.onclick = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }}
  if (children) {{
    if (typeof children === 'string') e.textContent = children;
    else if (Array.isArray(children)) for (const c of children) if (c != null) e.append(typeof c === 'string' ? document.createTextNode(c) : c);
    else e.append(children);
  }}
  return e;
}}

// Build stat cards
const statBox = document.getElementById('stat-cards');
const cards = [
  ['total runs', DATA.stats.total],
  ['groups', DATA.stats.n_groups],
  ['explicit (authoritative)', DATA.stats.n_explicit],
  ['backward-searched', DATA.stats.n_backward],
  ['baselines (is_baseline=1)', DATA.stats.n_baselines_explicit],
];
for (const [lbl, num] of cards) {{
  statBox.append(el('div', {{class: 'stat-card'}}, [
    el('div', {{class: 'num'}}, String(num)),
    el('div', {{class: 'lbl'}}, lbl),
  ]));
}}

// Purpose chart
const purposeEntries = Object.entries(DATA.stats.by_purpose)
  .sort((a, b) => b[1] - a[1])
  .filter(([k]) => k !== '(null)');
new Chart(document.getElementById('chart-purpose'), {{
  type: 'bar',
  data: {{
    labels: purposeEntries.map(e => e[0]),
    datasets: [{{
      data: purposeEntries.map(e => e[1]),
      backgroundColor: purposeEntries.map(e => PURPOSE_COLORS[e[0]] || '#9ca3af'),
    }}],
  }},
  options: {{
    indexAxis: 'y',
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ beginAtZero: true, ticks: {{ stepSize: 25 }} }} }},
  }},
}});

// Provenance chart (doughnut)
new Chart(document.getElementById('chart-provenance'), {{
  type: 'doughnut',
  data: {{
    labels: ['explicit (authoritative)', 'backward_search (inferred)'],
    datasets: [{{
      data: [DATA.stats.n_explicit, DATA.stats.n_backward],
      backgroundColor: ['#16a34a', '#94a3b8'],
    }}],
  }},
  options: {{
    plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 11 }} }} }} }},
  }},
}});

// Purpose legend
const legend = document.getElementById('purpose-legend');
for (const [p, c] of Object.entries(PURPOSE_COLORS)) {{
  legend.append(el('div', {{class: 'legend-item'}}, [
    el('div', {{class: 'legend-swatch', style: {{background: c}}}}),
    p,
  ]));
}}

// Purpose filter dropdown
const purposeSelect = document.getElementById('filter-purpose');
for (const [p, _] of purposeEntries) {{
  purposeSelect.append(el('option', {{value: p}}, p));
}}

function renderGroups() {{
  const ftxt = (document.getElementById('filter-text').value || '').toLowerCase();
  const fpur = document.getElementById('filter-purpose').value;
  const fprv = document.getElementById('filter-provenance').value;
  const sort = document.getElementById('sort-by').value;

  let groups = DATA.groups.slice();
  if (ftxt) groups = groups.filter(g =>
    g.name.toLowerCase().includes(ftxt) ||
    (g.hypothesis || '').toLowerCase().includes(ftxt) ||
    g.members.some(m => (m.run_tag || '').includes(ftxt))
  );
  if (fpur) groups = groups.filter(g => g.purposes.includes(fpur));
  if (fprv) groups = groups.filter(g => g.provenances.includes(fprv));

  if (sort === 'size') groups.sort((a, b) => b.size - a.size);
  else if (sort === 'name') groups.sort((a, b) => a.name.localeCompare(b.name));
  else if (sort === 'date') groups.sort((a, b) => (a.date_range || '').localeCompare(b.date_range || ''));
  else if (sort === 'purpose') groups.sort((a, b) => (a.purposes[0] || '').localeCompare(b.purposes[0] || ''));

  const container = document.getElementById('groups');
  container.innerHTML = '';
  for (const g of groups) container.append(renderGroup(g));

  document.getElementById('count-display').textContent =
    `${{groups.length}} groups / ${{groups.reduce((s, g) => s + g.size, 0)}} runs`;
}}

function renderGroup(g) {{
  const dominantPurpose = g.purposes[0];
  const groupEl = el('div', {{class: 'group'}});
  groupEl.style.borderLeftColor = PURPOSE_COLORS[dominantPurpose] || '#9ca3af';

  const purposeBadges = g.purposes.map(p =>
    el('span', {{class: 'badge', style: {{background: PURPOSE_COLORS[p] || '#9ca3af', marginRight: '4px'}}}}, p));

  const provBadge = el('span', {{
    class: 'badge',
    style: {{background: g.provenances.includes('explicit') ? '#16a34a' : '#94a3b8'}},
  }}, g.provenances.join(' + '));

  const conf = g.confidences[0];
  const confEl = el('span', {{class: 'conf-' + (conf || 'na')}}, conf || '-');

  const header = el('div', {{class: 'group-header'}}, [
    el('div', {{}}, [
      el('div', {{class: 'group-name'}}, g.name),
      el('div', {{class: 'group-hypothesis'}}, g.hypothesis || '(no hypothesis recorded)'),
    ]),
    el('div', {{}}, purposeBadges),
    el('div', {{style: {{textAlign: 'center', fontSize: '0.85rem'}}}}, String(g.size)),
    el('div', {{style: {{textAlign: 'center'}}}}, confEl),
    el('div', {{style: {{textAlign: 'right'}}}}, provBadge),
  ]);
  header.onclick = () => groupEl.classList.toggle('open');

  const body = el('div', {{class: 'group-body'}});
  body.append(el('div', {{class: 'group-meta'}}, [
    el('strong', {{}}, 'Hypothesis: '), g.hypothesis || '-',
  ]));
  body.append(el('div', {{class: 'group-meta'}}, [
    el('strong', {{}}, 'Date range: '), g.date_range || '-',
    ' · ',
    el('strong', {{}}, 'Channels: '), g.channel_summary.join(', '),
    ' · ',
    el('strong', {{}}, 'Experiment types: '), g.experiment_types.join(', '),
    ' · ',
    el('strong', {{}}, 'Tasks: '), g.tasks.join(', '),
  ]));

  const tbl = el('table', {{class: 'members'}});
  const thead = el('tr', {{}}, ['run_tag', 'created_at', 'experiment_type', 'task', 'channels', 'cfg', 'baseline?', 'superseded']
    .map(h => el('th', {{}}, h)));
  tbl.append(thead);
  for (const m of g.members) {{
    tbl.append(el('tr', {{}}, [
      el('td', {{class: 'run-tag'}}, m.run_tag),
      el('td', {{}}, m.created_at),
      el('td', {{}}, m.experiment_type),
      el('td', {{}}, m.task),
      el('td', {{}}, String(m.n_channels)),
      el('td', {{}}, m.channel_config || '-'),
      el('td', {{}}, m.is_baseline ? '✓' : ''),
      el('td', {{class: 'run-tag', style: {{color: '#ef4444'}}}}, m.superseded_by || ''),
    ]));
  }}
  body.append(tbl);
  groupEl.append(header, body);
  return groupEl;
}}

for (const id of ['filter-text', 'filter-purpose', 'filter-provenance', 'sort-by']) {{
  document.getElementById(id).addEventListener('input', renderGroups);
}}
renderGroups();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
