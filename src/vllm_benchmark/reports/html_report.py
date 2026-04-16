"""Interactive HTML benchmark report with Plotly charts.

Generates a self-contained HTML file that can be shared with teams,
posted in Slack, or emailed to managers — no dependencies required to view.

Author: amit
License: MIT
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_benchmark.analysis.diagnostics import Diagnostic
    from vllm_benchmark.analysis.scoring import ScoreBreakdown


def generate_html_report(
    results: list[dict],
    metadata: dict,
    score: ScoreBreakdown | None = None,
    diagnostics: list[Diagnostic] | None = None,
    output_path: str = "./outputs",
) -> str:
    """Generate self-contained HTML benchmark report.

    Returns:
        Path to the generated HTML file.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = metadata.get("server_info", {}).get("model_name", "unknown")
    safe_name = model_name.replace("/", "_").replace("\\", "_")[:60]
    filename = output_dir / f"benchmark_{safe_name}_{timestamp}.html"

    results_json = json.dumps(results)
    system_info = metadata.get("system_info", {})
    server_info = metadata.get("server_info", {})

    # Build diagnostics HTML
    diag_html = ""
    if diagnostics:
        cards = []
        for d in diagnostics:
            color_map = {
                "critical": ("#ff4444", "#fff5f5", "CRITICAL"),
                "warning": ("#ff9800", "#fff8e1", "WARNING"),
                "success": ("#4caf50", "#e8f5e9", "OK"),
                "info": ("#2196f3", "#e3f2fd", "INFO"),
            }
            border, bg, label = color_map.get(d.severity, ("#999", "#f5f5f5", "INFO"))
            cards.append(
                f'<div style="border-left:4px solid {border};background:{bg};padding:12px 16px;'
                f'margin:8px 0;border-radius:0 6px 6px 0">'
                f'<strong style="color:{border}">[{label}] {d.title}</strong>'
                f'<p style="margin:4px 0 0;color:#333">{d.message}</p></div>'
            )
        diag_html = "\n".join(cards)
    else:
        diag_html = '<p style="color:#4caf50">No diagnostics generated.</p>'

    # Build score HTML
    score_html = ""
    if score:
        grade_colors = {"S": "#e040fb", "A": "#4caf50", "B": "#8bc34a", "C": "#ffc107", "D": "#ff5722", "F": "#d32f2f"}
        gc = grade_colors.get(score.grade, "#999")
        score_html = f"""
        <div style="text-align:center;padding:24px;background:linear-gradient(135deg,#1a1a2e,#16213e);
                     border-radius:12px;margin:16px 0">
            <div style="font-size:14px;color:#aaa;text-transform:uppercase;letter-spacing:2px">vLLM Benchmark Score</div>
            <div style="font-size:64px;font-weight:bold;color:{gc};margin:8px 0">{score.overall:,}</div>
            <div style="font-size:28px;color:{gc};margin-bottom:12px">Grade: {score.grade}</div>
            <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap">
                <div style="color:#ddd"><small>Throughput</small><br><b>{score.throughput:,}</b></div>
                <div style="color:#ddd"><small>Latency</small><br><b>{score.latency:,}</b></div>
                <div style="color:#ddd"><small>Efficiency</small><br><b>{score.efficiency:,}</b></div>
                <div style="color:#ddd"><small>Energy</small><br><b>{score.energy:,}</b></div>
                <div style="color:#ddd"><small>Consistency</small><br><b>{score.consistency:,}</b></div>
            </div>
        </div>"""

    # System info table rows
    sys_rows = ""
    for key, label in [
        ("gpu_name", "GPU"), ("total_vram_gb", "VRAM (GB)"),
        ("cuda_version", "CUDA"), ("driver_version", "Driver"),
        ("python_version", "Python"), ("platform", "Platform"),
    ]:
        val = system_info.get(key)
        if val:
            if key == "total_vram_gb":
                val = f"{val:.0f} GB"
            sys_rows += f"<tr><td><strong>{label}</strong></td><td>{val}</td></tr>"

    for key, label in [
        ("model_name", "Model"), ("version", "vLLM Version"),
        ("quantization", "Quantization"), ("max_model_len", "Max Context"),
    ]:
        val = server_info.get(key)
        if val:
            if key == "max_model_len":
                val = f"{val:,} tokens"
            sys_rows += f"<tr><td><strong>{label}</strong></td><td>{val}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>vLLM Benchmark Report — {model_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#0f0f1a;color:#e0e0e0;padding:24px;max-width:1400px;margin:0 auto}}
  h1{{font-size:28px;color:#fff;margin-bottom:4px}}
  h2{{font-size:20px;color:#64b5f6;margin:32px 0 16px;border-bottom:1px solid #333;padding-bottom:8px}}
  .subtitle{{color:#999;font-size:14px;margin-bottom:24px}}
  .card{{background:#1a1a2e;border-radius:8px;padding:20px;margin:16px 0}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#16213e;color:#64b5f6;padding:10px 12px;text-align:left;font-weight:600}}
  td{{padding:8px 12px;border-bottom:1px solid #222}}
  tr:hover td{{background:#16213e}}
  .chart-container{{background:#1a1a2e;border-radius:8px;padding:16px;margin:16px 0}}
  .footer{{text-align:center;color:#666;font-size:12px;margin-top:40px;padding:20px}}
</style>
</head>
<body>
<h1>vLLM Benchmark Report</h1>
<p class="subtitle">{model_name} &mdash; {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

{score_html}

<h2>System &amp; Server</h2>
<div class="card">
<table>{sys_rows}</table>
</div>

<h2>Performance Charts</h2>
<div class="chart-container" id="throughput-chart"></div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
  <div class="chart-container" id="latency-heatmap"></div>
  <div class="chart-container" id="throughput-heatmap"></div>
</div>
<div class="chart-container" id="ttft-chart"></div>
<div class="chart-container" id="itl-chart"></div>

<h2>Diagnostics</h2>
<div class="card">{diag_html}</div>

<h2>Raw Results</h2>
<div class="card" style="overflow-x:auto">
<table>
<thead><tr>
<th>Context</th><th>Users</th><th>Tok/s</th><th>Latency(s)</th>
<th>TTFT(ms)</th><th>Tok/s/User</th><th>Success</th>
</tr></thead>
<tbody>
"""

    for r in sorted(results, key=lambda x: (x.get("context_length", 0), x.get("concurrent_users", 0))):
        ctx_k = r.get("context_length", 0) / 1000
        html += (
            f"<tr><td>{ctx_k:.0f}K</td>"
            f"<td>{r.get('concurrent_users', '?')}</td>"
            f"<td>{r.get('tokens_per_second', 0):.1f}</td>"
            f"<td>{r.get('avg_latency', 0):.2f}</td>"
            f"<td>{r.get('ttft_estimate', 0) * 1000:.0f}</td>"
            f"<td>{r.get('throughput_per_user', 0):.1f}</td>"
            f"<td>{r.get('successful', 0)}/{r.get('successful', 0) + r.get('failed', 0)}</td>"
            f"</tr>\n"
        )

    html += f"""</tbody></table></div>

<div class="footer">
  Generated by vLLM Benchmark Suite v3.0.0 &mdash; {datetime.now().isoformat()}
</div>

<script>
const results = {results_json};

// Group by concurrent_users
const userGroups = {{}};
results.forEach(r => {{
    const u = r.concurrent_users;
    if (!userGroups[u]) userGroups[u] = [];
    userGroups[u].push(r);
}});

const colors = ['#2E86AB','#A23B72','#F18F01','#C73E1D','#6A994E','#BC4B51'];
const layout_base = {{
    paper_bgcolor: '#1a1a2e', plot_bgcolor: '#1a1a2e',
    font: {{ color: '#e0e0e0', size: 12 }},
    xaxis: {{ gridcolor: '#333', title: 'Context Length (K tokens)' }},
    yaxis: {{ gridcolor: '#333' }},
    margin: {{ l: 60, r: 30, t: 50, b: 50 }}
}};

// Throughput chart
const tpTraces = Object.keys(userGroups).sort((a,b)=>a-b).map((u, i) => {{
    const data = userGroups[u].sort((a,b) => a.context_length - b.context_length);
    return {{
        x: data.map(r => r.context_length / 1000),
        y: data.map(r => r.tokens_per_second),
        mode: 'lines+markers', name: u + ' users',
        line: {{ color: colors[i % colors.length], width: 3 }},
        marker: {{ size: 10 }}
    }};
}});
Plotly.newPlot('throughput-chart', tpTraces, {{
    ...layout_base, title: 'Throughput vs Context Length',
    yaxis: {{ ...layout_base.yaxis, title: 'Tokens/sec' }}
}});

// TTFT chart
const ttftTraces = Object.keys(userGroups).sort((a,b)=>a-b).map((u, i) => {{
    const data = userGroups[u].sort((a,b) => a.context_length - b.context_length);
    return {{
        x: data.map(r => r.context_length / 1000),
        y: data.map(r => (r.ttft_estimate || 0) * 1000),
        mode: 'lines+markers', name: u + ' users',
        line: {{ color: colors[i % colors.length], width: 3 }},
        marker: {{ size: 10 }}
    }};
}});
Plotly.newPlot('ttft-chart', ttftTraces, {{
    ...layout_base, title: 'Time to First Token',
    yaxis: {{ ...layout_base.yaxis, title: 'TTFT (ms)' }},
    shapes: [
        {{ type:'rect', x0:0, x1:1, xref:'paper', y0:0, y1:200, fillcolor:'green', opacity:0.05, line:{{width:0}} }},
        {{ type:'rect', x0:0, x1:1, xref:'paper', y0:200, y1:1000, fillcolor:'yellow', opacity:0.05, line:{{width:0}} }},
        {{ type:'rect', x0:0, x1:1, xref:'paper', y0:1000, y1:3000, fillcolor:'orange', opacity:0.05, line:{{width:0}} }},
    ]
}});

// ITL chart
const itlTraces = Object.keys(userGroups).sort((a,b)=>a-b).map((u, i) => {{
    const data = userGroups[u].sort((a,b) => a.context_length - b.context_length);
    return {{
        x: data.map(r => r.context_length / 1000),
        y: data.map(r => {{
            const decode = r.decode_time_estimate || (r.avg_latency * 0.85);
            const comp = r.avg_completion_tokens || 1;
            return (decode / comp) * 1000;
        }}),
        mode: 'lines+markers', name: u + ' users',
        line: {{ color: colors[i % colors.length], width: 3 }},
        marker: {{ size: 10 }}
    }};
}});
Plotly.newPlot('itl-chart', itlTraces, {{
    ...layout_base, title: 'Inter-Token Latency',
    yaxis: {{ ...layout_base.yaxis, title: 'ITL (ms)' }}
}});

// Heatmaps
const contexts = [...new Set(results.map(r => r.context_length))].sort((a,b)=>a-b);
const users = [...new Set(results.map(r => r.concurrent_users))].sort((a,b)=>a-b);

function buildMatrix(metric) {{
    return contexts.map(ctx => users.map(u => {{
        const r = results.find(x => x.context_length === ctx && x.concurrent_users === u);
        return r ? r[metric] : null;
    }}));
}}

Plotly.newPlot('latency-heatmap', [{{
    z: buildMatrix('avg_latency'), x: users.map(String), y: contexts.map(c => (c/1000)+'K'),
    type: 'heatmap', colorscale: 'RdYlGn', reversescale: true,
    colorbar: {{ title: 'Seconds' }}
}}], {{ ...layout_base, title: 'Latency Heatmap', xaxis: {{ title: 'Concurrent Users' }}, yaxis: {{ title: 'Context' }} }});

Plotly.newPlot('throughput-heatmap', [{{
    z: buildMatrix('tokens_per_second'), x: users.map(String), y: contexts.map(c => (c/1000)+'K'),
    type: 'heatmap', colorscale: 'RdYlGn',
    colorbar: {{ title: 'Tok/s' }}
}}], {{ ...layout_base, title: 'Throughput Heatmap', xaxis: {{ title: 'Concurrent Users' }}, yaxis: {{ title: 'Context' }} }});
</script>
</body></html>"""

    filename.write_text(html, encoding="utf-8")
    return str(filename)
