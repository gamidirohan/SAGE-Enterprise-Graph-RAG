import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'pilot_results.json'
OUT_JSON = ROOT / 'results' / 'pilot_full_results.json'
OUT_CSV = ROOT / 'results' / 'pilot_metrics.csv'
OUT_HTML = ROOT / 'results' / 'summary.html'

if not RESULTS.exists():
    print(f"Missing mock results at {RESULTS}")
    raise SystemExit(1)

with open(RESULTS, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Populate synthetic metrics
for entry in data:
    # similarity (0-1)
    sim = round(random.uniform(0.45, 0.95), 3)
    rouge_l = round(random.uniform(0.2, 0.9), 3)
    llm_score = round(random.uniform(4.0, 9.5), 2)
    entry['similarity'] = sim
    entry['rouge'] = {'rouge_l': rouge_l}
    entry['llm_evaluation'] = {'score_1_10': llm_score, 'preference': random.choice(['sage','traditional','tie'])}
    # grounding: if any cypher mentions 'doc-' assume exists
    for g in entry.get('grounding_checks', []):
        cy = g.get('cypher','')
        if 'doc-' in cy or 'doc_policy' in cy:
            g['exists'] = True
            g['error'] = None
        else:
            g['exists'] = False
            g['error'] = None

# Save augmented results
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

# Create CSV summary: query_index, similarity, rouge_l, llm_score
with open(OUT_CSV, 'w', encoding='utf-8') as f:
    f.write('query_index,query,similarity,rouge_l,llm_score,preference\n')
    for e in data:
        qi = e.get('query_index')
        q = e.get('query','').replace(',', ' ')
        sim = e.get('similarity')
        rouge_l = e.get('rouge',{}).get('rouge_l')
        llm = e.get('llm_evaluation',{}).get('score_1_10')
        pref = e.get('llm_evaluation',{}).get('preference')
        f.write(f"{qi},\"{q}\",{sim},{rouge_l},{llm},{pref}\n")

# Generate simple HTML with embedded data and Chart.js
chart_data = [{'index': e.get('query_index'), 'similarity': e.get('similarity'), 'rouge': e.get('rouge',{}).get('rouge_l'), 'llm': e.get('llm_evaluation',{}).get('score_1_10')} for e in data]

html = (
    '<!doctype html>\n'
    '<html>\n'
    '<head>\n'
    '  <meta charset="utf-8">\n'
    '  <title>Pilot Results Summary</title>\n'
    '  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n'
    '</head>\n'
    '<body>\n'
    '  <h2>Pilot Results Summary</h2>\n'
    '  <canvas id="similarityChart" width="800" height="300"></canvas>\n'
    '  <canvas id="scoresChart" width="800" height="300"></canvas>\n'
    '  <script>\n'
    '    const data = ' + json.dumps(chart_data) + ';\n'
    '    const labels = data.map(d=>d.index);\n'
    '    const similarity = data.map(d=>d.similarity);\n'
    '    const rouge = data.map(d=>d.rouge);\n'
    '    const llm = data.map(d=>d.llm);\n'
    '    new Chart(document.getElementById(\'similarityChart\'), {\n'
    "      type: 'line',\n"
    "      data: {\n"
    "        labels: labels,\n"
    "        datasets: [{label: 'Similarity', data: similarity, borderColor: 'blue', fill:false}]\n"
    "      }\n"
    "    });\n"
    '    new Chart(document.getElementById(\'scoresChart\'), {\n'
    "      type: 'bar',\n"
    "      data: {\n"
    "        labels: labels,\n"
    "        datasets: [\n"
    "          {label: 'ROUGE-L', data: rouge, backgroundColor: 'orange'},\n"
    "          {label: 'LLM Score', data: llm, backgroundColor: 'green'}\n"
    "        ]\n"
    "      },\n"
    "      options: {scales: {y: {beginAtZero: true}}}\n"
    "    });\n"
    '  </script>\n'
    '</body>\n'
    '</html>\n'
)

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print('Simulated full pilot results written to:', OUT_JSON)
print('CSV metrics written to:', OUT_CSV)
print('HTML summary written to:', OUT_HTML)
