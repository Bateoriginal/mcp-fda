# openFDA MCP server

An [MCP](https://modelcontextprotocol.io/) server that exposes [openFDA](https://open.fda.gov/) — the FDA's public API for drug, device, and food data — as **26 scenario-shaped tools** and **6 guided prompts** any MCP host (Claude Desktop, VS Code, Cursor, …) can call in natural language.

> **Disclaimer.** openFDA data is not validated for clinical use. Don't rely on it for medical decisions. See <https://open.fda.gov/license/>.

---

## Requirements

- macOS / Linux / Windows
- Python **3.12+** (managed automatically by `uv`)
- An [openFDA API key](https://open.fda.gov/apis/authentication/) — free, instant. Without one you still get 240 req/min and 1,000 req/day per IP, which is enough for trying things out.

## Setup

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/<your-handle>/openfda-mcp.git
cd openfda-mcp
uv sync                 # creates .venv and installs runtime + dev deps

# 3. Add your API key
cp config.example.json config.json
# then edit config.json and replace YOUR_KEY_HERE
```

You can also set `OPENFDA_API_KEY` as an environment variable instead of editing `config.json` — the server checks the env first. `config.json` is gitignored so your key never leaves your machine.

## Smoke test

```bash
uv run pytest                       # 43 tests, fully mocked, no network needed
uv run python fda_mcp_server.py     # starts the stdio server (Ctrl+C to exit)
```

---

## Connect it to an MCP host

### VS Code

A portable workspace config ships at [.vscode/mcp.json.example](.vscode/mcp.json.example). Copy it once:

```bash
cp .vscode/mcp.json.example .vscode/mcp.json
```

Reload the window — the `openfda` server appears in the MCP panel. The example uses `${workspaceFolder}` and a bare `uv` so it works anywhere `uv` is on your `PATH`.

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "openfda": {
      "command": "uv",
      "args": [
        "--directory", "/ABSOLUTE/PATH/TO/openfda-mcp",
        "run", "python", "fda_mcp_server.py"
      ]
    }
  }
}
```

If `uv` isn't on Claude's `PATH`, replace `"uv"` with the full path printed by `which uv`. Restart Claude Desktop.

### MCP Inspector

For a browser UI to try every tool by hand:

```bash
uv run mcp dev fda_mcp_server.py
```

---

## Tools

### Drug — basics

| Tool | Scenario |
|---|---|
| `search_drug_label` | Fetch package-insert / SPL label sections |
| `find_drug_recalls` | Recall enforcement reports, newest first |
| `lookup_drug_approval` | NDA / ANDA / BLA approval history + submissions |
| `list_drug_ndcs` | Every National Drug Code SKU for a brand |
| `summarize_adverse_events` | Server-side counts: top reactions, indications, seriousness, country, sex, by year |
| `recent_adverse_event_reports` | A few raw FAERS reports (trimmed) |
| `resolve_drug_identifiers` | Brand → generic, UNII, application_number, NDCs, manufacturers, pharm class |
| `build_drug_dossier` | One call → totals + samples across every drug endpoint |

### Drug — deep dive

| Tool | Scenario |
|---|---|
| `get_label_section` | Pull one labeled section verbatim (clinical_studies, mechanism_of_action, …) |
| `list_label_revisions` | Every SPL revision with effective date + version + set_id |
| `get_application_documents` | Submission PDFs (approval letter, label, medical review) per application |
| `summarize_drug_shortages` | Shortage status breakdown, sorted by most-recent change |
| `get_ndc_packaging` | Marketing dates, DEA schedule, full packaging[] per NDC |
| `get_recall_details` | Code info, distribution pattern, quantity, termination_date for one recall_number |
| `summarize_event_outcomes` | Death / hospitalization / 4 other seriousness facets + reporter qualification + country + action_taken |
| `event_demographics` | Age / sex / weight / reports-by-year facets |
| `concomitant_drugs` | Top co-reported drugs in FAERS (auto-strips brand + all synonyms) |
| `disproportionality_signal` | 2×2 ROR with 95% CI + signal flag for one drug × reaction |
| `lookup_substance` | UNII / synonyms / molecular formula via `other/substance` |

### Device

| Tool | Scenario |
|---|---|
| `search_device_510k` | Premarket notification clearances |
| `find_device_recalls` | Device recalls by name |
| `summarize_device_events` | MAUDE aggregates: event types, problems, manufacturers, by year |

### Utility

| Tool | Scenario |
|---|---|
| `list_openfda_endpoints` | Catalog of nouns and categories |
| `count_records` | Cardinality / facet count for any field on any endpoint |
| `raw_openfda_query` | Escape hatch — arbitrary `noun/category/search/count/sort/limit/skip` |
| `plan_drug_question` | Returns the recommended tool-call plan for a freeform question |

## Prompts

| Prompt | What it scripts |
|---|---|
| `drug_overview` | Resolve identifiers, summarize the label, list recent recalls and adverse-event signals |
| `recall_check` | Walk every recall for a brand and explain what happened |
| `safety_profile` | FAERS reactions + outcomes + demographics + co-reported drugs |
| `device_overview` | 510(k) clearances, recalls, and MAUDE events for a device family |
| `fda_application_brief` | 12-step deep-dive recipe across labels, approvals, NDCs, recalls, FAERS, substance |
| `fda_inventory` | Tour every endpoint to show what FDA actually publishes |

---

## How openFDA is organized

Every API call has the shape:

```
https://api.fda.gov/{noun}/{category}.json?{search|count|sort|limit|skip}
```

- **Noun** — `drug`, `device`, `food`, `animalandveterinary`, `tobacco`, `other`
- **Category** — depends on noun (e.g. `drug/event`, `drug/label`, `device/510k`, `device/enforcement`, `other/substance`)
- **`search=`** — Lucene query (e.g. `openfda.brand_name:"OZEMPIC"`)
- **`count=`** — server-side aggregation on a field (use the `.exact` suffix for full-string counts)
- **`limit`** ≤ 1000, **`skip`** ≤ 25,000

### `openfda.*` harmonization

Different datasets identify drugs differently (NDC vs RxCUI vs brand vs generic). openFDA attaches a normalized `openfda` sub-object on every record so you can search across endpoints with the same field names. The server is built around that idea: pick a stable identifier (`unii`, `application_number`, `product_ndc`) and the same query works on `label`, `event`, `enforcement`, `ndc`, `drugsfda`.

> **Gotcha.** On the `drug/event` endpoint, harmonized fields are nested under `patient.drug.openfda.*` rather than top-level `openfda.*`. The server handles that for you.

---

## What it does *not* cover

openFDA exposes labels, approvals, NDCs, recalls, shortages, FAERS, MAUDE, 510(k), and substance lookups — and that is what this server tools. The FDA does **not** publish the following through openFDA, so they are out of scope here:

- Orange Book therapeutic-equivalence codes
- REMS (Risk Evaluation and Mitigation Strategy) program details
- Post-market commitments (PMR / PMC) status
- Advisory committee transcripts and briefing books
- ClinicalTrials.gov data (that's NIH, not FDA)

---

## Project layout

```
openfda-mcp/
├── fda_mcp_server.py            # MCP server (FastMCP + httpx) — 26 tools, 6 prompts
├── config.example.json          # Copy to config.json and add your openFDA key
├── .vscode/mcp.json.example     # Copy to .vscode/mcp.json for VS Code MCP discovery
├── tests/                       # 43 unit tests, fully mocked with respx
│   ├── conftest.py
│   └── test_fda_mcp_server.py
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

---

## Tests

```bash
uv run pytest               # all tests
uv run pytest -v            # verbose
```

`respx` intercepts `httpx` calls so the suite needs no network and no API key.

---

## Example prompts to try in your MCP host

- "Pull the FDA label for Ozempic and summarize the boxed warning."
- "List every recall of semaglutide products since 2024."
- "What are the top 10 adverse reactions reported for Lipitor in FAERS?"
- "Resolve the brand Wegovy to its UNII and NDC numbers."
- "Find 510(k) clearances for insulin pumps from the last two years."
- "Compute the disproportionality signal for semaglutide and medullary thyroid cancer."
- "Show me every label revision for Ozempic and the clinical_studies section of the latest one."

The host picks the matching tool (`find_drug_recalls`, `summarize_adverse_events`, `disproportionality_signal`, …) and the server issues the right openFDA query.

---

## License & data terms

Code: [MIT](LICENSE). openFDA data is governed by the FDA's [terms of service](https://open.fda.gov/terms/) and [data license](https://open.fda.gov/license/).
