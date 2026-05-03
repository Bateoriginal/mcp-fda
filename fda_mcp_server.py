"""
openFDA MCP server.

Exposes one MCP tool per common scenario, plus a raw escape hatch.

Run (stdio, for Claude Desktop / VS Code MCP integration):
    uv run python fda_mcp_server.py

Or test interactively:
    uv run mcp dev fda_mcp_server.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BASE_URL = "https://api.fda.gov"
USER_AGENT = "rhizome-fda-mcp/1.0"

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"


def _load_api_key() -> str | None:
    """Resolve the openFDA API key.

    Order of precedence: ``OPENFDA_API_KEY`` environment variable, then the
    ``openFDA_api_key`` field of ``config.json`` next to this file. Returns
    ``None`` if neither is set, in which case the server falls back to the
    anonymous (lower) rate limit.
    """
    # 1. environment variable wins
    if key := os.environ.get("OPENFDA_API_KEY"):
        return key
    # 2. fall back to config.json next to this file
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text()).get("openFDA_api_key")
        except Exception:
            return None
    return None


API_KEY = _load_api_key()

# Endpoint catalog (used for validation + the raw tool)
ENDPOINTS: dict[str, list[str]] = {
    "drug": ["event", "label", "ndc", "enforcement", "drugsfda", "shortages"],
    "device": [
        "event", "classification", "510k", "pma",
        "recall", "enforcement", "udi", "registrationlisting", "covid19serology",
    ],
    "food": ["enforcement", "event"],
    "animalandveterinary": ["event"],
    "tobacco": ["problem"],
    "other": ["historicaldocument", "nsde", "substance", "unii"],
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _fda_request(
    noun: str,
    category: str,
    params: dict[str, Any],
) -> dict:
    """Call api.fda.gov and return parsed JSON. Raises RuntimeError on failure."""
    if noun not in ENDPOINTS:
        raise RuntimeError(f"Unknown noun {noun!r}. Try one of {list(ENDPOINTS)}")
    if category not in ENDPOINTS[noun]:
        raise RuntimeError(
            f"Unknown category {category!r} for {noun!r}. "
            f"Try one of {ENDPOINTS[noun]}"
        )

    clean = {k: v for k, v in params.items() if v not in (None, "")}
    if API_KEY:
        clean = {"api_key": API_KEY, **clean}

    url = f"{BASE_URL}/{noun}/{category}.json?{urlencode(clean)}"
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": USER_AGENT}) as client:
        try:
            r = await client.get(url)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Network error: {e}") from None
        if r.status_code == 404:
            # openFDA uses 404 for "no results" — return an empty payload
            return {"meta": {"results": {"total": 0, "limit": 0, "skip": 0}}, "results": []}
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from {url}\n{r.text[:500]}")
        return r.json()


def _trim_meta(payload: dict) -> dict:
    """Strip the long disclaimer/license URLs from meta to keep responses small."""
    meta = payload.get("meta", {})
    return {
        "last_updated": meta.get("last_updated"),
        "total_matching": meta.get("results", {}).get("total"),
        "returned": len(payload.get("results", [])),
    }


def _drug_search(field: str, value: str, *, on_event_endpoint: bool = False) -> str:
    """Build a Lucene search clause for a harmonized drug field."""
    prefix = "patient.drug.openfda" if on_event_endpoint else "openfda"
    # Quote the value to handle multi-word brand names safely
    return f'{prefix}.{field}:"{value}"'


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

SERVER_INSTRUCTIONS = """\
openFDA MCP server. Use this server to answer questions about FDA-regulated
drugs and medical devices using the public openFDA API.

WHAT THIS SERVER COVERS
- Drug labels (package inserts), approval history (NDA/ANDA/BLA), National Drug
  Codes (NDC), recalls / enforcement reports, drug shortages, and FAERS
  adverse-event reports.
- Device 510(k) clearances, device recalls, and MAUDE adverse-event reports.

WHAT THIS SERVER DOES NOT COVER
- Clinical-trial outcomes, efficacy data, off-label use, prescribing guidelines,
  drug interactions, pricing, insurance coverage. Tell the user openFDA does
  not have this and suggest other sources (DailyMed, ClinicalTrials.gov,
  RxNorm, etc.).

IMPORTANT CAVEATS YOU MUST APPLY WHEN PRESENTING RESULTS
- FAERS adverse-event reports are SPONTANEOUS, voluntary submissions. They are
  NOT proof a drug caused a reaction and CANNOT be used to compute incidence
  rates ("X% of patients had Y"). Always say so when summarizing.
- A 404 from an endpoint means "no records," not an error. The server already
  normalizes this to an empty result set.
- openFDA data lags reality (often by a quarter or more). Don't claim it is
  real-time.
- Drug names: openFDA stores brand names UPPERCASE in the harmonized
  `openfda.brand_name` field (e.g. "OZEMPIC", not "Ozempic"). The server
  quotes values for you, but pass names as the user said them.

HOW TO ROUTE A USER REQUEST
1. If the user mentions any drug, ALWAYS call `plan_drug_question` FIRST. It
   returns the slots you still need to fill (drug name, identifier type,
   intent, timeframe) and the recommended tool sequence. Ask the user only
   for the slots that are missing.
2. If you don't know whether the user means a brand or a generic name, call
   `resolve_drug_identifiers` to disambiguate before doing anything else.
3. Prefer the specialized tools over `raw_openfda_query`. Use the raw tool
   only when no specialized tool fits.
4. For "side effects" questions, default to `summarize_adverse_events`
   (server-side counts) before fetching raw reports.
5. For "tell me everything about <drug>" questions, use `build_drug_dossier`
   plus `summarize_adverse_events`.

Useful pre-baked recipes are exposed as PROMPTS: drug-overview, recall-check,
safety-profile, device-overview. Suggest them when relevant.
"""

mcp = FastMCP("openfda", instructions=SERVER_INSTRUCTIONS)


# ----- DRUG TOOLS ----------------------------------------------------------

@mcp.tool()
async def search_drug_label(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name", "unii"] = "brand_name",
    limit: int = 3,
) -> dict:
    """Fetch FDA Structured Product Labels (package inserts) for a drug.

    WHEN TO USE: user asks for the official label, package insert, indications,
    contraindications, boxed warning, dosing instructions, or pharmacology.
    WHEN NOT TO USE: user asks about real-world side effects (use
    `summarize_adverse_events`) or recalls (use `find_drug_recalls`).
    ASK THE USER FIRST: drug name; whether they mean a brand ("Ozempic") or a
    generic ("semaglutide"); which section of the label they care about.

    Args:
        name: Drug name or identifier as the user said it (e.g. "OZEMPIC",
            "semaglutide"). Brand names are matched case-insensitively.
        match_field: Which harmonized field to match against. Use "unii" only
            when you have a real UNII code (10-char alphanumeric).
        limit: Max labels to return (1-10). Multiple labels appear when a drug
            has several approved forms or manufacturers.
    """
    payload = await _fda_request("drug", "label", {
        "search": _drug_search(match_field, name),
        "limit": max(1, min(limit, 10)),
    })
    return {
        "summary": _trim_meta(payload),
        "labels": payload.get("results", []),
    }


@mcp.tool()
async def find_drug_recalls(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name"] = "brand_name",
    limit: int = 25,
) -> dict:
    """Look up FDA recall enforcement reports for a drug, newest first.

    WHEN TO USE: user asks "has X been recalled?", "any recent recalls of X?",
    or "why was lot Y of X recalled?".
    WHEN NOT TO USE: user asks about adverse-event reports (use
    `summarize_adverse_events`); user asks about devices (use
    `find_device_recalls`).
    ASK THE USER FIRST: drug name (brand or generic); optional timeframe
    ("in the last year", "since 2024"). If the user gives a timeframe, mention
    you'll filter post-hoc — the API doesn't take a date range here, but
    results come back sorted newest-first so it's easy to truncate.
    NOTE: An empty result is normal and means "no recalls on file," not error.
    """
    payload = await _fda_request("drug", "enforcement", {
        "search": _drug_search(match_field, name),
        "sort": "recall_initiation_date:desc",
        "limit": max(1, min(limit, 100)),
    })
    results = [
        {
            "recall_initiation_date": r.get("recall_initiation_date"),
            "status": r.get("status"),
            "classification": r.get("classification"),
            "product_description": r.get("product_description"),
            "reason_for_recall": r.get("reason_for_recall"),
            "recalling_firm": r.get("recalling_firm"),
            "voluntary_mandated": r.get("voluntary_mandated"),
        }
        for r in payload.get("results", [])
    ]
    return {"summary": _trim_meta(payload), "recalls": results}


@mcp.tool()
async def lookup_drug_approval(name: str) -> dict:
    """Get FDA approval history (NDA/ANDA/BLA) for a drug brand name.

    WHEN TO USE: user asks "when was X approved?", "who makes X?", "what
    forms / dosages exist?", "what's the application number?", or wants the
    submission timeline (original approval + supplements).
    WHEN NOT TO USE: user wants the package insert (use `search_drug_label`)
    or just the SKU list (use `list_drug_ndcs`).
    ASK THE USER FIRST: brand name (this tool only matches on brand_name).
    For generics, call `resolve_drug_identifiers` first to find the brand(s).
    """
    payload = await _fda_request("drug", "drugsfda", {
        "search": _drug_search("brand_name", name),
        "limit": 10,
    })
    approvals = []
    for r in payload.get("results", []):
        approvals.append({
            "application_number": r.get("application_number"),
            "sponsor_name": r.get("sponsor_name"),
            "products": [
                {
                    "brand_name": p.get("brand_name"),
                    "active_ingredients": p.get("active_ingredients"),
                    "dosage_form": p.get("dosage_form"),
                    "route": p.get("route"),
                    "marketing_status": p.get("marketing_status"),
                }
                for p in r.get("products", [])
            ],
            "submissions": [
                {
                    "submission_type": s.get("submission_type"),
                    "submission_number": s.get("submission_number"),
                    "submission_status": s.get("submission_status"),
                    "submission_status_date": s.get("submission_status_date"),
                }
                for s in r.get("submissions", [])
            ],
        })
    return {"summary": _trim_meta(payload), "approvals": approvals}


@mcp.tool()
async def list_drug_ndcs(name: str, limit: int = 25) -> dict:
    """List National Drug Code (NDC) products for a brand name.

    WHEN TO USE: user wants every SKU / package size / strength of a drug, or
    needs the 10-digit product NDC for cross-referencing other systems.
    WHEN NOT TO USE: user wants a clinical summary (use `search_drug_label`).
    ASK THE USER FIRST: exact brand name. NDC matching is on the literal
    `brand_name` field (not the harmonized `openfda.brand_name`), so spelling
    matters more here than elsewhere.
    """
    payload = await _fda_request("drug", "ndc", {
        "search": f'brand_name:"{name}"',
        "limit": max(1, min(limit, 100)),
    })
    products = [
        {
            "product_ndc": r.get("product_ndc"),
            "brand_name": r.get("brand_name"),
            "generic_name": r.get("generic_name"),
            "labeler_name": r.get("labeler_name"),
            "dosage_form": r.get("dosage_form"),
            "route": r.get("route"),
            "marketing_category": r.get("marketing_category"),
            "active_ingredients": r.get("active_ingredients"),
            "packaging": [p.get("description") for p in r.get("packaging", [])],
        }
        for r in payload.get("results", [])
    ]
    return {"summary": _trim_meta(payload), "products": products}


@mcp.tool()
async def summarize_adverse_events(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name"] = "brand_name",
    top_n: int = 15,
) -> dict:
    """Server-side aggregation of FAERS adverse events for a drug.

    Returns top reactions, indications, seriousness breakdown, reporter
    countries, patient sex distribution, and reports per year.

    WHEN TO USE: user asks "what are the side effects of X?", "how often is
    X associated with Y?", "what's the safety profile of X?". This is the
    DEFAULT tool for adverse-event questions — it's a single fast call that
    summarizes thousands of reports server-side.
    WHEN NOT TO USE: user wants to read individual case narratives (use
    `recent_adverse_event_reports`); user wants causal/efficacy claims
    (FAERS cannot answer those).
    ASK THE USER FIRST: drug name; whether they want brand or generic
    coverage. Use generic (substance) for the broadest count, brand for a
    single product.

    PRESENT RESULTS WITH THESE CAVEATS:
    - Counts are voluntary reports, NOT incidence rates. "NAUSEA: 8,000"
      means 8,000 reports mentioned nausea, not that 8,000 patients had it.
    - The same patient can appear in multiple reports.
    - Reports include suspected, concomitant, and interacting drugs — the
      drug may not have caused the reaction.
    - Older drugs accumulate more reports simply because they've been on the
      market longer.
    """
    search = _drug_search(match_field, name, on_event_endpoint=True)
    facets = {
        "top_reactions":     "patient.reaction.reactionmeddrapt.exact",
        "top_indications":   "patient.drug.drugindication.exact",
        "seriousness":       "serious",
        "reporter_country":  "primarysource.reportercountry.exact",
        "patient_sex":       "patient.patientsex",
        "reports_by_year":   "receivedate",
    }

    # First call gives us the total count
    base = await _fda_request("drug", "event", {"search": search, "limit": 1})
    total = base.get("meta", {}).get("results", {}).get("total", 0)

    aggregates: dict[str, Any] = {}
    for label, field in facets.items():
        try:
            payload = await _fda_request("drug", "event", {
                "search": search,
                "count": field,
                "limit": 100 if label == "reports_by_year" else top_n,
            })
            aggregates[label] = payload.get("results", [])
        except RuntimeError as e:
            aggregates[label] = {"error": str(e).splitlines()[0]}

    return {
        "drug": name,
        "matched_on": match_field,
        "total_reports_in_faers": total,
        "aggregates": aggregates,
    }


@mcp.tool()
async def recent_adverse_event_reports(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name"] = "brand_name",
    limit: int = 5,
) -> dict:
    """Return a few recent raw FAERS adverse-event reports for inspection.

    WHEN TO USE: user wants to see actual case narratives or recent
    individual reports ("show me a recent serious report").
    WHEN NOT TO USE: user wants aggregate side-effect data — use
    `summarize_adverse_events` instead, it's far cheaper and more useful.
    ASK THE USER FIRST: drug name; how many reports they want (default 5,
    max 25); whether they only want serious reports.
    Same FAERS caveats apply (see `summarize_adverse_events`).
    """
    payload = await _fda_request("drug", "event", {
        "search": _drug_search(match_field, name, on_event_endpoint=True),
        "sort": "receivedate:desc",
        "limit": max(1, min(limit, 25)),
    })
    trimmed = []
    for r in payload.get("results", []):
        patient = r.get("patient", {})
        trimmed.append({
            "safetyreportid": r.get("safetyreportid"),
            "receivedate": r.get("receivedate"),
            "serious": r.get("serious"),
            "seriousnessdeath": r.get("seriousnessdeath"),
            "reporter_country": r.get("primarysource", {}).get("reportercountry"),
            "patient_age": patient.get("patientonsetage"),
            "patient_sex": patient.get("patientsex"),
            "reactions": [rx.get("reactionmeddrapt") for rx in patient.get("reaction", [])],
            "drugs": [
                {
                    "medicinalproduct": d.get("medicinalproduct"),
                    "indication": d.get("drugindication"),
                    "route": d.get("drugadministrationroute"),
                }
                for d in patient.get("drug", [])
            ],
        })
    return {"summary": _trim_meta(payload), "reports": trimmed}


@mcp.tool()
async def resolve_drug_identifiers(brand_name: str) -> dict:
    """Look up a brand name and return harmonized cross-dataset identifiers.

    Returns generic_name, UNII, application_number, NDCs, manufacturers,
    pharmacologic class — the IDs you can use to query other endpoints
    precisely.

    WHEN TO USE: as the FIRST step for any non-trivial drug question, when
    you need to disambiguate a brand vs generic, or when subsequent tools
    need a UNII / NDC / application number. Also useful when the user names
    a brand and you want to cover all generic equivalents.
    WHEN NOT TO USE: user already gave you the exact brand name and wants a
    single specific lookup (just call the lookup tool directly).
    ASK THE USER FIRST: the brand name as they said it. If they gave a
    generic name, search the label endpoint via `raw_openfda_query` instead.
    """
    payload = await _fda_request("drug", "label", {
        "search": _drug_search("brand_name", brand_name),
        "limit": 5,
    })
    bag: dict[str, set[str]] = {
        "brand_name": set(),
        "generic_name": set(),
        "manufacturer_name": set(),
        "substance_name": set(),
        "unii": set(),
        "product_ndc": set(),
        "application_number": set(),
        "rxcui": set(),
        "pharm_class_epc": set(),
        "pharm_class_moa": set(),
        "route": set(),
        "dosage_form": set(),
    }
    for rec in payload.get("results", []):
        of = rec.get("openfda", {})
        for key in bag:
            for v in of.get(key, []) or []:
                bag[key].add(str(v))
    return {
        "brand_name_query": brand_name,
        "matches": payload.get("meta", {}).get("results", {}).get("total", 0),
        "identifiers": {k: sorted(v) for k, v in bag.items()},
    }


@mcp.tool()
async def build_drug_dossier(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name", "unii"] = "brand_name",
    sample_events: int = 0,
) -> dict:
    """One-shot summary across every drug endpoint for a single medicine.

    Hits label, NDC, drugsfda, enforcement, shortages, and the FAERS event
    count in a single call and returns totals + a small sample per endpoint.

    WHEN TO USE: user asks "tell me everything about X," wants a quick
    overview, or you don't yet know which specific endpoint matters.
    WHEN NOT TO USE: user already specified an exact intent (label, recalls,
    approvals, etc.) — call the targeted tool, it's faster and trims output.
    For the side-effect breakdown you still need `summarize_adverse_events`
    afterward; this tool only returns the FAERS count, not the facets.
    ASK THE USER FIRST: drug name; brand vs generic; whether they want a
    sample of raw FAERS reports inline (`sample_events` > 0) or just counts.

    Args:
        name: The drug name or identifier.
        match_field: Which harmonized field to match.
        sample_events: How many raw adverse-event reports to include (0-25).
            For aggregated counts use `summarize_adverse_events` instead.
    """
    out: dict[str, Any] = {"drug": name, "matched_on": match_field, "endpoints": {}}

    for category in ["label", "ndc", "drugsfda", "enforcement", "shortages"]:
        try:
            page = await _fda_request("drug", category, {
                "search": _drug_search(match_field, name),
                "limit": 5,
            })
            out["endpoints"][category] = {
                "total": page.get("meta", {}).get("results", {}).get("total", 0),
                "sample": page.get("results", [])[:3],
            }
        except RuntimeError as e:
            out["endpoints"][category] = {"error": str(e).splitlines()[0]}

    # Adverse events: just the count by default
    try:
        first = await _fda_request("drug", "event", {
            "search": _drug_search(match_field, name, on_event_endpoint=True),
            "limit": max(0, min(sample_events, 25)) or 1,
        })
        out["endpoints"]["event"] = {
            "total": first.get("meta", {}).get("results", {}).get("total", 0),
            "sample": first.get("results", []) if sample_events > 0 else [],
            "tip": "Call summarize_adverse_events for top reactions / indications.",
        }
    except RuntimeError as e:
        out["endpoints"]["event"] = {"error": str(e).splitlines()[0]}

    return out


# ----- DEVICE TOOLS --------------------------------------------------------

@mcp.tool()
async def search_device_510k(
    query: str,
    match_field: Literal["device_name", "k_number", "applicant"] = "device_name",
    limit: int = 10,
) -> dict:
    """Search 510(k) premarket notification clearances.

    WHEN TO USE: user asks "is device X cleared?", "who got 510(k) clearance
    for Y?", or wants the K-number / decision date / product code.
    WHEN NOT TO USE: user asks about device safety problems (use
    `summarize_device_events`) or recalls (use `find_device_recalls`).
    ASK THE USER FIRST: device name OR an exact K-number
    (e.g. "K203510") OR the applicant company. Pick `match_field`
    accordingly.
    NOTE: 510(k) is the most common but not the only premarket pathway.
    PMA (Class III) clearances live in a separate endpoint; mention this if
    relevant.
    """
    if match_field == "device_name":
        search = f'device_name:"{query}"'
    else:
        search = f'{match_field}:"{query}"'
    payload = await _fda_request("device", "510k", {"search": search, "limit": limit})
    items = [
        {
            "k_number": r.get("k_number"),
            "device_name": r.get("device_name"),
            "applicant": r.get("applicant"),
            "decision_date": r.get("decision_date"),
            "decision_description": r.get("decision_description"),
            "product_code": r.get("product_code"),
            "device_class": r.get("openfda", {}).get("device_class"),
            "medical_specialty": r.get("openfda", {}).get("medical_specialty_description"),
        }
        for r in payload.get("results", [])
    ]
    return {"summary": _trim_meta(payload), "clearances": items}


@mcp.tool()
async def find_device_recalls(device_name: str, limit: int = 25) -> dict:
    """Look up device recalls (enforcement reports) by device name.

    WHEN TO USE: user asks "any recalls for device X?", "why was X recalled?".
    WHEN NOT TO USE: drug recalls (use `find_drug_recalls`); device adverse
    events / malfunctions (use `summarize_device_events`).
    ASK THE USER FIRST: device name. Empty result is normal.
    """
    payload = await _fda_request("device", "enforcement", {
        "search": f'openfda.device_name:"{device_name}"',
        "sort": "recall_initiation_date:desc",
        "limit": limit,
    })
    items = [
        {
            "recall_initiation_date": r.get("recall_initiation_date"),
            "classification": r.get("classification"),
            "product_description": r.get("product_description"),
            "reason_for_recall": r.get("reason_for_recall"),
            "recalling_firm": r.get("recalling_firm"),
        }
        for r in payload.get("results", [])
    ]
    return {"summary": _trim_meta(payload), "recalls": items}


@mcp.tool()
async def summarize_device_events(device_name: str, top_n: int = 15) -> dict:
    """Aggregate device adverse-event reports (MAUDE) by problem and manufacturer.

    WHEN TO USE: user asks about device malfunctions, injuries, or deaths
    associated with a device.
    WHEN NOT TO USE: drug side effects (use `summarize_adverse_events`).
    ASK THE USER FIRST: device name; whether they want a specific
    manufacturer.

    PRESENT RESULTS WITH THESE CAVEATS:
    - MAUDE reports are voluntary and unverified, just like FAERS.
    - Counts are reports, not patients or events.
    - Manufacturer-submitted vs voluntary reports have very different shapes.
    """
    search = f'device.openfda.device_name:"{device_name}"'
    base = await _fda_request("device", "event", {"search": search, "limit": 1})
    total = base.get("meta", {}).get("results", {}).get("total", 0)

    facets = {
        "top_event_types": "event_type",
        "top_problems": "device.device_report_product_code.exact",
        "top_manufacturers": "manufacturer_name.exact",
        "events_by_year": "date_received",
    }
    aggregates: dict[str, Any] = {}
    for label, field in facets.items():
        try:
            payload = await _fda_request("device", "event", {
                "search": search, "count": field,
                "limit": 100 if "year" in label else top_n,
            })
            aggregates[label] = payload.get("results", [])
        except RuntimeError as e:
            aggregates[label] = {"error": str(e).splitlines()[0]}
    return {
        "device": device_name,
        "total_reports": total,
        "aggregates": aggregates,
    }


# ----- ESCAPE HATCH --------------------------------------------------------

@mcp.tool()
async def raw_openfda_query(
    noun: Literal["drug", "device", "food", "animalandveterinary", "tobacco", "other"],
    category: str,
    search: str | None = None,
    count: str | None = None,
    sort: str | None = None,
    limit: int = 10,
    skip: int = 0,
) -> dict:
    """Run an arbitrary openFDA query when no specialized tool fits.

    WHEN TO USE: food recalls, animal & vet events, tobacco reports, or any
    field/endpoint combination not covered by the specialized tools above.
    Also useful for advanced Lucene queries (date ranges, AND/OR/NOT).
    WHEN NOT TO USE: any query a specialized tool already handles — those
    tools trim output and apply known-good defaults.
    REMINDER: on `drug/event`, harmonized fields are nested under
    `patient.drug.openfda.*`, not top-level `openfda.*`.

    Args:
        noun: drug | device | food | animalandveterinary | tobacco | other
        category: Endpoint within the noun (e.g. "label", "510k", "enforcement").
        search: Lucene search string (optional).
        count: Field to aggregate counts on; mutually useful with `search`.
        sort: e.g. "receivedate:desc".
        limit: 1-1000.
        skip: Offset for pagination (max 25,000).
    """
    payload = await _fda_request(noun, category, {
        "search": search, "count": count, "sort": sort,
        "limit": max(1, min(limit, 1000)),
        "skip": max(0, min(skip, 25_000)),
    })
    return payload


# ----- LIST ENDPOINTS (handy for discovery) --------------------------------

@mcp.tool()
async def list_openfda_endpoints() -> dict:
    """Return the catalog of openFDA nouns and their categories.

    WHEN TO USE: you (the model) need to remember which categories live under
    which noun before constructing a `raw_openfda_query`.
    WHEN NOT TO USE: as a user-facing answer — the user almost never wants
    the raw catalog.
    """
    return {"endpoints": ENDPOINTS, "base_url": BASE_URL}


@mcp.tool()
async def count_records(
    noun: Literal["drug", "device", "food", "animalandveterinary", "tobacco", "other"],
    category: str,
    search: str | None = None,
) -> dict:
    """Return total record count on a noun/category, optionally filtered.

    WHEN TO USE: any "how many", "total", or catalog-size question. Examples:
    "how many drugs are FDA-approved?", "how many Class III device recalls
    in 2024?", "how many semaglutide adverse-event reports?".
    WHEN NOT TO USE: when the user wants the actual records (use the
    specialized tool); when the user wants a per-field breakdown (use a
    `raw_openfda_query` with `count=...`).

    IMPORTANT — disclose the LAYER you are counting:
    - drug/drugsfda  = FDA application records (NDA/ANDA/BLA). Filter
      `submissions.submission_status:"AP"` to count applications with at
      least one approved submission.
    - drug/ndc       = product SKUs (one drug → many SKUs by strength /
      package). Inflates the count.
    - drug/label     = Structured Product Labels (multiple per drug, kept
      across revisions). Inflates the count further.
    - device/510k    = 510(k) submissions. Filter
      `decision_description:"substantially equivalent"` for actual
      clearances. Does NOT include PMA / De Novo / HDE pathways.
    - device/classification = device categories, not products.

    A 404 ("no results") is normalized to 0; this is not an error.
    """
    payload = await _fda_request(noun, category, {"search": search, "limit": 1})
    meta = payload.get("meta", {})
    return {
        "noun": noun,
        "category": category,
        "search": search,
        "total": meta.get("results", {}).get("total", 0),
        "last_updated": meta.get("last_updated"),
    }


# ----- DEEP-DIVE DRUG TOOLS (label, drugsfda, ndc, enforcement, shortages) --

# Sections we expose verbatim from drug/label. Anything in the SPL that the
# upstream parser surfaced is fair game; this list is the curated subset that
# answers the most common clinician / patient questions.
LABEL_SECTIONS = (
    "boxed_warning", "indications_and_usage", "contraindications",
    "warnings_and_cautions", "warnings", "precautions",
    "adverse_reactions", "drug_interactions",
    "dosage_and_administration", "dosage_forms_and_strengths",
    "how_supplied", "storage_and_handling",
    "clinical_pharmacology", "mechanism_of_action", "pharmacodynamics",
    "pharmacokinetics", "clinical_studies", "nonclinical_toxicology",
    "use_in_specific_populations", "pregnancy", "pediatric_use", "geriatric_use",
    "overdosage", "description", "references",
    "information_for_patients", "patient_information", "medication_guide",
    "recent_major_changes", "spl_product_data_elements",
)


@mcp.tool()
async def get_label_section(
    name: str,
    section: str,
    match_field: Literal["brand_name", "generic_name", "substance_name", "unii"] = "brand_name",
    label_index: int = 0,
) -> dict:
    """Return ONE label section verbatim (e.g. clinical_studies, mechanism_of_action).

    Use this when `search_drug_label` already gave a high-level view and the
    user wants to read a specific section in full — most often
    `clinical_studies` (where SUSTAIN / PIONEER / SOUL trial summaries live for
    Ozempic-like drugs), `clinical_pharmacology`, `mechanism_of_action`,
    `pharmacokinetics`, or `medication_guide`.

    WHEN TO USE: "show me the clinical-trial section of the X label,"
    "what's the mechanism of action," "what does the medication guide say."
    WHEN NOT TO USE: aggregate questions across the whole label —
    `search_drug_label` already returns the curated subset.
    ASK THE USER FIRST: drug name; section name (default the most-requested
    sections are listed in `LABEL_SECTIONS` returned by `list_label_revisions`).

    Args:
        name: Drug identifier as the user said it.
        section: SPL section key (e.g. `"clinical_studies"`). Unknown keys
            return an `available_sections` listing instead of an error.
        match_field: Which harmonized field to match.
        label_index: Which matching label to read (0 = first / latest). Use
            `list_label_revisions` to see all revisions.
    """
    payload = await _fda_request("drug", "label", {
        "search": _drug_search(match_field, name),
        "sort": "effective_time:desc",
        "limit": max(1, label_index + 1),
    })
    results = payload.get("results", [])
    if not results:
        return {"drug": name, "section": section, "error": "no labels found"}
    label = results[min(label_index, len(results) - 1)]
    if section not in label:
        return {
            "drug": name,
            "section": section,
            "error": "section not present in this label",
            "available_sections": sorted(
                k for k in label.keys() if k != "openfda" and not k.endswith("_table")
            ),
        }
    return {
        "drug": name,
        "section": section,
        "effective_time": label.get("effective_time"),
        "version": label.get("version"),
        "set_id": label.get("set_id"),
        "content": label.get(section),
    }


@mcp.tool()
async def list_label_revisions(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name", "unii"] = "brand_name",
) -> dict:
    """List every SPL revision on file for a drug (effective_time, version, set_id).

    WHEN TO USE: "how many label revisions does X have?", "when was the last
    label update?", "what set_id does DailyMed use for X?".
    WHEN NOT TO USE: you want to read a section — use `get_label_section`.
    ASK THE USER FIRST: drug name and identifier type.

    Returns one row per label record, plus `known_sections` (the union of
    section keys present across every revision) so the host knows what to
    pass to `get_label_section`.
    """
    payload = await _fda_request("drug", "label", {
        "search": _drug_search(match_field, name),
        "sort": "effective_time:desc",
        "limit": 100,
    })
    rows = []
    sections: set[str] = set()
    for r in payload.get("results", []):
        rows.append({
            "effective_time": r.get("effective_time"),
            "version": r.get("version"),
            "set_id": r.get("set_id"),
            "id": r.get("id"),
            "brand_names": r.get("openfda", {}).get("brand_name", []),
            "manufacturer": r.get("openfda", {}).get("manufacturer_name", []),
        })
        for k in r.keys():
            if k != "openfda" and not k.endswith("_table"):
                sections.add(k)
    return {
        "summary": _trim_meta(payload),
        "revisions": rows,
        "known_sections": sorted(sections),
    }


@mcp.tool()
async def get_application_documents(application_number: str) -> dict:
    """Return all FDA-hosted approval documents for an NDA/ANDA/BLA application.

    Surfaces every `submissions[].application_docs[]` entry in `drug/drugsfda`
    — these are the direct accessdata.fda.gov URLs to approval letters,
    Summary Basis of Approval, printed labels, medical / statistical reviews,
    and chemistry reviews.

    WHEN TO USE: "give me the FDA approval letter for NDA 209637", "where can
    I read the original SBA / medical review for X", "what supplements has
    application N had and what changed in each".
    WHEN NOT TO USE: you only need approval dates and dosage forms —
    `lookup_drug_approval` is lighter.
    ASK THE USER FIRST: the application number (NDA######, ANDA######,
    BLA######). If they only know the brand, call `lookup_drug_approval` first.
    """
    payload = await _fda_request("drug", "drugsfda", {
        "search": f'application_number:"{application_number}"',
        "limit": 5,
    })
    out = []
    for app in payload.get("results", []):
        submissions = []
        for s in app.get("submissions", []) or []:
            submissions.append({
                "submission_type": s.get("submission_type"),
                "submission_number": s.get("submission_number"),
                "submission_class_code": s.get("submission_class_code"),
                "submission_class_code_description": s.get("submission_class_code_description"),
                "submission_status": s.get("submission_status"),
                "submission_status_date": s.get("submission_status_date"),
                "review_priority": s.get("review_priority"),
                "application_docs": [
                    {
                        "id": d.get("id"),
                        "date": d.get("date"),
                        "type": d.get("type"),
                        "url": d.get("url"),
                    }
                    for d in s.get("application_docs", []) or []
                ],
            })
        out.append({
            "application_number": app.get("application_number"),
            "sponsor_name": app.get("sponsor_name"),
            "submissions": submissions,
        })
    return {"summary": _trim_meta(payload), "applications": out}


@mcp.tool()
async def summarize_drug_shortages(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name"] = "generic_name",
) -> dict:
    """Summarize FDA drug-shortage history for a drug.

    WHEN TO USE: "is X on the FDA shortage list?", "when did the X shortage
    start / resolve?", "why was X in shortage?".
    WHEN NOT TO USE: device shortages (not exposed by openFDA).
    ASK THE USER FIRST: drug name. **Generic name usually works best** because
    shortages are tracked by molecule, not brand.

    PRESENT RESULTS WITH THESE CAVEATS:
    - openFDA mirrors the FDA Drug Shortage Database, which lags real time.
    - Status values include "Currently in Shortage", "Resolved", "Discontinuation".
    - An empty result means "no shortage records on file," not "never in shortage."
    """
    payload = await _fda_request("drug", "shortages", {
        "search": _drug_search(match_field, name),
        "sort": "change_date:desc",
        "limit": 50,
    })
    rows = []
    statuses: Counter[str] = Counter()
    for r in payload.get("results", []):
        rows.append({
            "generic_name": r.get("generic_name"),
            "proprietary_name": r.get("proprietary_name"),
            "company_name": r.get("company_name"),
            "strength": r.get("strength"),
            "dosage_form": r.get("dosage_form"),
            "route": r.get("route"),
            "status": r.get("status"),
            "availability": r.get("availability"),
            "shortage_reason": r.get("shortage_reason"),
            "initial_posting_date": r.get("initial_posting_date"),
            "change_date": r.get("change_date"),
            "update_type": r.get("update_type"),
            "resolved_note": r.get("resolved_note"),
            "therapeutic_category": r.get("therapeutic_category"),
        })
        if r.get("status"):
            statuses[r["status"]] += 1
    return {
        "summary": _trim_meta(payload),
        "status_breakdown": dict(statuses),
        "records": rows,
    }


@mcp.tool()
async def get_ndc_packaging(name: str, limit: int = 50) -> dict:
    """Return marketing dates, package sizes, and DEA schedule for every NDC of a drug.

    Richer than `list_drug_ndcs`: includes `marketing_start_date`,
    `marketing_end_date`, `listing_expiration_date`, `dea_schedule`, and the
    full `packaging[]` array (carton size, package NDC, marketing dates per
    package).

    WHEN TO USE: "which strengths of X are still on the market vs
    discontinued?", "what package sizes does X ship in?", "is X a
    controlled substance?".
    WHEN NOT TO USE: high-level brand listing — `list_drug_ndcs` is lighter.
    ASK THE USER FIRST: exact brand name (literal `brand_name` match).
    """
    payload = await _fda_request("drug", "ndc", {
        "search": f'brand_name:"{name}"',
        "limit": max(1, min(limit, 100)),
    })
    rows = []
    for r in payload.get("results", []):
        rows.append({
            "product_ndc": r.get("product_ndc"),
            "brand_name": r.get("brand_name"),
            "generic_name": r.get("generic_name"),
            "labeler_name": r.get("labeler_name"),
            "product_type": r.get("product_type"),
            "dosage_form": r.get("dosage_form"),
            "route": r.get("route"),
            "marketing_category": r.get("marketing_category"),
            "marketing_start_date": r.get("marketing_start_date"),
            "marketing_end_date": r.get("marketing_end_date"),
            "listing_expiration_date": r.get("listing_expiration_date"),
            "dea_schedule": r.get("dea_schedule"),
            "finished": r.get("finished"),
            "active_ingredients": r.get("active_ingredients"),
            "packaging": [
                {
                    "package_ndc": p.get("package_ndc"),
                    "description": p.get("description"),
                    "marketing_start_date": p.get("marketing_start_date"),
                    "marketing_end_date": p.get("marketing_end_date"),
                    "sample": p.get("sample"),
                }
                for p in r.get("packaging", []) or []
            ],
        })
    return {"summary": _trim_meta(payload), "products": rows}


@mcp.tool()
async def get_recall_details(recall_number: str) -> dict:
    """Return the full enforcement record for a single recall, including lot codes.

    Includes `code_info` (lot numbers / expiry), `distribution_pattern` (which
    states / countries), `product_quantity`, `recall_initiation_date`,
    `termination_date`, `event_id`, `more_code_info`, and the full product
    description — i.e., the fields a pharmacist actually needs to decide
    whether their stock is affected.

    WHEN TO USE: "is lot ABC123 affected by recall Z?", "where was recall Z
    distributed?", "how many units were recalled?".
    WHEN NOT TO USE: you don't yet know the recall_number — call
    `find_drug_recalls` or `find_device_recalls` first.
    """
    # Drug enforcement first; fall back to device enforcement.
    for noun in ("drug", "device"):
        payload = await _fda_request(noun, "enforcement", {
            "search": f'recall_number:"{recall_number}"',
            "limit": 1,
        })
        results = payload.get("results", [])
        if results:
            r = results[0]
            return {
                "noun": noun,
                "summary": _trim_meta(payload),
                "recall": {
                    "recall_number": r.get("recall_number"),
                    "event_id": r.get("event_id"),
                    "status": r.get("status"),
                    "classification": r.get("classification"),
                    "recall_initiation_date": r.get("recall_initiation_date"),
                    "report_date": r.get("report_date"),
                    "termination_date": r.get("termination_date"),
                    "recalling_firm": r.get("recalling_firm"),
                    "voluntary_mandated": r.get("voluntary_mandated"),
                    "product_description": r.get("product_description"),
                    "product_quantity": r.get("product_quantity"),
                    "reason_for_recall": r.get("reason_for_recall"),
                    "code_info": r.get("code_info"),
                    "more_code_info": r.get("more_code_info"),
                    "distribution_pattern": r.get("distribution_pattern"),
                    "country": r.get("country"),
                    "state": r.get("state"),
                    "city": r.get("city"),
                    "address_1": r.get("address_1"),
                },
            }
    return {"noun": None, "recall": None, "error": "no enforcement record found for that recall_number"}


# ----- DEEP-DIVE FAERS TOOLS -----------------------------------------------

@mcp.tool()
async def summarize_event_outcomes(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name"] = "brand_name",
) -> dict:
    """Break down FAERS reports for a drug by outcome type and reporter qualification.

    Returns counts for: death, life-threatening, hospitalization, disability,
    congenital anomaly, required intervention, other-serious, plus reporter
    qualification (physician / pharmacist / other HCP / consumer / lawyer)
    and country-level distribution.

    WHEN TO USE: "how many Ozempic FAERS reports involved death?", "are most
    reports from doctors or consumers?", "what's the seriousness mix?".
    WHEN NOT TO USE: top reactions — use `summarize_adverse_events`.
    Same FAERS caveats apply: voluntary, unverified, not incidence rates.
    """
    search = _drug_search(match_field, name, on_event_endpoint=True)
    base = await _fda_request("drug", "event", {"search": search, "limit": 1})
    total = base.get("meta", {}).get("results", {}).get("total", 0)

    facets = {
        "serious":                       "serious",
        "seriousnessdeath":              "seriousnessdeath",
        "seriousnesslifethreatening":    "seriousnesslifethreatening",
        "seriousnesshospitalization":    "seriousnesshospitalization",
        "seriousnessdisabling":          "seriousnessdisabling",
        "seriousnesscongenitalanomali":  "seriousnesscongenitalanomali",
        "seriousnessother":              "seriousnessother",
        "reporter_qualification":        "primarysource.qualification",
        "reporter_country":              "primarysource.reportercountry.exact",
        "action_taken":                  "patient.drug.actiondrug",
        "drug_characterization":         "patient.drug.drugcharacterization",
    }
    aggregates: dict[str, Any] = {}
    for label, field in facets.items():
        try:
            payload = await _fda_request("drug", "event", {
                "search": search, "count": field, "limit": 25,
            })
            aggregates[label] = payload.get("results", [])
        except RuntimeError as e:
            aggregates[label] = {"error": str(e).splitlines()[0]}
    return {
        "drug": name,
        "matched_on": match_field,
        "total_reports_in_faers": total,
        "aggregates": aggregates,
        "legend": {
            "serious": "1 = serious, 2 = non-serious",
            "seriousness*": "1 = yes (this category applied)",
            "primarysource.qualification": "1=physician, 2=pharmacist, 3=other HCP, 4=lawyer, 5=consumer/non-HCP",
            "actiondrug": "1=withdrawn, 2=dose reduced, 3=dose increased, 4=dose unchanged, 5=unknown, 6=not applicable",
            "drugcharacterization": "1=suspect, 2=concomitant, 3=interacting",
        },
    }


@mcp.tool()
async def event_demographics(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name"] = "brand_name",
) -> dict:
    """Return age / sex / weight / onset-year distributions for a drug's FAERS reports.

    WHEN TO USE: "who's reporting Ozempic events — what age, sex, weight?".
    WHEN NOT TO USE: top reactions or outcomes — use the dedicated tools.
    Same FAERS caveats.
    """
    search = _drug_search(match_field, name, on_event_endpoint=True)
    facets = {
        "patient_sex":     "patient.patientsex",
        "patient_age":     "patient.patientonsetage",
        "patient_age_unit":"patient.patientonsetageunit",
        "patient_weight":  "patient.patientweight",
        "reports_by_year": "receivedate",
    }
    aggregates: dict[str, Any] = {}
    for label, field in facets.items():
        try:
            payload = await _fda_request("drug", "event", {
                "search": search, "count": field, "limit": 100,
            })
            aggregates[label] = payload.get("results", [])
        except RuntimeError as e:
            aggregates[label] = {"error": str(e).splitlines()[0]}
    return {
        "drug": name,
        "matched_on": match_field,
        "aggregates": aggregates,
        "legend": {
            "patientsex": "0=unknown, 1=male, 2=female",
            "patientonsetageunit": "800=decade, 801=year, 802=month, 803=week, 804=day, 805=hour",
        },
    }


@mcp.tool()
async def concomitant_drugs(
    name: str,
    match_field: Literal["brand_name", "generic_name", "substance_name"] = "brand_name",
    top_n: int = 25,
) -> dict:
    """Top OTHER drugs co-reported with the target drug in FAERS.

    Useful as a CHEAP proxy for "what's commonly taken with X" — NOT a
    drug-interaction analysis. The query simply counts every
    `patient.drug.openfda.generic_name` value in reports that also mention
    the target drug, then strips the target itself from the result.

    WHEN TO USE: "what other drugs do Ozempic patients commonly report?",
    "are there any unusual co-medications in semaglutide FAERS reports?".
    WHEN NOT TO USE: clinical drug-interaction questions — redirect to
    DailyMed / Lexicomp.
    Same FAERS caveats apply.
    """
    search = _drug_search(match_field, name, on_event_endpoint=True)
    payload = await _fda_request("drug", "event", {
        "search": search,
        "count": "patient.drug.openfda.generic_name.exact",
        "limit": max(top_n + 10, 30),
    })

    # Build the self-strip set: brand + every generic / substance the resolver
    # finds for that brand. One extra request, but makes the result usable.
    strip = {name.upper()}
    try:
        resolved = await resolve_drug_identifiers(name)
        for k in ("brand_name", "generic_name", "substance_name"):
            for v in resolved["identifiers"].get(k, []) or []:
                strip.add(v.upper())
    except Exception:
        pass

    rows = [
        r for r in payload.get("results", [])
        if r.get("term", "").upper() not in strip
    ][:top_n]
    return {
        "drug": name,
        "matched_on": match_field,
        "stripped_self_terms": sorted(strip),
        "co_reported_drugs": rows,
        "caveat": "Co-reporting in FAERS does not imply interaction or causation.",
    }


@mcp.tool()
async def disproportionality_signal(
    drug: str,
    reaction: str,
    match_field: Literal["brand_name", "generic_name", "substance_name"] = "generic_name",
) -> dict:
    """Compute a Reporting Odds Ratio (ROR) for one drug-reaction pair in FAERS.

    Builds the standard 2x2 contingency table:

        |                | reaction | other reactions |
        | drug           |    a     |       b         |
        | other drugs    |    c     |       d         |

    and returns ROR = (a*d) / (b*c) with a 95% confidence interval.

    A signal of disproportionate reporting is conventionally flagged when the
    LOWER bound of the 95% CI exceeds 1 AND a >= 3. This is a SCREENING tool,
    not a causal conclusion.

    WHEN TO USE: "is pulmonary aspiration over-reported with Ozempic?",
    "signal-mining for X+Y in FAERS".
    WHEN NOT TO USE: anything where the user expects a clinical answer.
    Always disclose the limitations: notoriety bias, channeling bias,
    indication confounding, no denominator.

    Args:
        drug: Drug name.
        reaction: MedDRA preferred term, case insensitive (e.g.
            `"pulmonary aspiration"`, `"medullary thyroid carcinoma"`).
        match_field: Field to match the drug on.
    """
    drug_clause = _drug_search(match_field, drug, on_event_endpoint=True)
    rxn_exact = f'patient.reaction.reactionmeddrapt.exact:"{reaction.upper()}"'
    rxn_loose = f'patient.reaction.reactionmeddrapt:"{reaction}"'

    async def _count(search: str) -> int:
        payload = await _fda_request("drug", "event", {"search": search, "limit": 1})
        return payload.get("meta", {}).get("results", {}).get("total", 0)

    # Try the strict .exact match first; if FAERS has zero rows for that exact
    # MedDRA PT spelling, retry with the looser non-exact match so users don't
    # have to know the exact casing.
    rxn_clause = rxn_exact
    rxn_total = await _count(rxn_clause)
    if rxn_total == 0:
        rxn_clause = rxn_loose
        rxn_total = await _count(rxn_clause)

    a = await _count(f"{drug_clause} AND {rxn_clause}")
    drug_total = await _count(drug_clause)
    grand_total = await _count("_exists_:safetyreportid")

    b = max(drug_total - a, 0)
    c = max(rxn_total - a, 0)
    d = max(grand_total - a - b - c, 0)

    out: dict[str, Any] = {
        "drug": drug,
        "reaction": reaction,
        "contingency": {"a": a, "b": b, "c": c, "d": d,
                         "drug_total": drug_total, "reaction_total": rxn_total,
                         "grand_total": grand_total},
        "ror": None,
        "ror_ci_95": None,
        "signal": False,
        "caveat": (
            "ROR is a screening signal only. Disproportionality ≠ causation. "
            "FAERS suffers from notoriety, channeling, and indication biases."
        ),
    }
    if a == 0 or b == 0 or c == 0 or d == 0:
        out["note"] = "Cannot compute ROR — at least one cell is zero."
        return out
    ror = (a * d) / (b * c)
    se_log = math.sqrt(1/a + 1/b + 1/c + 1/d)
    log_ror = math.log(ror)
    lo = math.exp(log_ror - 1.96 * se_log)
    hi = math.exp(log_ror + 1.96 * se_log)
    out["ror"] = round(ror, 3)
    out["ror_ci_95"] = [round(lo, 3), round(hi, 3)]
    out["signal"] = lo > 1 and a >= 3
    return out


# ----- REFERENCE-DATA TOOLS ------------------------------------------------

@mcp.tool()
async def lookup_substance(query: str) -> dict:
    """Look up a chemical substance by UNII or name in `other/substance`.

    Returns the canonical chemistry: UNII, registry name, IUPAC name,
    molecular formula, average molecular weight, code system mappings
    (CAS, INN, USAN, ChEMBL where available), and parent / derivative
    substance graph.

    WHEN TO USE: "what's the molecular formula of semaglutide?", "what's the
    UNII for X?", "give me the canonical chemistry record".
    WHEN NOT TO USE: drug product / dosage questions — use `drug/*` tools.

    Args:
        query: A UNII (10-char alphanumeric, e.g. `"53AXN4NNHX"`) or a
            substance name (matched on `names.name.exact`).
    """
    is_unii = len(query) == 10 and query.isalnum() and query.isupper()
    search = (
        f'unii:"{query}"' if is_unii
        else f'names.name.exact:"{query.upper()}"'
    )
    payload = await _fda_request("other", "substance", {"search": search, "limit": 5})
    return {"summary": _trim_meta(payload), "substances": payload.get("results", [])}


# ----- INTAKE / ROUTER -----------------------------------------------------

# Per-intent question plans. Each entry tells the host which slots to fill
# before doing real work, and which tool sequence to run once they are filled.
_INTENT_PLAN: dict[str, dict[str, Any]] = {
    "label": {
        "description": "User wants the official label / package insert.",
        "ask": [
            "drug_name",
            "identifier_type (brand vs generic vs UNII)",
            "section of interest (indications, warnings, dosing, ...) — optional",
        ],
        "tools": ["resolve_drug_identifiers (if brand/generic unclear)", "search_drug_label"],
    },
    "recall": {
        "description": "User wants to know if a drug has been recalled and why.",
        "ask": [
            "drug_name",
            "identifier_type (brand vs generic)",
            "timeframe — optional (results are sorted newest-first; you can truncate post-hoc)",
        ],
        "tools": ["find_drug_recalls"],
        "note": "Empty result = no recalls on file, not an error.",
    },
    "approval": {
        "description": "User wants the FDA approval history (NDA/ANDA/BLA, sponsor, dates, dosage forms).",
        "ask": [
            "brand_name (this endpoint matches on brand only)",
        ],
        "tools": ["resolve_drug_identifiers (if user gave a generic)", "lookup_drug_approval"],
    },
    "ndc": {
        "description": "User wants the SKU list / package sizes / strengths.",
        "ask": ["exact brand_name"],
        "tools": ["list_drug_ndcs"],
    },
    "side_effects_summary": {
        "description": "User wants aggregate adverse-event data (top reactions, seriousness, by year).",
        "ask": [
            "drug_name",
            "identifier_type — generic gives broadest coverage, brand narrows to one product",
            "top_n (default 15) — optional",
        ],
        "tools": ["summarize_adverse_events"],
        "must_disclose": [
            "FAERS counts are voluntary reports, NOT incidence rates.",
            "Reports include suspected, concomitant, and interacting drugs.",
            "Older drugs accumulate more reports just from being on market longer.",
        ],
    },
    "side_effects_recent": {
        "description": "User wants to read recent individual adverse-event reports.",
        "ask": [
            "drug_name",
            "identifier_type",
            "limit (1-25, default 5)",
            "serious-only? — optional",
        ],
        "tools": ["recent_adverse_event_reports"],
        "must_disclose": ["Same FAERS caveats as the summary tool."],
    },
    "shortages": {
        "description": "User wants drug-shortage status.",
        "ask": ["drug_name (generic name often works best for shortages)"],
        "tools": ["summarize_drug_shortages"],
        "note": "Many drugs return no results. That means no current shortage on file.",
    },
    "pharmacovigilance": {
        "description": "User wants deep FAERS analytics: outcomes, demographics, concomitant drugs, or signal detection.",
        "ask": [
            "drug_name and identifier_type",
            "specific axis: outcomes (death/hospitalization), demographics (age/sex/weight), co-reported drugs, OR a specific reaction to test for disproportionality",
        ],
        "tools": [
            "summarize_event_outcomes",
            "event_demographics",
            "concomitant_drugs",
            "disproportionality_signal (only when user names a specific reaction)",
        ],
        "must_disclose": [
            "FAERS is voluntary, unverified, and cannot give incidence rates.",
            "ROR / disproportionality is a screening signal, not proof of causation.",
            "Notoriety bias inflates counts for media-covered drugs.",
        ],
    },
    "label_section": {
        "description": "User wants a SPECIFIC label section verbatim (e.g. clinical_studies, mechanism_of_action, medication_guide).",
        "ask": [
            "drug_name",
            "section name (default options: clinical_studies, mechanism_of_action, pharmacokinetics, medication_guide, etc.)",
        ],
        "tools": ["list_label_revisions (to see available sections)", "get_label_section"],
    },
    "approval_documents": {
        "description": "User wants the actual FDA-hosted approval letters / SBA / medical reviews PDFs.",
        "ask": [
            "application_number (NDA######, ANDA######, BLA######) — if missing, run lookup_drug_approval first",
        ],
        "tools": ["lookup_drug_approval (to find application_number)", "get_application_documents"],
    },
    "recall_lot": {
        "description": "User wants the lot codes / distribution / quantity of a specific recall.",
        "ask": ["recall_number (if missing, run find_drug_recalls or find_device_recalls first)"],
        "tools": ["find_drug_recalls", "get_recall_details"],
    },
    "chemistry": {
        "description": "User wants chemical / substance reference data (UNII, formula, MW, CAS).",
        "ask": ["substance name OR UNII"],
        "tools": ["lookup_substance"],
    },
    "dossier": {
        "description": "User wants 'everything about drug X'.",
        "ask": [
            "drug_name",
            "identifier_type",
            "include sample FAERS reports? (sample_events 0-25)",
        ],
        "tools": ["resolve_drug_identifiers", "build_drug_dossier", "summarize_adverse_events"],
    },
    "device_clearance": {
        "description": "User wants 510(k) premarket clearance info for a device.",
        "ask": [
            "device_name OR k_number OR applicant",
        ],
        "tools": ["search_device_510k"],
        "note": "PMA (Class III) clearances live in a separate endpoint not yet wrapped.",
    },
    "device_recall": {
        "description": "User wants device recall info.",
        "ask": ["device_name"],
        "tools": ["find_device_recalls"],
    },
    "device_safety": {
        "description": "User wants device adverse-event / malfunction data (MAUDE).",
        "ask": ["device_name", "manufacturer — optional"],
        "tools": ["summarize_device_events"],
        "must_disclose": ["MAUDE reports are voluntary and unverified."],
    },
    "catalog_stats": {
        "description": "User wants catalog-level counts (e.g. 'how many drugs are approved?', 'how many 510(k) clearances?').",
        "ask": [
            "which layer they care about: applications, products (SKUs), or labels for drugs; 510(k) vs PMA for devices",
            "optional filter: status, year, manufacturer",
        ],
        "tools": ["count_records (use multiple times for the right layers)"],
        "must_disclose": [
            "openFDA does not have a single 'approved drugs' number. Report each layer with its definition.",
            "For drugs, the closest answer is drug/drugsfda filtered to submissions.submission_status:\"AP\" (~25k applications).",
            "For devices, 510(k) clearances dominate but PMA / De Novo / HDE pathways are separate endpoints.",
            "Always cite the meta.last_updated date so the count is reproducible.",
        ],
        "recipe": "See the `fda_inventory` prompt for a balanced answer template.",
    },
}

_OUT_OF_SCOPE = [
    ("efficacy / clinical-trial outcomes", "ClinicalTrials.gov"),
    ("drug interactions", "DailyMed, RxNorm, Lexicomp"),
    ("pricing / insurance coverage", "GoodRx, payer formularies"),
    ("off-label use guidelines", "UpToDate, professional society guidelines"),
    ("dosing calculators", "Lexicomp, Micromedex"),
]

# Cheap keyword router. Good enough as a first pass; the model can override.
_INTENT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("catalog_stats", [
        "how many", "total number", "count of", "how big",
        "approved drugs", "approved devices",
        "cleared devices", "510(k) clearances", "510k clearances",
        "fda inventory", "fda catalog",
    ]),
    ("approval_documents", [
        "approval letter", "approval letters", "sba", "summary basis of approval",
        "medical review", "statistical review", "chemistry review", "approval pdf",
        "application document", "application docs",
    ]),
    ("recall_lot", ["lot number", "lot code", "distribution pattern", "recall number", "affected lot"]),
    ("label_section", [
        "clinical studies", "clinical trial section", "mechanism of action",
        "pharmacokinetics", "pharmacodynamics", "medication guide", "patient information",
        "sustain", "pioneer", "soul trial",
    ]),
    ("pharmacovigilance", [
        "death", "deaths", "hospitaliz", "life-threatening", "life threatening",
        "disability", "reporter qualification", "physician report", "consumer report",
        "co-reported", "concomitant", "interaction signal", "signal detection",
        "disproportionality", "reporting odds ratio", "ror", "prr",
        "demographic", "age distribution", "weight distribution",
    ]),
    ("chemistry", [
        "unii", "molecular formula", "molecular weight", "cas number", "iupac",
        "substance record", "chemistry of",
    ]),
    ("recall", ["recall", "withdrawn", "pulled from market"]),
    ("approval", ["approved", "approval", "fda cleared", "nda", "anda", "bla", "when did fda"]),
    ("ndc", ["ndc", "sku", "package size", "strength", "national drug code", "dea schedule", "controlled substance"]),
    ("side_effects_recent", ["case report", "individual report", "case narrative", "show me reports"]),
    ("side_effects_summary", ["side effect", "adverse", "reaction", "safety", "faers", "seriousness"]),
    ("shortages", ["shortage", "out of stock", "supply"]),
    ("label", ["label", "package insert", "indication", "warning", "boxed warning", "dosing", "contraindication"]),
    ("device_recall", ["device recall"]),
    ("device_clearance", ["510(k)", "510k", "premarket", "k number", "k-number", "device clearance"]),
    ("device_safety", ["device adverse", "malfunction", "maude", "device problem"]),
    ("dossier", ["everything about", "tell me about", "overview", "summary of", "everything fda knows"]),
]


def _guess_intents(question: str) -> list[str]:
    q = question.lower()
    matches = [intent for intent, kws in _INTENT_KEYWORDS if any(kw in q for kw in kws)]
    return matches or []


@mcp.tool()
async def plan_drug_question(question: str) -> dict:
    """Routing & intake helper. Call this FIRST whenever the user asks a drug
    or device question.

    Returns:
        intents: best-guess intent labels matched from the question.
        plans: for each guessed intent, the slots you still need from the user
            and the tool sequence to run once they are filled.
        global_caveats: things you must disclose when presenting any result.
        out_of_scope: topics openFDA cannot answer, with redirect suggestions.
        clarifying_questions: a short list to ask the user verbatim if the
            request is ambiguous.

    The intent guess is a cheap keyword match. You (the model) are free to
    override it based on the conversation. Do not invent slots that aren't
    listed here; if the user already gave you a slot, don't ask again.
    """
    intents = _guess_intents(question)
    if not intents:
        # No keyword hit: ask the user to pick.
        return {
            "intents": [],
            "plans": {},
            "global_caveats": _GLOBAL_CAVEATS,
            "out_of_scope": [{"topic": t, "try_instead": s} for t, s in _OUT_OF_SCOPE],
            "clarifying_questions": [
                "Which of these would help most: the FDA label, recalls, approval history, "
                "side-effect summary, recent case reports, drug shortages, or a full overview?",
                "Do you have a specific brand name (e.g. \"Ozempic\") or a generic name "
                "(e.g. \"semaglutide\") in mind?",
            ],
            "available_intents": list(_INTENT_PLAN),
        }

    return {
        "intents": intents,
        "plans": {i: _INTENT_PLAN[i] for i in intents if i in _INTENT_PLAN},
        "global_caveats": _GLOBAL_CAVEATS,
        "out_of_scope": [{"topic": t, "try_instead": s} for t, s in _OUT_OF_SCOPE],
        "clarifying_questions": [
            "Brand name or generic name?",
            "Anything specific you care about (timeframe, manufacturer, route of administration)?",
        ],
    }


_GLOBAL_CAVEATS = [
    "FAERS / MAUDE reports are voluntary, unverified, and cannot be used to "
    "compute incidence rates.",
    "openFDA data lags reality \u2014 often by a quarter or more.",
    "A 404 from an endpoint means \"no records,\" not an error.",
    "openFDA has no efficacy / clinical-trial / pricing / interaction data \u2014 "
    "redirect users to other sources for those.",
]


# ----- PROMPTS (slash-command recipes) -------------------------------------

@mcp.prompt()
def drug_overview(drug_name: str, identifier_type: str = "brand") -> str:
    """Recipe: produce a balanced overview of a drug across every openFDA endpoint."""
    return (
        f"Give me an overview of the drug \"{drug_name}\" (treat as a {identifier_type} name).\n\n"
        "Steps:\n"
        "1. Call `resolve_drug_identifiers` to confirm spelling and pull harmonized IDs.\n"
        "2. Call `build_drug_dossier` with the resolved name.\n"
        "3. Call `summarize_adverse_events` for the top reactions and seriousness mix.\n"
        "4. Present a single short writeup with these sections:\n"
        "   - What it is (generic name, sponsor, approval year, dosage forms)\n"
        "   - Common uses (from the label indications)\n"
        "   - Most-reported adverse events (with the FAERS caveat verbatim)\n"
        "   - Recent recalls, if any\n"
        "   - Current shortage status, if any\n"
        "5. Always add: \"openFDA data is not validated for clinical use.\""
    )


@mcp.prompt()
def recall_check(drug_name: str, since_year: int | None = None) -> str:
    """Recipe: check whether a drug has been recalled and summarize the reasons."""
    window = f" since {since_year}" if since_year else ""
    return (
        f"Has \"{drug_name}\" been recalled{window}?\n\n"
        "Steps:\n"
        "1. Call `find_drug_recalls` with the drug name.\n"
        "2. If results come back, group them by `recalling_firm` and `reason_for_recall`.\n"
        "3. Present the most recent few with date, classification (Class I/II/III), and reason.\n"
        "4. If no results, say \"openFDA has no recall records on file for this drug\" \u2014 "
        "do not say there has never been a recall."
    )


@mcp.prompt()
def safety_profile(drug_name: str, identifier_type: str = "generic") -> str:
    """Recipe: build a FAERS-based safety profile with the right caveats baked in."""
    return (
        f"Build a FAERS safety profile for \"{drug_name}\" (treat as {identifier_type}).\n\n"
        "Steps:\n"
        "1. Call `summarize_adverse_events` with top_n=20.\n"
        "2. Optionally call `recent_adverse_event_reports` (limit=3) for color.\n"
        "3. Present:\n"
        "   - Total reports in FAERS\n"
        "   - Top 10 reactions with counts\n"
        "   - Seriousness breakdown (death, hospitalized, life-threatening)\n"
        "   - Reports by year (note any spike)\n"
        "   - 1\u20132 example case summaries from the recent reports\n"
        "4. Begin and end with the FAERS caveat: voluntary spontaneous reports, "
        "not incidence rates, not proof of causation."
    )


@mcp.prompt()
def device_overview(device_name: str) -> str:
    """Recipe: overview of a medical device (clearance + recalls + MAUDE)."""
    return (
        f"Give me an overview of the medical device \"{device_name}\".\n\n"
        "Steps:\n"
        "1. Call `search_device_510k` to find clearance info (K-numbers, decision dates, applicants).\n"
        "2. Call `find_device_recalls` for recall history.\n"
        "3. Call `summarize_device_events` for MAUDE adverse events.\n"
        "4. Present:\n"
        "   - Device class & medical specialty\n"
        "   - Most relevant 510(k) clearances with dates and applicants\n"
        "   - Recent recalls (date, classification, reason)\n"
        "   - Top reported problems / event types from MAUDE (with the voluntary-report caveat)\n"
        "5. Note that PMA (Class III) clearances live in a separate endpoint not yet covered here."
    )


@mcp.prompt()
def fda_application_brief(drug_name: str, identifier_type: str = "brand") -> str:
    """Recipe: 'tell me literally everything FDA knows about drug X' — deep dive."""
    return (
        f"Build the most complete openFDA-sourced profile possible for \"{drug_name}\" "
        f"(treat as a {identifier_type} name).\n\n"
        "Run these tools, IN ORDER, and present the result as a structured report:\n\n"
        "1. `resolve_drug_identifiers` — confirm spelling and pull every harmonized ID.\n"
        "2. `lookup_drug_approval` then `get_application_documents` for EACH application_number returned.\n"
        "   Present the FDA approval letter / SBA / medical review URLs as clickable links.\n"
        "3. `list_label_revisions` — show every SPL revision date.\n"
        "4. `get_label_section` for each of: indications_and_usage, boxed_warning, contraindications,\n"
        "   warnings_and_cautions, adverse_reactions, drug_interactions, dosage_and_administration,\n"
        "   clinical_pharmacology, mechanism_of_action, pharmacokinetics, clinical_studies,\n"
        "   use_in_specific_populations, medication_guide.\n"
        "5. `get_ndc_packaging` — every SKU, marketing dates, DEA schedule, package sizes.\n"
        "6. `find_drug_recalls` — then `get_recall_details` for each recall_number returned.\n"
        "7. `summarize_drug_shortages` (use the GENERIC name).\n"
        "8. `summarize_adverse_events` (top 20 reactions, by year).\n"
        "9. `summarize_event_outcomes` (death / hospitalization / reporter mix).\n"
        "10. `event_demographics` (age / sex / weight).\n"
        "11. `concomitant_drugs` (top 15 co-reported).\n"
        "12. `lookup_substance` using the UNII from step 1 — chemistry / formula / CAS.\n\n"
        "Present as a single document with sections: Identity, Approval History, Application\n"
        "Documents (with URLs), Label, Marketed Products, Recalls, Shortages, Pharmacovigilance,\n"
        "Chemistry. End with the global FAERS caveat verbatim and a list of the openFDA fields\n"
        "that are STILL not covered (Orange Book TE codes, REMS, PMR/PMC, advisory-committee transcripts)."
    )


@mcp.prompt()
def fda_inventory() -> str:
    """Recipe: balanced answer to 'how many drugs / devices does the FDA have on file?'"""
    return (
        "Answer the question 'how many drugs / devices does the FDA have on file?' "
        "by counting each layer separately. Use the `count_records` tool for every line.\n\n"
        "Drugs — run these counts:\n"
        "  a. count_records('drug', 'drugsfda')                                   # all FDA application records\n"
        "  b. count_records('drug', 'drugsfda', search='submissions.submission_status:\"AP\"')  # apps with an approved submission\n"
        "  c. count_records('drug', 'ndc')                                        # marketed product SKUs\n"
        "  d. count_records('drug', 'label')                                      # Structured Product Labels on file\n\n"
        "Devices — run these counts:\n"
        "  e. count_records('device', '510k')                                     # all 510(k) submissions\n"
        "  f. count_records('device', '510k', search='decision_description:\"substantially equivalent\"')  # cleared 510(k)s\n"
        "  g. count_records('device', 'classification')                           # distinct device categories\n\n"
        "Then present a single short table with one row per count, including the meta.last_updated date.\n\n"
        "You MUST disclose:\n"
        "  - There is no single 'approved drugs' number. The closest is (b): applications with at least one approved submission.\n"
        "  - NDC and label counts are inflated relative to (b) because one drug → many SKUs and many label revisions.\n"
        "  - 510(k) is only one device pathway; PMA (Class III) and De Novo / HDE are separate endpoints not counted here.\n"
        "  - Counts reflect the openFDA refresh date, not real-time FDA records."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server over stdio (the transport used by Claude Desktop and VS Code)."""
    if not API_KEY:
        print("WARNING: no openFDA API key found (env OPENFDA_API_KEY or config.json). "
              "Falling back to anonymous rate limits.", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
