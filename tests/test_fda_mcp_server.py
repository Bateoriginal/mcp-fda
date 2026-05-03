"""Tests for fda_mcp_server tool functions."""
from __future__ import annotations

import re

import httpx
import pytest
import respx

import fda_mcp_server as srv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FDA_URL = re.compile(r"^https://api\.fda\.gov/.*")


def _payload(total: int = 1, results=None, **meta):
    """Build a typical openFDA payload (``meta`` block + ``results`` list) for mocked responses."""
    return {
        "meta": {
            "last_updated": "2026-04-25",
            "results": {"skip": 0, "limit": len(results or []), "total": total},
            **meta,
        },
        "results": results or [],
    }


# ---------------------------------------------------------------------------
# _fda_request: validation + 404 handling
# ---------------------------------------------------------------------------

async def test_unknown_noun_raises():
    """`_fda_request` rejects unknown nouns before issuing any HTTP call."""
    with pytest.raises(RuntimeError, match="Unknown noun"):
        await srv._fda_request("widgets", "label", {})


async def test_unknown_category_raises():
    """`_fda_request` rejects unknown category for a known noun."""
    with pytest.raises(RuntimeError, match="Unknown category"):
        await srv._fda_request("drug", "not-a-thing", {})


@respx.mock
async def test_404_returns_empty_payload():
    """openFDA's 404 (\"no results\") is normalized to an empty payload, not an error."""
    respx.get(FDA_URL).mock(return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND"}}))
    out = await srv._fda_request("drug", "label", {"search": "openfda.brand_name:\"NOPE\""})
    assert out["results"] == []
    assert out["meta"]["results"]["total"] == 0


@respx.mock
async def test_5xx_raises_runtime():
    """5xx responses become :class:`RuntimeError` so MCP clients see a clear failure."""
    respx.get(FDA_URL).mock(return_value=httpx.Response(503, text="busy"))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await srv._fda_request("drug", "label", {})


@respx.mock
async def test_api_key_attached_when_present(monkeypatch):
    """When ``API_KEY`` is set, every outgoing request carries ``api_key=...``."""
    monkeypatch.setattr(srv, "API_KEY", "TESTKEY")
    route = respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload()))
    await srv._fda_request("drug", "label", {"limit": 1})
    assert "api_key=TESTKEY" in str(route.calls.last.request.url)


# ---------------------------------------------------------------------------
# Search-string builder
# ---------------------------------------------------------------------------

def test_drug_search_default_prefix():
    """Default search clause uses the harmonized ``openfda.<field>`` prefix."""
    assert srv._drug_search("brand_name", "X") == 'openfda.brand_name:"X"'


def test_drug_search_event_prefix():
    """On the FAERS endpoint the prefix nests under ``patient.drug.openfda``."""
    assert (
        srv._drug_search("brand_name", "X", on_event_endpoint=True)
        == 'patient.drug.openfda.brand_name:"X"'
    )


# ---------------------------------------------------------------------------
# Drug tools
# ---------------------------------------------------------------------------

@respx.mock
async def test_search_drug_label_returns_labels():
    """`search_drug_label` returns the trimmed summary plus the raw label records."""
    label = {"id": "abc", "openfda": {"brand_name": ["LIPITOR"]}, "indications_and_usage": ["uses"]}
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload(total=2, results=[label])))

    out = await srv.search_drug_label("LIPITOR")
    assert out["summary"]["total_matching"] == 2
    assert out["labels"][0]["id"] == "abc"


@respx.mock
async def test_search_drug_label_clamps_limit():
    """The label tool clamps the user-provided limit to the documented 1–10 range."""
    route = respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload()))
    await srv.search_drug_label("X", limit=999)
    assert "limit=10" in str(route.calls.last.request.url)


@respx.mock
async def test_find_drug_recalls_trims_fields_and_sorts():
    """Recall results are sorted newest-first and reduced to the curated field set."""
    rec = {
        "recall_initiation_date": "20251219",
        "status": "Ongoing",
        "classification": "Class II",
        "product_description": "Wegovy 2.4 mg",
        "reason_for_recall": "Particulate matter",
        "recalling_firm": "Novo Nordisk",
        "voluntary_mandated": "Voluntary",
        "ignored_field": "should not appear",
    }
    route = respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload(results=[rec])))

    out = await srv.find_drug_recalls("semaglutide", match_field="generic_name", limit=3)

    assert "sort=recall_initiation_date%3Adesc" in str(route.calls.last.request.url)
    only = out["recalls"][0]
    assert "ignored_field" not in only
    assert only["recalling_firm"] == "Novo Nordisk"


@respx.mock
async def test_lookup_drug_approval_extracts_products_and_submissions():
    """Approval records expose application number, products, and submission history."""
    payload = _payload(results=[{
        "application_number": "NDA209637",
        "sponsor_name": "Novo Nordisk",
        "products": [{"brand_name": "OZEMPIC", "active_ingredients": [{"name": "SEMAGLUTIDE"}],
                      "dosage_form": "INJECTION", "route": "SUBCUTANEOUS", "marketing_status": "Prescription"}],
        "submissions": [{"submission_type": "ORIG", "submission_number": "1",
                         "submission_status": "AP", "submission_status_date": "20171205"}],
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))

    out = await srv.lookup_drug_approval("OZEMPIC")
    assert out["approvals"][0]["application_number"] == "NDA209637"
    assert out["approvals"][0]["products"][0]["brand_name"] == "OZEMPIC"
    assert out["approvals"][0]["submissions"][0]["submission_status_date"] == "20171205"


@respx.mock
async def test_list_drug_ndcs_flattens_packaging():
    """NDC packaging entries collapse to a flat list of human-readable descriptions."""
    payload = _payload(results=[{
        "product_ndc": "0169-1704",
        "brand_name": "Ozempic",
        "generic_name": "semaglutide",
        "labeler_name": "Novo Nordisk",
        "dosage_form": "INJECTION",
        "route": ["SUBCUTANEOUS"],
        "marketing_category": "NDA",
        "active_ingredients": [{"name": "SEMAGLUTIDE", "strength": "2 mg/1.5mL"}],
        "packaging": [{"description": "1 PEN in 1 CARTON"}, {"description": "2 PEN in 1 CARTON"}],
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))

    out = await srv.list_drug_ndcs("Ozempic")
    p = out["products"][0]
    assert p["packaging"] == ["1 PEN in 1 CARTON", "2 PEN in 1 CARTON"]


@respx.mock
async def test_summarize_adverse_events_runs_all_facets():
    """FAERS summary issues one base count + one server-side count per facet."""
    base = _payload(total=42)
    facet = _payload(results=[{"term": "NAUSEA", "count": 10}])
    # First call returns the base count, all subsequent calls return the facet
    route = respx.get(FDA_URL).mock(side_effect=[
        httpx.Response(200, json=base),
        *[httpx.Response(200, json=facet)] * 6,
    ])
    out = await srv.summarize_adverse_events("OZEMPIC")
    assert out["total_reports_in_faers"] == 42
    assert set(out["aggregates"]) == {
        "top_reactions", "top_indications", "seriousness",
        "reporter_country", "patient_sex", "reports_by_year",
    }
    assert out["aggregates"]["top_reactions"] == [{"term": "NAUSEA", "count": 10}]
    assert route.call_count == 7  # 1 base + 6 facets


@respx.mock
async def test_recent_adverse_event_reports_trims_records():
    """Each FAERS report is reduced to the curated subset useful for inspection."""
    raw = {
        "safetyreportid": "1",
        "receivedate": "20240101",
        "serious": "1",
        "seriousnessdeath": None,
        "primarysource": {"reportercountry": "US"},
        "patient": {
            "patientonsetage": "50",
            "patientsex": "1",
            "reaction": [{"reactionmeddrapt": "NAUSEA"}],
            "drug": [{"medicinalproduct": "OZEMPIC", "drugindication": "DM",
                      "drugadministrationroute": "SC"}],
        },
        "dropped_field": "x",
    }
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload(results=[raw])))

    out = await srv.recent_adverse_event_reports("OZEMPIC", limit=5)
    rep = out["reports"][0]
    assert rep["reactions"] == ["NAUSEA"]
    assert rep["drugs"][0]["medicinalproduct"] == "OZEMPIC"
    assert "dropped_field" not in rep


@respx.mock
async def test_resolve_drug_identifiers_dedupes():
    """Identifier resolution unions cross-record values and deduplicates them."""
    label_a = {"openfda": {
        "brand_name": ["OZEMPIC"], "generic_name": ["SEMAGLUTIDE"],
        "unii": ["53AXN4NNHX"], "product_ndc": ["0169-1704"],
        "manufacturer_name": ["Novo Nordisk"],
    }}
    label_b = {"openfda": {
        "brand_name": ["Ozempic"], "generic_name": ["SEMAGLUTIDE"],  # dup with diff casing
        "unii": ["53AXN4NNHX"], "product_ndc": ["0169-1709"],
        "manufacturer_name": ["Novo Nordisk"],
    }}
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload(total=2, results=[label_a, label_b])))

    out = await srv.resolve_drug_identifiers("OZEMPIC")
    ids = out["identifiers"]
    assert ids["unii"] == ["53AXN4NNHX"]
    assert sorted(ids["product_ndc"]) == ["0169-1704", "0169-1709"]
    assert "OZEMPIC" in ids["brand_name"] and "Ozempic" in ids["brand_name"]


@respx.mock
async def test_build_drug_dossier_aggregates_endpoints():
    """`build_drug_dossier` collects a count + sample for every drug endpoint."""
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload(total=3, results=[{"x": 1}])))

    out = await srv.build_drug_dossier("OZEMPIC")
    assert set(out["endpoints"]) >= {"label", "ndc", "drugsfda", "enforcement", "shortages", "event"}
    for ep, info in out["endpoints"].items():
        if "error" in info:
            continue
        assert info["total"] == 3


@respx.mock
async def test_build_drug_dossier_includes_event_samples_when_requested():
    """Setting ``sample_events`` > 0 attaches raw FAERS records to the dossier."""
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload(total=1, results=[{"safetyreportid": "1"}])))
    out = await srv.build_drug_dossier("OZEMPIC", sample_events=2)
    assert out["endpoints"]["event"]["sample"] == [{"safetyreportid": "1"}]


# ---------------------------------------------------------------------------
# Device tools
# ---------------------------------------------------------------------------

@respx.mock
async def test_search_device_510k_returns_clearances():
    """510(k) tool returns the curated clearance fields plus the harmonized class."""
    payload = _payload(results=[{
        "k_number": "K050312",
        "device_name": "AMIGO INSULIN PUMP",
        "applicant": "Asante",
        "decision_date": "2005-05-09",
        "decision_description": "SUBSTANTIALLY EQUIVALENT",
        "product_code": "LZG",
        "openfda": {"device_class": "2", "medical_specialty_description": "Clinical Chemistry"},
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))

    out = await srv.search_device_510k("insulin pump")
    c = out["clearances"][0]
    assert c["k_number"] == "K050312"
    assert c["device_class"] == "2"


@respx.mock
async def test_find_device_recalls_returns_items():
    """Device enforcement records map to the same trimmed shape as drug recalls."""
    payload = _payload(results=[{
        "recall_initiation_date": "20240101",
        "classification": "Class II",
        "product_description": "Pump tubing",
        "reason_for_recall": "Leak",
        "recalling_firm": "Acme",
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))

    out = await srv.find_device_recalls("insulin pump")
    assert out["recalls"][0]["recalling_firm"] == "Acme"


@respx.mock
async def test_summarize_device_events_runs_facets():
    """MAUDE summary issues a base count + one server-side count per device facet."""
    base = _payload(total=10)
    facet = _payload(results=[{"term": "Malfunction", "count": 5}])
    respx.get(FDA_URL).mock(side_effect=[
        httpx.Response(200, json=base),
        *[httpx.Response(200, json=facet)] * 4,
    ])
    out = await srv.summarize_device_events("insulin pump")
    assert out["total_reports"] == 10
    assert set(out["aggregates"]) == {
        "top_event_types", "top_problems", "top_manufacturers", "events_by_year",
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

async def test_list_openfda_endpoints_returns_catalog():
    """The discovery tool returns the noun → categories catalog and the base URL."""
    out = await srv.list_openfda_endpoints()
    assert "drug" in out["endpoints"]
    assert "label" in out["endpoints"]["drug"]
    assert out["base_url"].startswith("https://api.fda.gov")


@respx.mock
async def test_raw_openfda_query_passes_through():
    """`raw_openfda_query` forwards parameters and returns the raw payload untouched."""
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload(total=1, results=[{"y": 1}])))
    out = await srv.raw_openfda_query("drug", "label", search='openfda.brand_name:"X"', limit=5)
    assert out["results"] == [{"y": 1}]


@respx.mock
async def test_raw_openfda_query_clamps_limit_and_skip():
    """`raw_openfda_query` clamps ``limit`` to 1000 and ``skip`` to the API's 25,000 cap."""
    route = respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload()))
    await srv.raw_openfda_query("drug", "label", limit=99999, skip=99999)
    url = str(route.calls.last.request.url)
    assert "limit=1000" in url
    assert "skip=25000" in url


# ---------------------------------------------------------------------------
# count_records + plan_drug_question
# ---------------------------------------------------------------------------

@respx.mock
async def test_count_records_returns_total_and_last_updated():
    """`count_records` exposes meta.results.total and meta.last_updated for the host."""
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=_payload(total=25246)))
    out = await srv.count_records("drug", "drugsfda", search='submissions.submission_status:"AP"')
    assert out["total"] == 25246
    assert out["last_updated"] == "2026-04-25"
    assert out["noun"] == "drug"
    assert out["category"] == "drugsfda"


@respx.mock
async def test_count_records_normalizes_404_to_zero():
    """A 404 from openFDA is reported as a count of 0, not raised."""
    respx.get(FDA_URL).mock(return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND"}}))
    out = await srv.count_records("drug", "shortages", search='openfda.generic_name:"nope"')
    assert out["total"] == 0


async def test_plan_drug_question_routes_catalog_stats():
    """The 'how many drugs are approved?' phrasing routes to the catalog_stats intent."""
    out = await srv.plan_drug_question("How many drugs are approved by the FDA?")
    assert "catalog_stats" in out["intents"]
    plan = out["plans"]["catalog_stats"]
    assert any("count_records" in tool for tool in plan["tools"])
    assert any("approved drugs" in c for c in plan["must_disclose"])


async def test_plan_drug_question_routes_recall():
    """A recall question still routes to the recall intent (regression)."""
    out = await srv.plan_drug_question("Has Ozempic been recalled in the last year?")
    assert "recall" in out["intents"]


async def test_plan_drug_question_unrecognized_returns_menu():
    """A vague question yields no intent + a menu of clarifying questions."""
    out = await srv.plan_drug_question("what about lipitor")
    assert out["intents"] == []
    assert out["clarifying_questions"]
    assert "available_intents" in out


# ---------------------------------------------------------------------------
# Deep-dive drug tools (label, drugsfda docs, ndc, enforcement, shortages)
# ---------------------------------------------------------------------------

@respx.mock
async def test_get_label_section_returns_named_section():
    """`get_label_section` returns one section verbatim with version metadata."""
    payload = _payload(results=[{
        "set_id": "abc-123",
        "version": "42",
        "effective_time": "20260101",
        "clinical_studies": ["SUSTAIN-6 trial showed ..."],
        "indications_and_usage": ["adjunct to diet ..."],
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))

    out = await srv.get_label_section("ozempic", "clinical_studies")
    assert out["section"] == "clinical_studies"
    assert out["content"] == ["SUSTAIN-6 trial showed ..."]
    assert out["version"] == "42"
    assert out["set_id"] == "abc-123"


@respx.mock
async def test_get_label_section_unknown_returns_available_list():
    """Asking for a missing section returns the available section keys, not an error."""
    payload = _payload(results=[{
        "set_id": "abc",
        "indications_and_usage": ["..."],
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))
    out = await srv.get_label_section("ozempic", "does_not_exist")
    assert "available_sections" in out
    assert "indications_and_usage" in out["available_sections"]


@respx.mock
async def test_list_label_revisions_returns_one_row_per_label():
    """One row per SPL revision plus a union of section keys."""
    payload = _payload(results=[
        {"effective_time": "20260101", "version": "2", "set_id": "s1",
         "indications_and_usage": ["a"], "openfda": {"brand_name": ["OZEMPIC"]}},
        {"effective_time": "20240101", "version": "1", "set_id": "s1",
         "boxed_warning": ["b"], "openfda": {"brand_name": ["OZEMPIC"]}},
    ])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))
    out = await srv.list_label_revisions("ozempic")
    assert len(out["revisions"]) == 2
    assert out["revisions"][0]["version"] == "2"
    assert "indications_and_usage" in out["known_sections"]
    assert "boxed_warning" in out["known_sections"]


@respx.mock
async def test_get_application_documents_surfaces_urls():
    """Approval-document URLs and submission_class_code are returned per submission."""
    payload = _payload(results=[{
        "application_number": "NDA209637",
        "sponsor_name": "NOVO",
        "submissions": [{
            "submission_type": "ORIG", "submission_number": "1",
            "submission_class_code": "TYPE 1", "submission_status": "AP",
            "submission_status_date": "20171205",
            "application_docs": [
                {"id": 1, "date": "20171205", "type": "Letter",
                 "url": "https://accessdata.fda.gov/letter.pdf"},
                {"id": 2, "date": "20171205", "type": "Label",
                 "url": "https://accessdata.fda.gov/label.pdf"},
            ],
        }],
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))
    out = await srv.get_application_documents("NDA209637")
    docs = out["applications"][0]["submissions"][0]["application_docs"]
    assert {d["type"] for d in docs} == {"Letter", "Label"}
    assert docs[0]["url"].startswith("https://")


@respx.mock
async def test_summarize_drug_shortages_groups_status():
    """Status counts are aggregated and rows preserve the raw shortage record."""
    payload = _payload(results=[
        {"generic_name": "SEMAGLUTIDE", "status": "Resolved",
         "change_date": "20240101", "company_name": "Novo", "strength": "0.5 mg"},
        {"generic_name": "SEMAGLUTIDE", "status": "Resolved",
         "change_date": "20230601", "company_name": "Novo", "strength": "1 mg"},
        {"generic_name": "SEMAGLUTIDE", "status": "Currently in Shortage",
         "change_date": "20220101", "company_name": "Novo", "strength": "2 mg"},
    ])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))
    out = await srv.summarize_drug_shortages("semaglutide")
    assert out["status_breakdown"] == {"Resolved": 2, "Currently in Shortage": 1}
    assert len(out["records"]) == 3


@respx.mock
async def test_get_ndc_packaging_includes_marketing_dates():
    """Each NDC row includes marketing dates, DEA schedule, and package detail."""
    payload = _payload(results=[{
        "product_ndc": "0169-4132", "brand_name": "OZEMPIC",
        "generic_name": "SEMAGLUTIDE", "labeler_name": "Novo",
        "marketing_start_date": "20171205", "marketing_end_date": None,
        "dea_schedule": None, "finished": True,
        "packaging": [
            {"package_ndc": "0169-4132-90", "description": "1 PEN in 1 CARTON",
             "marketing_start_date": "20171205", "sample": False},
        ],
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))
    out = await srv.get_ndc_packaging("OZEMPIC")
    p = out["products"][0]
    assert p["marketing_start_date"] == "20171205"
    assert p["packaging"][0]["package_ndc"] == "0169-4132-90"


@respx.mock
async def test_get_recall_details_returns_lot_codes():
    """`get_recall_details` exposes code_info + distribution_pattern from drug enforcement."""
    payload = _payload(results=[{
        "recall_number": "D-1234-2024", "event_id": "9999",
        "status": "Ongoing", "classification": "Class II",
        "recall_initiation_date": "20240101", "termination_date": None,
        "recalling_firm": "Novo", "voluntary_mandated": "Voluntary: Firm initiated",
        "product_description": "Ozempic 1mg pen", "product_quantity": "10000 pens",
        "reason_for_recall": "Sterility concern",
        "code_info": "Lots ABC123, ABC124", "distribution_pattern": "Nationwide",
    }])
    respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))
    out = await srv.get_recall_details("D-1234-2024")
    assert out["noun"] == "drug"
    assert out["recall"]["code_info"] == "Lots ABC123, ABC124"
    assert out["recall"]["distribution_pattern"] == "Nationwide"


# ---------------------------------------------------------------------------
# Deep-dive FAERS tools
# ---------------------------------------------------------------------------

@respx.mock
async def test_summarize_event_outcomes_runs_facets():
    """One base count + one facet call per outcome / qualification field."""
    base = _payload(total=100)
    facet = _payload(results=[{"term": 1, "count": 30}])
    # 1 base + 11 facets
    respx.get(FDA_URL).mock(side_effect=[
        httpx.Response(200, json=base),
        *[httpx.Response(200, json=facet)] * 11,
    ])
    out = await srv.summarize_event_outcomes("ozempic")
    assert out["total_reports_in_faers"] == 100
    assert "seriousnessdeath" in out["aggregates"]
    assert "reporter_qualification" in out["aggregates"]


@respx.mock
async def test_concomitant_drugs_strips_target_drug():
    """The target drug name AND its resolved generic/substance terms are removed."""
    co_payload = _payload(results=[
        {"term": "OZEMPIC", "count": 9999},
        {"term": "SEMAGLUTIDE", "count": 9999},
        {"term": "METFORMIN", "count": 5000},
        {"term": "INSULIN GLARGINE", "count": 2000},
    ])
    resolve_payload = _payload(results=[{
        "openfda": {
            "brand_name": ["OZEMPIC"],
            "generic_name": ["SEMAGLUTIDE"],
            "substance_name": ["SEMAGLUTIDE"],
        },
    }])
    # 1st call = the count facet, 2nd = resolve_drug_identifiers' label fetch
    respx.get(FDA_URL).mock(side_effect=[
        httpx.Response(200, json=co_payload),
        httpx.Response(200, json=resolve_payload),
    ])
    out = await srv.concomitant_drugs("ozempic", top_n=5)
    terms = [row["term"] for row in out["co_reported_drugs"]]
    assert "OZEMPIC" not in terms
    assert "SEMAGLUTIDE" not in terms
    assert "METFORMIN" in terms


@respx.mock
async def test_disproportionality_signal_computes_ror():
    """ROR matches the textbook 2x2 calculation; signal flag honours CI lower bound."""
    # Call order is: rxn_total (exact), a, drug_total, grand_total.
    # a=100, drug_total=1000 -> b=900; rxn_total=10000 -> c=9900;
    # grand_total=10_000_000 -> d=10_000_000 - 100 - 900 - 9900 = 9_989_100
    # ROR = (100*9989100)/(900*9900) ~= 112
    respx.get(FDA_URL).mock(side_effect=[
        httpx.Response(200, json=_payload(total=10_000)),    # rxn_total (exact)
        httpx.Response(200, json=_payload(total=100)),       # a
        httpx.Response(200, json=_payload(total=1_000)),     # drug_total
        httpx.Response(200, json=_payload(total=10_000_000)),# grand_total
    ])
    out = await srv.disproportionality_signal("semaglutide", "pulmonary aspiration")
    assert out["contingency"]["a"] == 100
    assert out["ror"] is not None and out["ror"] > 50
    assert out["signal"] is True


@respx.mock
async def test_disproportionality_signal_zero_cell_returns_note():
    """Zero in any cell means we cannot compute ROR; a `note` explains why."""
    # rxn_total=0 -> fallback rxn_total=0 -> a=0 -> drug_total=1000 -> grand=1M
    respx.get(FDA_URL).mock(side_effect=[
        httpx.Response(200, json=_payload(total=0)),         # rxn_total exact
        httpx.Response(200, json=_payload(total=0)),         # rxn_total fallback
        httpx.Response(200, json=_payload(total=0)),         # a
        httpx.Response(200, json=_payload(total=1_000)),     # drug_total
        httpx.Response(200, json=_payload(total=1_000_000)), # grand_total
    ])
    out = await srv.disproportionality_signal("foo", "bar")
    assert out["ror"] is None
    assert "note" in out


# ---------------------------------------------------------------------------
# Reference / chemistry
# ---------------------------------------------------------------------------

@respx.mock
async def test_lookup_substance_by_unii():
    """A 10-char uppercase alphanumeric query is treated as a UNII lookup."""
    payload = _payload(results=[{"unii": "53AXN4NNHX", "names": [{"name": "SEMAGLUTIDE"}]}])
    route = respx.get(FDA_URL).mock(return_value=httpx.Response(200, json=payload))
    out = await srv.lookup_substance("53AXN4NNHX")
    assert out["substances"][0]["unii"] == "53AXN4NNHX"
    assert "unii%3A" in str(route.calls.last.request.url)


# ---------------------------------------------------------------------------
# Router additions
# ---------------------------------------------------------------------------

async def test_plan_drug_question_routes_pharmacovigilance():
    """Death / hospitalization / signal phrasing routes to pharmacovigilance."""
    out = await srv.plan_drug_question("How many Ozempic deaths reported to FAERS?")
    assert "pharmacovigilance" in out["intents"]


async def test_plan_drug_question_routes_label_section():
    """Asking for clinical_studies routes to the label_section intent."""
    out = await srv.plan_drug_question("Show me the clinical studies section of the Ozempic label")
    assert "label_section" in out["intents"]


async def test_plan_drug_question_routes_approval_documents():
    """Asking for the approval letter routes to the approval_documents intent."""
    out = await srv.plan_drug_question("Give me the FDA approval letter for NDA 209637")
    assert "approval_documents" in out["intents"]
