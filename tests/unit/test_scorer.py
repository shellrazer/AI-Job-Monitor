"""Tests for job_monitor.scorer (spec §10-12).

Calibration strategy: ``semantic`` is the only non-deterministic component (it
depends on the caller's embedding cosine), so the worked-example tests assert
every RULE component (seniority, industry, company, location_salary) EXACTLY and
pin the final tier under a MOCKED similarity. The remaining tests are targeted
unit checks of each table and helper.
"""

from __future__ import annotations

import hashlib

import pytest

from job_monitor.config import ProfileConfig, load_config
from job_monitor.models import Job, PriorityTier, ScoreComponents, Source
from job_monitor.scorer import (
    canonical_seniority,
    classify_company,
    classify_industry,
    classify_location,
    compute_semantic,
    detect_signals,
    final_score,
    priority_tier,
    resume_tips,
    score_company,
    score_industry,
    score_job,
    score_location_salary,
    score_salary,
    score_seniority,
    strong_alert,
)

# A real ScoringConfig loaded from config/scoring.yaml (the runtime source of truth).
_SCORING = load_config().scoring
_PROFILE = ProfileConfig(full_text="Senior food quality manager, FSSC 22000, HACCP, multi-site, supplier audits.")


def _job(
    title: str,
    description: str,
    location: str | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
    salary_currency: str | None = None,
) -> Job:
    return Job(
        source=Source.SEEK,
        title=title,
        normalized_title=title.lower(),
        company_name="Example Co",
        apply_url="https://example.com/job",
        description_hash=hashlib.sha256(description.encode()).hexdigest(),
        location=location,
        description=description,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
    )


def _similarity_for(semantic_target: float) -> float:
    """Invert compute_semantic: a target semantic in [0,100] -> cosine similarity."""
    return semantic_target / 100 * 0.8


# --------------------------------------------------------------------------- #
# Worked examples (tier-calibration anchors).                                 #
# --------------------------------------------------------------------------- #
def test_example1_site_quality_manager_lands_a_plus() -> None:
    job = _job(
        "Site Quality Manager",
        "Lead the site quality team. HACCP, GMP, audit readiness, QA team, site leadership.",
        location="Western Sydney NSW",
        salary_min=140000,
        salary_max=160000,
        salary_currency="AUD",
    )
    assert canonical_seniority(job.title) == "site_quality_manager"
    assert score_seniority("site_quality_manager", scoring=_SCORING) == pytest.approx(93.333, abs=0.01)

    jd_text = f"{job.title}\n{job.description}"
    signals = detect_signals(jd_text)
    # industry = clamp(25 + [site_leadership 15 + manages_quality_team 15
    #            + food_safety_cert 15 + customer_audit 10]) = 80
    assert score_industry("food manufacturing", signals, jd_text, scoring=_SCORING) == 80
    # is_tier1 -> large_food_group -> 100
    assert score_company("food manufacturing", jd_text, "Example Co", False, True, scoring=_SCORING) == 100
    assert score_location_salary("Western Sydney NSW", 140000, 160000, True, scoring=_SCORING) == 95.0

    result = score_job(
        job,
        similarity=_similarity_for(85.0),
        scoring=_SCORING,
        profile=_PROFILE,
        sector="food manufacturing",
        is_tier1=True,
    )
    assert result.components.semantic == pytest.approx(85.0, abs=0.01)
    assert result.components.seniority == pytest.approx(93.333, abs=0.01)
    assert result.components.industry == 80
    assert result.components.company == 100
    assert result.components.location_salary == 95.0
    # final = .30*93.333 + .25*85 + .20*80 + .15*100 + .10*95 = 89.8
    assert result.final_score == pytest.approx(89.8, abs=0.05)
    assert result.priority_tier == PriorityTier.A_PLUS
    assert result.strong_alert is True


def test_example2_senior_food_quality_specialist_lands_a() -> None:
    job = _job(
        "Senior Food Quality Specialist",
        "Responsible for supplier quality, food safety, and risk management across our retail group.",
        location="Sydney",
    )
    # Promotion rule: senior + quality + in-domain, no manager/officer head-noun.
    assert canonical_seniority(job.title) == "senior_quality_manager"

    jd_text = f"{job.title}\n{job.description}"
    signals = detect_signals(jd_text)
    # industry = clamp(22 + [supplier 15 + food_safety_cert 15 + capa 10]) = 62
    assert score_industry("retail group", signals, jd_text, scoring=_SCORING) == 62
    # sector "retail group" -> large_retail_food -> 12/15*100 = 80
    assert score_company("retail group", jd_text, "Example Co", False, False, scoring=_SCORING) == 80
    assert score_location_salary("Sydney", None, None, False, scoring=_SCORING) == 75.0

    result = score_job(
        job,
        similarity=_similarity_for(85.0),
        scoring=_SCORING,
        profile=_PROFILE,
        sector="retail group",
    )
    assert result.components.seniority == pytest.approx(83.333, abs=0.01)
    assert result.components.industry == 62
    assert result.components.company == 80
    assert result.components.location_salary == 75.0
    # final = .30*83.333 + .25*85 + .20*62 + .15*80 + .10*75 = 78.2
    assert result.final_score == pytest.approx(78.2, abs=0.05)
    assert result.priority_tier == PriorityTier.A


def test_example3_qa_officer_lands_d() -> None:
    job = _job(
        "QA Officer",
        "Entry-level QA officer, food factory.",
        location="Sydney",
        salary_min=75000,
        salary_max=85000,
        salary_currency="AUD",
    )
    assert canonical_seniority(job.title) == "qa_officer_coordinator"
    assert score_seniority("qa_officer_coordinator", scoring=_SCORING) == 10.0

    jd_text = f"{job.title}\n{job.description}"
    signals = detect_signals(jd_text)
    # industry = clamp(25 + 0) = 25 (no responsibility signals fire)
    assert score_industry("food", signals, jd_text, scoring=_SCORING) == 25
    # sector "food" -> mid_food_manufacturer -> 8/15*100 = 53.333
    assert score_company("food", jd_text, "Example Co", False, False, scoring=_SCORING) == pytest.approx(
        53.333, abs=0.01
    )
    assert score_location_salary("Sydney", 75000, 85000, True, scoring=_SCORING) == pytest.approx(58.333, abs=0.01)

    result = score_job(
        job,
        similarity=_similarity_for(50.0),
        scoring=_SCORING,
        profile=_PROFILE,
        sector="food",
    )
    assert result.components.seniority == 10.0
    assert result.components.industry == 25
    assert result.components.company == pytest.approx(53.333, abs=0.01)
    assert result.components.location_salary == pytest.approx(58.333, abs=0.01)
    # final = .30*10 + .25*50 + .20*25 + .15*53.333 + .10*58.333 = 34.3
    assert result.final_score == pytest.approx(34.3, abs=0.05)
    assert result.priority_tier == PriorityTier.D


# --------------------------------------------------------------------------- #
# Seniority categories.                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("title", "category"),
    [
        ("National Quality Manager", "national_group_head"),
        ("Group Head of Quality", "national_group_head"),
        ("Head of Quality", "national_group_head"),
        ("Site Quality Manager", "site_quality_manager"),
        ("Senior Quality Manager", "senior_quality_manager"),
        ("Food Safety Quality Manager", "food_safety_quality_manager"),
        ("Supplier Quality Manager", "supplier_quality_manager"),
        ("Vendor Assurance Manager", "supplier_quality_manager"),
        ("Regulatory Affairs Manager", "regulatory_affairs_manager_food"),
        ("Site Technical Manager", "site_technical_manager"),
        ("Technical Manager", "site_technical_manager"),
        ("QA Manager", "qa_manager"),
        ("Quality Manager", "qa_manager"),
        ("Senior Specialist", "senior_specialist_quality_lead"),
        ("Quality Lead", "senior_specialist_quality_lead"),
        ("QA Officer", "qa_officer_coordinator"),
        ("Quality Coordinator", "qa_officer_coordinator"),
        ("Warehouse Picker", "other"),
    ],
)
def test_canonical_seniority_precedence(title: str, category: str) -> None:
    assert canonical_seniority(title) == category


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("national_group_head", 100.0),
        ("site_quality_manager", 93.333),
        ("senior_quality_manager", 83.333),
        ("food_safety_quality_manager", 83.333),
        ("supplier_quality_manager", 73.333),
        ("regulatory_affairs_manager_food", 66.667),
        ("site_technical_manager", 66.667),
        ("qa_manager", 50.0),
        ("senior_specialist_quality_lead", 33.333),
        ("qa_officer_coordinator", 10.0),
        ("other", 0.0),
    ],
)
def test_score_seniority_normalization(category: str, expected: float) -> None:
    assert score_seniority(category, scoring=_SCORING) == pytest.approx(expected, abs=0.01)


def test_senior_specialist_promotion_can_be_disabled() -> None:
    title = "Senior Food Quality Specialist"
    assert canonical_seniority(title, promote_senior_specialist=True) == "senior_quality_manager"
    assert canonical_seniority(title, promote_senior_specialist=False) == "senior_specialist_quality_lead"


def test_site_technical_manager_not_site_quality_manager() -> None:
    # A *technical* site role must not be misread as a Site Quality Manager.
    assert canonical_seniority("Site Technical Quality Manager") == "site_technical_manager"


# --------------------------------------------------------------------------- #
# Location & salary bands.                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("location", "key"),
    [
        ("Western Sydney NSW", "sydney_greater"),
        ("Parramatta", "sydney_greater"),
        ("Macquarie Park", "sydney_greater"),
        ("Norwest", "sydney_greater"),
        ("Some Suburb NSW", "sydney_greater"),
        ("Newcastle", "nsw_regional"),
        ("Wollongong", "nsw_regional"),
        ("Orange NSW Regional", "nsw_regional"),
        ("Melbourne VIC", "melbourne_brisbane"),
        ("Brisbane QLD", "melbourne_brisbane"),
        ("Perth WA", "other_au"),
        ("Adelaide SA", "other_au"),
        ("Australia", "other_au"),
        ("London UK", "overseas"),
        ("London", "overseas"),
        ("Hung Yen, Vietnam", "overseas"),
        ("Auckland, New Zealand", "overseas"),
        ("Toledo", "overseas"),
        ("Rayong Plant", "overseas"),
        ("Chicago, IL", "overseas"),
        ("", "other_au"),
        (None, "other_au"),
    ],
)
def test_classify_location(location: str | None, key: str) -> None:
    assert classify_location(location) == key


def test_classify_location_overseas_filtered() -> None:
    # Spot-check the spec's overseas examples (these are filtered by the pipeline).
    overseas = ["Hung Yen, Vietnam", "Auckland, New Zealand", "Toledo", "Rayong Plant", "Chicago, IL", "London"]
    for loc in overseas:
        assert classify_location(loc) == "overseas", loc
    # AU / empty / ambiguous stay non-overseas.
    for loc in ["Sydney", "Melbourne VIC", "Newcastle", "Perth WA", "Australia", "", None]:
        assert classify_location(loc) != "overseas", loc


@pytest.mark.parametrize(
    ("salary_max", "expected"),
    [
        (170000, 15.0),  # band edge
        (180000, 15.0),
        (150000, 12.0),  # band edge
        (160000, 12.0),
        (130000, 10.0),  # band edge
        (140000, 10.0),
        (110000, 5.0),  # band edge
        (105000, 0.0),  # documented neutral gap
        (100000, 0.0),  # documented neutral gap
        (90000, -10.0),
        (99999, -10.0),
    ],
)
def test_score_salary_bands(salary_max: int, expected: float) -> None:
    assert score_salary(None, salary_max, True, scoring=_SCORING) == expected


def test_score_salary_not_disclosed_is_zero() -> None:
    assert score_salary(140000, 160000, False, scoring=_SCORING) == 0.0


def test_score_salary_falls_back_to_min() -> None:
    assert score_salary(170000, None, True, scoring=_SCORING) == 15.0


def test_score_location_salary_rescale_bounds() -> None:
    # overseas (-20) + low salary (-10) -> floor at 0.
    assert score_location_salary("London UK", None, 80000, True, scoring=_SCORING) == 0.0
    # sydney_greater (15) + top salary (15) -> ceiling at 100.
    assert score_location_salary("Sydney", None, 200000, True, scoring=_SCORING) == 100.0


# --------------------------------------------------------------------------- #
# Signal detection / exclusions.                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "signal"),
    [
        ("Software QA Engineer with Selenium and CI/CD", "ex_software_qa"),
        ("Civil construction QA on an infrastructure project", "ex_construction"),
        ("NDIS support coordinator for disability services", "ex_ndis"),
        ("Aged care and healthcare clinical quality role", "ex_health_aged_edu"),
        ("Call centre customer service representative", "ex_call_centre"),
        ("Laboratory technician doing sample preparation", "ex_lab_tech"),
        ("Junior graduate trainee entry-level role", "ex_junior"),
        ("Machine operator and forklift production worker", "ex_operator"),
        ("Chef and barista in hospitality kitchen", "ex_food_service_role"),
    ],
)
def test_exclusion_signals_fire(text: str, signal: str) -> None:
    assert signal in detect_signals(text)


@pytest.mark.parametrize(
    ("text", "signal"),
    [
        ("Multi-site national quality across 8 plants", "multi_site_national"),
        ("Manage the quality team and direct reports", "manages_quality_team"),
        ("Laboratory management and micro testing", "laboratory"),
        ("Site leadership of the plant", "site_leadership"),
        ("Supplier quality and vendor approval", "supplier_vendor_assurance"),
        ("Manage co-packers and contract manufacturers", "co_packer"),
        ("Customer audit and retailer audit readiness", "customer_retailer_audit"),
        ("HACCP, FSSC 22000, GMP, food safety management", "food_safety_cert"),
        ("Regulatory compliance and labelling", "regulatory_compliance"),
        ("CAPA, non-conformance, traceability, risk assessment", "capa_nc_traceability"),
        ("Quality systems ownership and technical leadership", "alert_quality_systems_ownership"),
    ],
)
def test_responsibility_and_alert_signals_fire(text: str, signal: str) -> None:
    assert signal in detect_signals(text)


# --------------------------------------------------------------------------- #
# Industry / company classification.                                          #
# --------------------------------------------------------------------------- #
def test_classify_industry_out_of_domain_wins() -> None:
    assert classify_industry("software", "Software QA Engineer, SaaS platform") == "out_of_domain"
    assert classify_industry("construction", "Civil construction site quality") == "out_of_domain"


def test_classify_industry_food_vs_retailer() -> None:
    assert classify_industry("food manufacturing", "site quality") == "food_manufacturing_fmcg"
    assert classify_industry("food", "food factory") == "food_manufacturing_fmcg"
    assert classify_industry("dairy", "milk processing") == "food_subsector"
    assert classify_industry("retail group", "supplier quality, food safety") == "retailer_supplier_quality"
    assert classify_industry("foodservice", "central kitchen") == "foodservice"
    assert classify_industry("pharma", "medical device quality") == "pharma_meddevice"
    assert classify_industry("mining", "drill quality") == "unclear"


def test_classify_company_tiers() -> None:
    assert classify_company("food manufacturing", "", False) == "large_food_group"
    assert classify_company("retail group", "", False) == "large_retail_food"
    assert classify_company("food", "", False) == "mid_food_manufacturer"
    assert classify_company("unknown", "", True) == "recruiter_anonymous_food"
    assert classify_company("unknown", "", False) == "unclear"


def test_classify_company_by_employer_name() -> None:
    # Known major employers classify by name even with no sector (aggregator jobs).
    assert classify_company("", "", False, "Nestle Australia") == "large_food_group"
    assert classify_company("", "", False, "PepsiCo ANZ") == "large_food_group"
    assert classify_company("", "", False, "Coca-Cola Europacific Partners") == "large_food_group"
    assert classify_company("", "", False, "Woolworths Group") == "large_retail_food"
    assert classify_company("", "", False, "Coles Supermarkets") == "large_retail_food"
    assert classify_company("", "", False, "Michael Page") == "recruiter_anonymous_food"
    assert classify_company("", "", False, "Hays Recruitment") == "recruiter_anonymous_food"
    assert classify_company("", "", False, "Some Random Pty Ltd") == "unclear"


def test_score_company_normalization() -> None:
    # points/15*100 mapping.
    assert score_company("food manufacturing", "", "Co", False, False, scoring=_SCORING) == 100
    assert score_company("retail group", "", "Co", False, False, scoring=_SCORING) == 80
    assert score_company("food", "", "Co", False, False, scoring=_SCORING) == pytest.approx(53.333, abs=0.01)
    assert score_company("unknown", "", "Co", True, False, scoring=_SCORING) == pytest.approx(33.333, abs=0.01)
    assert score_company("unknown", "", "Co", False, False, scoring=_SCORING) == 0
    # is_tier1 forces at least large_food_group (100).
    assert score_company("unknown", "", "Co", False, True, scoring=_SCORING) == 100


def test_industry_audit_double_count_avoided() -> None:
    # supplier_vendor_assurance + customer_retailer_audit: subtract the audit once.
    text = "Supplier quality and customer audit readiness for our food factory"
    signals = detect_signals(text)
    assert "supplier_vendor_assurance" in signals
    assert "customer_retailer_audit" in signals
    # food_manufacturing 25 + (supplier 15 + customer_audit 10 - 10) = 40
    assert score_industry("food", signals, text, scoring=_SCORING) == 40


def test_industry_out_of_domain_floors_at_zero() -> None:
    text = "Software QA Engineer, SaaS platform, test automation, food safety jargon"
    signals = detect_signals(text)
    assert score_industry("software", signals, text, scoring=_SCORING) == 0


# --------------------------------------------------------------------------- #
# Semantic, final score, tier.                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("similarity", "expected"),
    [
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.4, 50.0),
        (0.8, 100.0),
        (0.95, 100.0),
    ],
)
def test_compute_semantic(similarity: float, expected: float) -> None:
    assert compute_semantic(similarity) == pytest.approx(expected, abs=0.01)


def test_final_score_weighted_sum() -> None:
    components = ScoreComponents(
        semantic=80.0, seniority=90.0, industry=70.0, company=50.0, location_salary=60.0
    )
    # 0.30*90 + 0.25*80 + 0.20*70 + 0.15*50 + 0.10*60 = 27 + 20 + 14 + 7.5 + 6 = 74.5
    assert final_score(components, scoring=_SCORING) == 74.5


@pytest.mark.parametrize(
    ("final", "tier"),
    [
        (90.0, PriorityTier.A_PLUS),
        (80.0, PriorityTier.A_PLUS),
        (79.9, PriorityTier.A),
        (65.0, PriorityTier.A),
        (64.9, PriorityTier.B),
        (50.0, PriorityTier.B),
        (49.9, PriorityTier.C),
        (38.0, PriorityTier.C),
        (37.9, PriorityTier.D),
        (0.0, PriorityTier.D),
    ],
)
def test_priority_tier_bands(final: float, tier: PriorityTier) -> None:
    assert priority_tier(final, scoring=_SCORING) == tier


# --------------------------------------------------------------------------- #
# Strong alert & hard exclude.                                                #
# --------------------------------------------------------------------------- #
def test_strong_alert_tier1_sydney_multisite() -> None:
    job = _job(
        "National Quality Manager",
        "Multi-site national quality across 8 plants. HACCP, FSSC 22000, supplier quality.",
        location="Western Sydney NSW",
        salary_min=150000,
        salary_max=170000,
        salary_currency="AUD",
    )
    result = score_job(
        job,
        similarity=_similarity_for(80.0),
        scoring=_SCORING,
        profile=_PROFILE,
        sector="food manufacturing",
        is_tier1=True,
        posted_days_ago=2,
    )
    assert result.strong_alert is True
    reasons = " ".join(result.match_reasons).lower()
    assert "tier-1" in reasons
    assert "sydney" in reasons or "nsw" in reasons
    assert "multi-site" in reasons or "national" in reasons
    assert "130k" in reasons
    assert "< 7 days" in reasons


def test_strong_alert_none_when_no_trigger() -> None:
    job = _job(
        "Quality Coordinator",
        "Entry-level quality coordinator at a food factory.",
        location="Perth WA",
        salary_min=70000,
        salary_max=85000,
        salary_currency="AUD",
    )
    fired, reasons = strong_alert(
        job,
        category="qa_officer_coordinator",
        signals=set(),
        location_key="other_au",
        salary_pts=-10.0,
        posted_days_ago=30,
        is_tier1=False,
    )
    assert fired is False
    assert reasons == []


def test_hard_excluded_only_for_out_of_domain_plus_exclusion() -> None:
    soft = _job(
        "Software QA Engineer",
        "Automated test analyst for our SaaS platform. Selenium, CI/CD.",
        location="Sydney NSW",
    )
    result = score_job(soft, similarity=0.5, scoring=_SCORING, profile=_PROFILE, sector="software")
    assert result.hard_excluded is True
    assert result.components.industry == 0

    # In-domain food role with a junior exclusion signal is NOT hard-excluded.
    food = _job("QA Officer", "Entry-level QA officer, food factory.", location="Sydney")
    food_result = score_job(food, similarity=0.5, scoring=_SCORING, profile=_PROFILE, sector="food")
    assert "ex_junior" in food_result.signals
    assert food_result.hard_excluded is False


# --------------------------------------------------------------------------- #
# Resume tips.                                                                #
# --------------------------------------------------------------------------- #
def test_resume_tips_for_signals() -> None:
    signals = {"multi_site_national", "supplier_vendor_assurance", "food_safety_cert", "laboratory"}
    tips = resume_tips(signals, "site_quality_manager", "food manufacturing")
    assert 2 <= len(tips) <= 5
    joined = " ".join(tips).lower()
    assert "national" in joined
    assert "supplier" in joined or "co-packer" in joined
    assert "fssc" in joined or "haccp" in joined


def test_resume_tips_always_returns_something() -> None:
    tips = resume_tips(set(), "other", "")
    assert len(tips) >= 1
