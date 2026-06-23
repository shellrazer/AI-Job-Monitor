"""Scoring engine: turn a job + similarity into a calibrated :class:`ScoreResult`.

This is the heart of the matcher (spec §10-12). The final score is a weighted
sum of five 0-100 components (relevance-first; company-size an important lever)::

    final = 0.30*seniority + 0.25*semantic + 0.20*industry + 0.15*company + 0.10*location_salary

Design notes
------------
* **industry component.** The 0.20 ``industry`` component is the food-relevance
  signal: ``clamp(industry_points + sum(fired responsibility points), 0, 100)``.
  It *absorbs* the additive responsibility table (spec §10.3). The
  ``out_of_domain`` industry value is a large negative (-50) so a software /
  construction / NDIS / healthcare posting drives the component to zero even when
  a few generic responsibility signals fire. When both
  ``supplier_vendor_assurance`` and ``customer_retailer_audit`` fire we subtract
  the customer/retailer audit points once, because "audit" is double-counted by
  the two overlapping signal vocabularies. NO company-size term lives here.

* **company component.** The 0.15 ``company`` component is the standalone
  company-size lever. It maps the COMPANY_POINTS class to 0-100 via
  ``points/15*100`` (large_food_group 15→100, large_retail_food 12→80,
  mid_food_manufacturer 8→53, recruiter_anonymous_food 5→33, unclear 0→0). A
  ``is_tier1`` target forces at least large_food_group (100). The employer
  ``company_name`` is also matched against known AU Tier-1 food/FMCG/retail
  brands so aggregator-sourced jobs (SEEK/Jora/LinkedIn, no sector) still earn
  company-size credit when the employer is a major company.

* **Senior-specialist promotion.** A ``senior`` + in-domain quality / food-safety
  title that has NO manager/officer/coordinator head-noun (e.g. "Senior Food
  Quality Specialist") is promoted to the Senior QM tier when
  ``scoring.promote_senior_specialist`` is set. This is what lets the spec's
  Example 2 ("~82 / A") reconcile: without the promotion such a title would land
  in ``senior_specialist_quality_lead`` (10 pts) and miss tier A.

* **Worked-example calibration.** The spec's three worked examples are *tier*
  anchors. Only ``semantic`` is non-deterministic (it depends on the caller's
  embedding cosine), so the tests assert every RULE component (seniority,
  industry_company, location_salary) EXACTLY and pin the final tier under a
  mocked ``semantic`` value.

All table values are read from the supplied :class:`ScoringConfig`; when a table
is empty we fall back to the in-module defaults, which are identical to
``config/scoring.yaml``.
"""

from __future__ import annotations

import re
from typing import Any

from job_monitor.config import ProfileConfig, ScoringConfig
from job_monitor.models import Job, PriorityTier, ScoreComponents, ScoreResult

__all__ = [
    "canonical_seniority",
    "classify_company",
    "classify_industry",
    "classify_location",
    "compute_semantic",
    "detect_signals",
    "final_score",
    "is_australian_location",
    "is_quality_relevant",
    "is_recruiter_company",
    "priority_tier",
    "resume_tips",
    "score_company",
    "score_industry",
    "score_job",
    "score_location",
    "score_location_salary",
    "score_salary",
    "score_seniority",
    "strong_alert",
]


# --------------------------------------------------------------------------- #
# In-module defaults (mirror config/scoring.yaml exactly).                    #
# --------------------------------------------------------------------------- #
_DEFAULT_SENIORITY_POINTS: dict[str, int] = {
    "national_group_head": 30,
    "site_quality_manager": 28,
    "senior_quality_manager": 25,
    "food_safety_quality_manager": 25,
    "supplier_quality_manager": 22,
    "regulatory_affairs_manager_food": 20,
    "site_technical_manager": 20,
    "qa_manager": 15,
    "senior_specialist_quality_lead": 10,
    "qa_officer_coordinator": 3,
    "other": 0,
}

_DEFAULT_INDUSTRY_POINTS: dict[str, int] = {
    "food_manufacturing_fmcg": 25,
    "food_subsector": 25,
    "retailer_supplier_quality": 22,
    "foodservice": 15,
    "pharma_meddevice": 5,
    "out_of_domain": -50,
    "unclear": 0,
}

_DEFAULT_RESPONSIBILITY_POINTS: dict[str, int] = {
    "multi_site_national": 20,
    "manages_quality_team": 15,
    "laboratory": 10,
    "site_leadership": 15,
    "supplier_vendor_assurance": 15,
    "co_packer": 15,
    "customer_retailer_audit": 10,
    "food_safety_cert": 15,
    "regulatory_compliance": 10,
    "capa_nc_traceability": 10,
}

_DEFAULT_COMPANY_POINTS: dict[str, int] = {
    "large_food_group": 15,
    "large_retail_food": 12,
    "mid_food_manufacturer": 8,
    "recruiter_anonymous_food": 5,
    "unclear": 0,
}

_DEFAULT_LOCATION_POINTS: dict[str, int] = {
    "sydney_greater": 15,
    "nsw_regional": 10,
    "melbourne_brisbane": 8,
    "other_au": 5,
    "overseas": -20,
}

_DEFAULT_SALARY_BANDS: list[list[float]] = [
    [170000, 15],
    [150000, 12],
    [130000, 10],
    [110000, 5],
    [0, -10],
]

_DEFAULT_TIER_BANDS: list[list[Any]] = [
    [85, "A+"],
    [75, "A"],
    [60, "B"],
    [45, "C"],
    [0, "D"],
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _tokens(text: str) -> set[str]:
    """Word-order-insensitive token set of the lowercased text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _seniority_points(scoring: ScoringConfig) -> dict[str, int]:
    return scoring.seniority_points or _DEFAULT_SENIORITY_POINTS


def _industry_points(scoring: ScoringConfig) -> dict[str, int]:
    return scoring.industry_points or _DEFAULT_INDUSTRY_POINTS


def _responsibility_points(scoring: ScoringConfig) -> dict[str, int]:
    return scoring.responsibility_points or _DEFAULT_RESPONSIBILITY_POINTS


def _company_points(scoring: ScoringConfig) -> dict[str, int]:
    return scoring.company_points or _DEFAULT_COMPANY_POINTS


def _location_points(scoring: ScoringConfig) -> dict[str, int]:
    return scoring.location_points or _DEFAULT_LOCATION_POINTS


def _salary_bands(scoring: ScoringConfig) -> list[list[float]]:
    return scoring.salary_bands or _DEFAULT_SALARY_BANDS


def _tier_bands(scoring: ScoringConfig) -> list[list[Any]]:
    return scoring.tier_bands or _DEFAULT_TIER_BANDS


# --------------------------------------------------------------------------- #
# 1. Seniority (spec §10.1)                                                   #
# --------------------------------------------------------------------------- #
def canonical_seniority(title: str, *, promote_senior_specialist: bool = True) -> str:
    """Map a job *title* to a seniority category key (STRICT precedence).

    This is deliberately separate from :func:`dedup.normalize_title`: that
    produces a display/dedup string, this produces a scoring category. Matching
    is word-order-insensitive (token-set based); the FIRST rule that matches
    wins.
    """
    t = _tokens(title)
    raw = title.lower()

    def has(*words: str) -> bool:
        return all(w in t for w in words)

    # national / group head of quality
    if (("national" in t or "group" in t) and "quality" in t) or "head of quality" in raw:
        return "national_group_head"

    # site quality manager (but not a *technical* site role)
    if has("site", "quality", "manager") and "technical" not in t:
        return "site_quality_manager"

    # Promotion: senior + quality with no manager/officer/coordinator head-noun
    # and one of {food, safety, supplier, specialist}.
    if promote_senior_specialist:
        head_nouns = {"manager", "officer", "coordinator"}
        if (
            "senior" in t
            and "quality" in t
            and not (t & head_nouns)
            and (t & {"food", "safety", "supplier", "specialist"})
        ):
            return "senior_quality_manager"

    if has("senior", "quality", "manager"):
        return "senior_quality_manager"

    # food safety quality manager — all four tokens present, any order.
    if has("food", "safety", "quality", "manager"):
        return "food_safety_quality_manager"

    if ("supplier" in t or "vendor" in t) and ("quality" in t or "assurance" in t) and "manager" in t:
        return "supplier_quality_manager"

    if "regulatory" in t and ("affairs" in t or "compliance" in t) and "manager" in t:
        return "regulatory_affairs_manager_food"

    if (has("technical", "manager")) or "site technical" in raw:
        return "site_technical_manager"

    if ("qa" in t or "quality assurance" in raw or "quality" in t) and "manager" in t:
        return "qa_manager"

    if (has("senior", "specialist")) or "quality lead" in raw or "quality systems lead" in raw:
        return "senior_specialist_quality_lead"

    if ("qa" in t or "quality" in t) and ("officer" in t or "coordinator" in t):
        return "qa_officer_coordinator"

    return "other"


def score_seniority(category: str, *, scoring: ScoringConfig | None = None) -> float:
    """Seniority component (0-100): ``clamp(points * 100/30, 0, 100)``."""
    points = (_seniority_points(scoring) if scoring else _DEFAULT_SENIORITY_POINTS)
    pts = points.get(category, 0)
    return _clamp(pts * 100 / 30, 0, 100)


# --------------------------------------------------------------------------- #
# 1b. Quality-relevance gate (Part B)                                         #
# --------------------------------------------------------------------------- #
# Title-level quality / QA / food-safety vocabulary. Deliberately EXCLUDES bare
# workplace-safety wording ("safety", "WHS", "OHS", "work health and safety") —
# only "food safety" counts — so a WHS coordinator is not treated as relevant.
_QUALITY_TITLE = re.compile(
    r"\bquality\b|\bqa\b|q\.a|quality assurance|quality control|\bqc\b|"
    r"food safety|\bhaccp\b|\bbrcgs\b|\bbrc\b|\bsqf\b|\bfssc\b|\bgmp\b|"
    r"food technolog|quality systems|product integrity|technical manager|"
    r"regulatory affairs|vendor assurance|supplier quality|"
    r"quality and compliance|quality & compliance"
)


def is_quality_relevant(title: str, jd_text: str = "") -> bool:
    """True if the *title* indicates a quality / QA / food-safety / technical role.

    Relevance is decided on the TITLE only (the ``jd_text`` argument is accepted
    for signature stability / future use): either the title maps to a non-"other"
    seniority category, or it matches the quality-title vocabulary. Bare
    workplace-safety wording (WHS/OHS/"work health and safety") does NOT count —
    only "food safety" does.
    """
    if canonical_seniority(title) != "other":
        return True
    return bool(_QUALITY_TITLE.search(title.lower()))


# --------------------------------------------------------------------------- #
# 2. Signal detection (spec §10.3, §6.4)                                      #
# --------------------------------------------------------------------------- #
# Each signal maps to a list of regex patterns; the signal fires if ANY pattern
# matches the lowercased JD text. Be generous with synonyms.
_SIGNAL_PATTERNS: dict[str, list[str]] = {
    # --- responsibility signals (keys match responsibility_points) ---------- #
    "multi_site_national": [
        r"multi[\s-]?site",
        r"\bnational\b",
        r"\bgroup[\s-]?wide\b",
        r"across (?:multiple|several|all) (?:sites|plants|facilities)",
        r"\b\d+\s+(?:sites|plants|factories|facilities)\b",
        r"network of (?:sites|plants|facilities)",
    ],
    "manages_quality_team": [
        r"manage[s]? (?:a|the)? ?(?:quality|qa) team",
        r"lead[s]? (?:a|the)? ?(?:quality|qa) team",
        r"(?:quality|qa) team",
        r"team of (?:quality|qa)",
        r"direct reports",
        r"manage[s]? (?:a team of|staff)",
        r"people management",
    ],
    "laboratory": [
        r"\blaborator(?:y|ies)\b",
        r"\blab\b",
        r"\bqfs\b",
        r"micro(?:biolog\w*)? testing",
        r"laboratory management",
        r"nata",
    ],
    "site_leadership": [
        r"site leadership",
        r"site lead",
        r"plant (?:manager|leadership|lead)",
        r"operations leadership",
        r"factory (?:manager|leadership)",
        r"leadership team",
        r"senior leadership",
    ],
    "supplier_vendor_assurance": [
        r"supplier (?:quality|assurance|approval|management|audit)",
        r"vendor (?:quality|assurance|approval|management|audit)",
        r"raw material (?:approval|quality)",
        r"supplier development",
        r"incoming (?:goods|materials) (?:inspection|quality)",
    ],
    "co_packer": [
        r"co[\s-]?pack\w*",
        r"contract manufactur\w*",
        r"3rd party manufactur\w*",
        r"third[\s-]?party manufactur\w*",
        r"toll manufactur\w*",
        r"external manufactur\w*",
    ],
    "customer_retailer_audit": [
        r"customer audit",
        r"retailer audit",
        r"second[\s-]?party audit",
        r"\baudit readiness\b",
        r"\baudits?\b",
        r"woolworths|coles|aldi|metcash",
        r"customer complaint",
    ],
    "food_safety_cert": [
        r"\bhaccp\b",
        r"\bgmp\b",
        r"fssc\s?22000",
        r"\bsqf\b",
        r"\bbrc(?:gs)?\b",
        r"\bgfsi\b",
        r"\bfsms\b",
        r"food safety (?:management|system|standard|cert\w*|program)?",
        r"\bfood safety\b",
        r"iso\s?22000",
    ],
    "regulatory_compliance": [
        r"regulatory (?:compliance|affairs|requirements)",
        r"\bcompliance\b",
        r"food standards",
        r"\blabel(?:l?ing)?\b",
        r"legislativ\w*",
        r"\bfsanz\b",
    ],
    "capa_nc_traceability": [
        r"\bcapa\b",
        r"corrective (?:and|&)? ?preventive action",
        r"non[\s-]?conformanc\w*",
        r"\bnc(?:r)?\b",
        r"traceab\w*",
        r"\brecall\b",
        r"risk (?:assessment|management|analysis)",
        r"root cause",
    ],
    # --- alert-only signal (NOT in responsibility_points) ------------------- #
    "alert_quality_systems_ownership": [
        r"technical (?:leadership|lead|manager|director)",
        r"quality systems? (?:ownership|owner|lead|management)",
        r"own(?:s|ership of)? the quality (?:system|management system)",
        r"head of (?:technical|quality)",
        r"quality manager",
    ],
    # --- exclusion signals (spec §6.4) -------------------------------------- #
    "ex_software_qa": [
        r"software (?:qa|quality|test\w*)",
        r"\bsdet\b",
        r"test automation",
        r"automated test\w*",
        r"\bselenium\b",
        r"\bcypress\b",
        r"\bci/cd\b",
        r"qa engineer",
        r"test analyst",
        r"saas (?:platform|product)",
    ],
    "ex_construction": [
        r"\bconstruction\b",
        r"\bcivil\b",
        r"\bbuilding\b",
        r"\binfrastructure\b",
        r"\bengineering project\b",
        r"\bitp\b",
        r"\bcommissioning\b",
    ],
    "ex_ndis": [
        r"\bndis\b",
        r"disability (?:services|support)",
        r"support coordinat\w*",
    ],
    "ex_health_aged_edu": [
        r"\baged care\b",
        r"\bhealthcare\b",
        r"\bhospital\b",
        r"\bclinical\b",
        r"\bnursing\b",
        r"\bpatient\b",
        r"\bschool\b",
        r"\buniversit\w*\b",
        r"\beducation\b",
        r"\bchildcare\b",
    ],
    "ex_call_centre": [
        r"call cent(?:re|er)",
        r"contact cent(?:re|er)",
        r"customer service (?:representative|agent)",
        r"\bbpo\b",
    ],
    "ex_lab_tech": [
        r"laboratory technician",
        r"lab technician",
        r"lab assistant",
        r"laboratory assistant",
        r"sample preparation",
    ],
    "ex_junior": [
        r"\bjunior\b",
        r"\bgraduate\b",
        r"\bentry[\s-]?level\b",
        r"\bintern(?:ship)?\b",
        r"\btrainee\b",
        r"\bcadet\b",
    ],
    "ex_operator": [
        r"machine operator",
        r"production operator",
        r"line operator",
        r"process worker",
        r"\bforklift\b",
        r"production worker",
    ],
    "ex_food_service_role": [
        r"\bchef\b",
        r"\bcook\b",
        r"\bbarista\b",
        r"\bwait(?:er|ress|staff)\b",
        r"kitchen hand",
        r"\bhospitality\b",
        r"front of house",
    ],
}

_COMPILED_SIGNALS: dict[str, list[re.Pattern[str]]] = {
    name: [re.compile(p) for p in patterns] for name, patterns in _SIGNAL_PATTERNS.items()
}

# Responsibility-only signal keys (folded into the 0.20 component).
_RESPONSIBILITY_SIGNALS = tuple(_DEFAULT_RESPONSIBILITY_POINTS.keys())


def detect_signals(jd_text: str) -> set[str]:
    """Return the set of fired signal names for the (lowercased) JD text."""
    lowered = jd_text.lower()
    return {name for name, patterns in _COMPILED_SIGNALS.items() if any(p.search(lowered) for p in patterns)}


# --------------------------------------------------------------------------- #
# 3. Industry / company (spec §10.2-§10.4, the 0.20 component)                #
# --------------------------------------------------------------------------- #
_OUT_OF_DOMAIN = re.compile(
    r"\bsoftware\b|\bsaas\b|\bsdet\b|test automation|\bconstruction\b|\bcivil\b|\bndis\b|"
    r"disability (?:services|support)|\bhealthcare\b|\bhospital\b|\bclinical\b|\baged care\b|"
    r"\bnursing\b|education quality|\bchildcare\b"
)
_FOOD_SUBSECTOR = re.compile(
    r"\bdairy\b|\bmeat\b|\bpoultry\b|\bbaker\w*|\bbeverage\b|\bgrain\b|\bsnack\b|\bconfection\w*|"
    r"\bbrew\w*|\bwiner\w*|\bseafood\b|\bsmallgoods\b"
)
_FOOD_MANUFACTURING = re.compile(
    r"\bfood (?:manufactur\w*|production|processing|factory)\b|\bfmcg\b|\bfood manufactur\w*|"
    r"\bfood (?:and|&) beverage\b|\bfood production\b"
)
_RETAILER_SUPPLIER = re.compile(
    r"\bretail\w*|\bsupermarket\b|\bgrocer\w*|woolworths|coles|aldi|metcash|\bprivate label\b|"
    r"own brand|supplier quality"
)
_FOODSERVICE = re.compile(
    r"\bfoodservice\b|\bfood service\b|meal kit|central kitchen|\bcatering\b|\bqsr\b|\brestaurant\b"
)
_PHARMA = re.compile(r"\bpharma\w*|\bmedical device\b|\bmeddevice\b|\bnutraceutical\b|\btga\b")


def classify_industry(sector: str, jd_text: str) -> str:
    """Single best industry key (STRICT precedence; out_of_domain wins).

    The ``sector`` field is the authoritative controlled vocabulary from
    ``companies.yaml`` (e.g. "food manufacturing", "retail group"). When it
    indicates retail/foodservice/pharma we honor that even if the JD also
    mentions generic food-safety wording, so a retailer's supplier-quality role
    is not mis-bucketed as food manufacturing.
    """
    blob = f"{sector}\n{jd_text}".lower()
    sec = sector.lower()

    if _OUT_OF_DOMAIN.search(blob):
        return "out_of_domain"
    if _FOOD_SUBSECTOR.search(blob):
        return "food_subsector"

    # Sector-driven classification (controlled vocab wins over loose JD text).
    if _RETAILER_SUPPLIER.search(sec):
        return "retailer_supplier_quality"
    if _FOODSERVICE.search(sec):
        return "foodservice"
    if _PHARMA.search(sec):
        return "pharma_meddevice"
    if _FOOD_MANUFACTURING.search(sec) or re.search(r"\bfood\b|\bfmcg\b", sec):
        return "food_manufacturing_fmcg"

    # Fall back to JD-text classification.
    if _FOOD_MANUFACTURING.search(blob):
        return "food_manufacturing_fmcg"
    if _RETAILER_SUPPLIER.search(blob):
        return "retailer_supplier_quality"
    if _FOODSERVICE.search(blob):
        return "foodservice"
    if _PHARMA.search(blob):
        return "pharma_meddevice"
    return "unclear"


# Known major AU food / FMCG manufacturers & multinationals -> large_food_group.
_TIER1_FOOD_GROUP = re.compile(
    r"\bnestl|\bpepsico\b|\bmars\b|mondelez|cadbury|kellanova|kellogg|\bbega\b|saputo|"
    r"fonterra|lactalis|\blion\b|\basahi\b|coca[\s-]?cola|\bccep\b|amatil|suntory|\bjbs\b|"
    r"\bteys\b|ingham|goodman fielder|george weston|tip[\s-]?top|arnott|graincorp|sunrice|"
    r"cargill|\bkerry\b|hellofresh|treasury wine"
)
# Known major AU food retailers / supermarkets -> large_retail_food.
_TIER1_RETAIL = re.compile(
    r"woolworths|\bcoles\b|\baldi\b|metcash|costco"
)
# Known recruiters / staffing agencies -> recruiter_anonymous_food.
_RECRUITER_NAMES = re.compile(
    r"michael page|\bhays\b|six degrees|peoplefusion|blackbook|randstad|hudson|\bmpau\b|"
    r"robert walters|robert half|adecco|\bhello recruitment\b|recruit\w*|\bagency\b|"
    r"talent\b|staffing"
)


def is_recruiter_company(company_name: str | None) -> bool:
    """True if the posting's company name looks like a recruiter / staffing agency
    (so the real employer is undisclosed). Used to flag jobs in the report."""
    return bool(company_name and _RECRUITER_NAMES.search(company_name.lower()))


def classify_company(
    sector: str,
    jd_text: str,
    is_recruiter: bool,
    company_name: str = "",
) -> str:
    """Single best company-type key from company name + sector keywords + recruiter flag.

    The employer ``company_name`` is matched first against a curated set of known
    AU Tier-1 food/FMCG/retail brands and recruiter/agency names. This lets
    aggregator-sourced jobs (SEEK/Jora/LinkedIn) that carry no ``sector`` still
    earn the right company-size credit. Failing that, classification falls back
    to the ``sector`` controlled vocabulary, then JD text / recruiter flag.
    """
    name = company_name.lower()
    sec = sector.lower()
    blob = f"{sector}\n{jd_text}".lower()

    # 1. Known-employer name patterns (works without a sector).
    if _TIER1_FOOD_GROUP.search(name):
        return "large_food_group"
    if _TIER1_RETAIL.search(name):
        return "large_retail_food"
    if _RECRUITER_NAMES.search(name):
        return "recruiter_anonymous_food"

    # 2. Sector-driven classification (controlled vocab).
    if re.search(
        r"food (?:manufactur\w*|group)|\bfmcg\b|multinational|global (?:food|fmcg)|large (?:food )?group",
        sec,
    ):
        return "large_food_group"
    if re.search(r"\bretail\w*|supermarket|grocer\w*|major retailer", sec):
        return "large_retail_food"
    if re.search(r"\bfood\b|\bbeverage\b|\bmanufactur\w*", sec):
        return "mid_food_manufacturer"

    # 3. Fall back to JD text / recruiter flag.
    if re.search(r"large (?:food )?group|food group|multinational|global (?:food|fmcg)", blob):
        return "large_food_group"
    if re.search(r"\bretail\w*|supermarket|grocer\w*|major retailer", blob):
        return "large_retail_food"
    if is_recruiter:
        return "recruiter_anonymous_food"
    if re.search(r"\bfood\b|\bbeverage\b|\bmanufactur\w*|\bfmcg\b", blob):
        return "mid_food_manufacturer"
    return "unclear"


def score_industry(
    sector: str,
    signals: set[str],
    jd_text: str,
    *,
    scoring: ScoringConfig | None = None,
) -> float:
    """The 0.20 ``industry`` component: ``clamp(industry + responsibility_sum, 0, 100)``.

    Food-relevance only: the industry table plus the additive responsibility
    table (spec §10.3). No company-size term lives here.
    """
    industry_points = _industry_points(scoring) if scoring else _DEFAULT_INDUSTRY_POINTS
    resp_points = _responsibility_points(scoring) if scoring else _DEFAULT_RESPONSIBILITY_POINTS

    industry = industry_points.get(classify_industry(sector, jd_text), 0)

    responsibility_sum = sum(resp_points.get(sig, 0) for sig in signals if sig in _RESPONSIBILITY_SIGNALS)
    # Avoid double-counting the generic "audit" vocabulary: if a supplier/vendor
    # assurance signal already fired, the customer/retailer audit points overlap.
    if "supplier_vendor_assurance" in signals and "customer_retailer_audit" in signals:
        responsibility_sum -= resp_points.get("customer_retailer_audit", 0)

    return _clamp(industry + responsibility_sum, 0, 100)


def score_company(
    sector: str,
    jd_text: str,
    company_name: str,
    is_recruiter: bool,
    is_tier1: bool,
    *,
    scoring: ScoringConfig | None = None,
) -> float:
    """The 0.15 ``company`` component: company-size class mapped to 0-100.

    Maps the COMPANY_POINTS class via ``points/15*100`` (large_food_group 15→100,
    large_retail_food 12→80, mid_food_manufacturer 8→53, recruiter_anonymous_food
    5→33, unclear 0→0). A ``is_tier1`` target company is forced to at least
    large_food_group (100). The employer name is classified too (see
    :func:`classify_company`) so aggregator jobs can still earn company credit.
    """
    company_points = _company_points(scoring) if scoring else _DEFAULT_COMPANY_POINTS
    large_group_pts = company_points.get("large_food_group", 15)

    key = classify_company(sector, jd_text, is_recruiter, company_name)
    pts = company_points.get(key, 0)
    if is_tier1:
        pts = max(pts, large_group_pts)

    # Normalize on the large_food_group anchor (15 by default) so that class -> 100.
    denom = large_group_pts if large_group_pts else 15
    return _clamp(pts / denom * 100, 0, 100)


# --------------------------------------------------------------------------- #
# 4. Location / salary (spec §10.5-§10.6, the 0.10 component)                 #
# --------------------------------------------------------------------------- #
# --- AU location allowlist (Part B) ---------------------------------------- #
# We classify as overseas by EXCLUSION: a non-empty location with no Australian
# signal at all is overseas. This is robust against global ATS tenants that post
# US/EU/Asia roles ("Franklin, WI", "Ho Chi Minh City", "Rayong Plant") which an
# explicit blocklist could never fully enumerate.
_AU_REMOTE_MARKERS = (
    "remote",
    "work from home",
    "work-from-home",
    "wfh",
    "anywhere",
    "hybrid",
)
_AU_STATE_NAMES = (
    "new south wales",
    "victoria",
    "queensland",
    "western australia",
    "south australia",
    "tasmania",
    "northern territory",
    "australian capital territory",
)
_AU_STATE_CODES = re.compile(r"\b(?:nsw|vic|qld|wa|sa|tas|nt|act)\b")
# Greater Sydney suburbs / hubs (-> sydney_greater).
_SYDNEY_SUBURBS = (
    "sydney", "parramatta", "chatswood", "north sydney", "macquarie park", "north ryde",
    "ryde", "rhodes", "epping", "eastwood", "hornsby", "st leonards", "lane cove", "artarmon",
    "crows nest", "mascot", "alexandria", "rosebery", "waterloo", "surry hills", "redfern",
    "pyrmont", "ultimo", "homebush", "lidcombe", "auburn", "silverwater", "rydalmere", "granville",
    "merrylands", "smithfield", "wetherill park", "fairfield", "liverpool", "prestons", "moorebank",
    "ingleburn", "minto", "campbelltown", "penrith", "st marys", "blacktown", "seven hills",
    "eastern creek", "rooty hill", "mount druitt", "castle hill", "baulkham hills", "bella vista",
    "norwest", "kellyville", "rouse hill", "frenchs forest", "brookvale", "dee why", "bankstown",
    "padstow", "revesby", "kingsgrove", "rockdale", "kogarah", "hurstville", "miranda", "caringbah",
    "taren point", "marrickville", "leichhardt", "gladesville", "hunters hill", "wentworthville",
    "greystanes", "erskine park", "kemps creek", "marsden park", "smeaton grange", "gregory hills",
    "edmondson park", "chipping norton", "villawood", "chullora", "olympic park", "western sydney",
    "greater sydney", "northern beaches", "sutherland",
)
# NSW regional towns (-> nsw_regional).
_NSW_REGIONAL = (
    "newcastle", "wollongong", "gosford", "central coast", "wyong", "tuggerah", "maitland",
    "cessnock", "port macquarie", "coffs harbour", "tamworth", "orange", "bathurst", "dubbo",
    "wagga", "albury", "lismore", "ballina", "byron", "nowra", "goulburn", "queanbeyan", "griffith",
    "leeton", "armidale", "taree", "kempsey", "grafton", "broken hill", "mudgee", "parkes", "forbes",
    "moree", "narrabri", "gunnedah", "singleton", "muswellbrook", "scone", "raymond terrace",
    "nelson bay", "batemans bay", "ulladulla", "bega", "eden", "cooma", "regional nsw", "nsw regional",
)
# Non-NSW AU cities/towns (-> melbourne_brisbane / other_au). Used for AU recognition
# and to win over a bare Sydney-suburb-name collision (e.g. "Manly QLD").
_OTHER_AU_CITIES = (
    "melbourne", "brisbane", "perth", "adelaide", "canberra", "hobart", "darwin", "gold coast",
    "sunshine coast", "geelong", "ballarat", "bendigo", "toowoomba", "townsville", "cairns",
    "mackay", "rockhampton", "launceston", "bunbury", "tatura", "shepparton", "dandenong",
    "yatala", "carole park", "tingalpa",
)
_AU_CITIES = _SYDNEY_SUBURBS + _NSW_REGIONAL + _OTHER_AU_CITIES
_OTHER_STATE_RE = re.compile(
    r"melbourne|brisbane|perth|adelaide|canberra|hobart|darwin|gold coast|sunshine coast|"
    r"geelong|ballarat|bendigo|toowoomba|townsville|cairns|mackay|rockhampton|launceston|bunbury|"
    r"tatura|shepparton|dandenong|yatala|carole park|tingalpa|"
    r"\bvic\b|victoria|\bqld\b|queensland|\bwa\b|western australia|\bsa\b|south australia|"
    r"\btas\b|tasmania|\bnt\b|northern territory|\bact\b|australian capital territory"
)
_NSW_RE = re.compile(r"\bnsw\b|new south wales")
_AU_STANDALONE_AUS = re.compile(r"\baus\b")


def is_australian_location(location: str | None) -> bool:
    """True if *location* carries any Australian signal (or is ambiguous/empty).

    Empty / unknown locations return True (ambiguous — we don't over-filter). A
    non-empty location is Australian if it mentions Australia, a remote/WFH
    marker, an AU state (full name or word-boundary code), or a known AU
    city/region. Everything else (a real, non-AU place) returns False.
    """
    if not location or not location.strip():
        return True
    loc = location.lower()

    if "australia" in loc or "australian" in loc or _AU_STANDALONE_AUS.search(loc):
        return True
    if any(marker in loc for marker in _AU_REMOTE_MARKERS):
        return True
    if any(name in loc for name in _AU_STATE_NAMES):
        return True
    if _AU_STATE_CODES.search(loc):
        return True
    return any(city in loc for city in _AU_CITIES)


def classify_location(location: str | None) -> str:
    """Single best location key (allowlist-based AU gate).

    A non-empty location with NO Australian signal is ``overseas`` (and is
    subsequently filtered by the pipeline). Otherwise we bucket the AU location
    into sydney_greater / nsw_regional / melbourne_brisbane / other_au; empty /
    ambiguous locations stay in the neutral ``other_au`` bucket.
    """
    if location and location.strip() and not is_australian_location(location):
        return "overseas"
    if not location or not location.strip():
        return "other_au"
    loc = location.lower()
    has_nsw = bool(_NSW_RE.search(loc))
    has_other_state = bool(_OTHER_STATE_RE.search(loc))

    # A clear OTHER-state signal with NO NSW signal wins first, so e.g. "Manly QLD"
    # (a Sydney suburb name) is correctly Brisbane, not Sydney.
    if has_other_state and not has_nsw:
        return "melbourne_brisbane"
    if "greater sydney" in loc or "western sydney" in loc or any(s in loc for s in _SYDNEY_SUBURBS):
        return "sydney_greater"
    if any(t in loc for t in _NSW_REGIONAL):
        return "nsw_regional"
    if has_nsw:  # NSW metro not matched to a specific suburb above
        return "sydney_greater"
    if has_other_state:
        return "melbourne_brisbane"
    # Recognized-AU but no specific region (e.g. bare "Australia", remote): ambiguous.
    return "other_au"


def score_location(location: str | None, *, scoring: ScoringConfig | None = None) -> float:
    """Raw location points for the classified location."""
    points = _location_points(scoring) if scoring else _DEFAULT_LOCATION_POINTS
    return points.get(classify_location(location), 0)


def score_salary(
    salary_min: int | None,
    salary_max: int | None,
    disclosed: bool,
    *,
    scoring: ScoringConfig | None = None,
) -> float:
    """Raw salary points: band the upper bound (fallback to min) high->low.

    Not disclosed -> 0 (neutral). The documented neutral gap (100k-110k, see
    ``config/scoring.yaml``) falls through to 0 rather than the ``[0, -10]``
    catch-all, which only penalizes genuinely low salaries (< 100k).
    """
    if not disclosed:
        return 0.0
    amount = salary_max if salary_max is not None else salary_min
    if amount is None:
        return 0.0
    # Documented neutral gap: 100k-110k is neither rewarded nor penalized.
    if 100000 <= amount < 110000:
        return 0.0
    bands = _salary_bands(scoring) if scoring else _DEFAULT_SALARY_BANDS
    for lower, points in sorted(bands, key=lambda b: b[0], reverse=True):
        if amount >= lower:
            return float(points)
    return 0.0


def score_location_salary(
    location: str | None,
    salary_min: int | None,
    salary_max: int | None,
    disclosed: bool,
    *,
    scoring: ScoringConfig | None = None,
) -> float:
    """The 0.10 component: rescale ``loc + sal`` from [-30, 30] into [0, 100].

    ``clamp(((loc_pts + sal_pts) + 30) / 60 * 100, 0, 100)``.
    """
    loc_pts = score_location(location, scoring=scoring)
    sal_pts = score_salary(salary_min, salary_max, disclosed, scoring=scoring)
    return _clamp(((loc_pts + sal_pts) + 30) / 60 * 100, 0, 100)


# --------------------------------------------------------------------------- #
# 5. Semantic (spec §10 — the 0.40 component)                                 #
# --------------------------------------------------------------------------- #
def compute_semantic(similarity: float) -> float:
    """Rescale cosine similarity in [-1, 1] into a 0-100 semantic score.

    ``clamp((similarity - 0.0) / (0.8 - 0.0), 0, 1) * 100`` — a cosine of 0.8 or
    above saturates at 100; 0 or below is 0.
    """
    return _clamp((similarity - 0.0) / (0.8 - 0.0), 0, 1) * 100


# --------------------------------------------------------------------------- #
# 6. Final score + tier (spec §11)                                            #
# --------------------------------------------------------------------------- #
def final_score(components: ScoreComponents, *, scoring: ScoringConfig | None = None) -> float:
    """Weighted sum of the five components, rounded to 1 dp."""
    if scoring is not None:
        w = scoring.weights
        weights = (w.semantic, w.seniority, w.industry, w.company, w.location_salary)
    else:
        weights = (0.25, 0.30, 0.20, 0.15, 0.10)
    total = (
        weights[0] * components.semantic
        + weights[1] * components.seniority
        + weights[2] * components.industry
        + weights[3] * components.company
        + weights[4] * components.location_salary
    )
    return round(total, 1)


def priority_tier(final: float, *, scoring: ScoringConfig | None = None) -> PriorityTier:
    """Map a final score to a :class:`PriorityTier` (tier_bands high->low)."""
    bands = _tier_bands(scoring) if scoring else _DEFAULT_TIER_BANDS
    for band in sorted(bands, key=lambda b: float(b[0]), reverse=True):
        minimum, tier = float(band[0]), band[1]
        if final >= minimum:
            return PriorityTier(tier)
    return PriorityTier.D


# --------------------------------------------------------------------------- #
# 7. Strong alert (spec §12)                                                  #
# --------------------------------------------------------------------------- #
_STRONG_ALERT_CATEGORIES = {
    "national_group_head",
    "site_quality_manager",
    "senior_quality_manager",
    "food_safety_quality_manager",
}


def strong_alert(
    job: Job,
    category: str,
    signals: set[str],
    location_key: str,
    salary_pts: float,
    posted_days_ago: int | None,
    is_tier1: bool,
) -> tuple[bool, list[str]]:
    """Decide whether this job earns a strong alert, with human-readable reasons."""
    reasons: list[str] = []

    if category in _STRONG_ALERT_CATEGORIES:
        reasons.append(f"High-fit seniority: {category.replace('_', ' ')}")
    if is_tier1:
        reasons.append("Tier-1 target company")
    if location_key in {"sydney_greater", "nsw_regional"}:
        reasons.append("Located in Sydney / NSW (preferred region)")
    if salary_pts >= 10:
        reasons.append("Salary >= $130k")
    if "multi_site_national" in signals:
        reasons.append("Multi-site / national quality scope")
    if "supplier_vendor_assurance" in signals or "co_packer" in signals:
        reasons.append("Supplier / co-packer quality scope")
    if "site_leadership" in signals or "alert_quality_systems_ownership" in signals:
        reasons.append("Site leadership / quality systems ownership")
    if "food_safety_cert" in signals:
        reasons.append("Food-safety certification scope (HACCP/FSSC/BRC/SQF)")
    if posted_days_ago is not None and posted_days_ago < 7:
        reasons.append("Freshly posted (< 7 days)")

    return (bool(reasons), reasons)


# --------------------------------------------------------------------------- #
# 8 + 9. Resume tips (spec §12)                                               #
# --------------------------------------------------------------------------- #
def resume_tips(signals: set[str], category: str, sector: str) -> list[str]:
    """Concise, rule-based bullets advising which experience to emphasize."""
    tips: list[str] = []

    if "multi_site_national" in signals:
        tips.append("Emphasize PepsiCo national quality across 8 plants")
    if "supplier_vendor_assurance" in signals or "co_packer" in signals:
        tips.append("Highlight co-packer & supplier quality from PepsiCo/Bunge")
    if "food_safety_cert" in signals:
        tips.append("Lead with FSSC 22000, HACCP, audit readiness")
    if "laboratory" in signals:
        tips.append("Highlight Bunge QFS lab management")
    if "manages_quality_team" in signals or "site_leadership" in signals:
        tips.append("Foreground team leadership and site-level QA ownership")
    if "regulatory_compliance" in signals or category == "regulatory_affairs_manager_food":
        tips.append("Surface regulatory/compliance and labelling experience")
    if "capa_nc_traceability" in signals:
        tips.append("Detail CAPA, non-conformance, traceability and recall readiness")

    if not tips:
        # Always give the candidate something actionable.
        tips.append("Tailor the summary to senior food-quality leadership outcomes")

    return tips[:5]


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def score_job(
    job: Job,
    *,
    similarity: float,
    scoring: ScoringConfig,
    profile: ProfileConfig,
    sector: str = "",
    company_name: str = "",
    is_tier1: bool = False,
    is_recruiter: bool = False,
    posted_days_ago: int | None = None,
) -> ScoreResult:
    """Score a single job into a :class:`ScoreResult` (spec §10-12).

    ``similarity`` is the cosine in [-1, 1] between the job and the candidate
    profile embeddings, computed by the caller. ``company_name`` is the employer
    name (used for company-size classification of aggregator jobs). ``profile``
    is accepted for future profile-aware tuning and keeps the signature stable.
    """
    jd_text = f"{job.title}\n{job.description or ''}"
    company_name = company_name or job.company_name

    # Components.
    category = canonical_seniority(job.title, promote_senior_specialist=scoring.promote_senior_specialist)
    signals = detect_signals(jd_text)

    semantic = compute_semantic(similarity)
    seniority = score_seniority(category, scoring=scoring)
    industry = score_industry(sector, signals, jd_text, scoring=scoring)
    company = score_company(sector, jd_text, company_name, is_recruiter, is_tier1, scoring=scoring)
    location_salary = score_location_salary(
        job.location, job.salary_min, job.salary_max, job.salary_currency is not None, scoring=scoring
    )

    components = ScoreComponents(
        semantic=semantic,
        seniority=seniority,
        industry=industry,
        company=company,
        location_salary=location_salary,
    )
    final = final_score(components, scoring=scoring)
    tier = priority_tier(final, scoring=scoring)

    # Strong alert (uses raw location/salary classifications, not the rescaled
    # component, so the thresholds in the spec apply directly).
    location_key = classify_location(job.location)
    salary_pts = score_salary(job.salary_min, job.salary_max, job.salary_currency is not None, scoring=scoring)
    alert, reasons = strong_alert(
        job, category, signals, location_key, salary_pts, posted_days_ago, is_tier1
    )

    # Hard-exclude: out_of_domain industry AND any exclusion signal fired.
    industry_key = classify_industry(sector, jd_text)
    hard_excluded = industry_key == "out_of_domain" and any(s.startswith("ex_") for s in signals)

    tips = resume_tips(signals, category, sector)

    return ScoreResult(
        final_score=final,
        priority_tier=tier,
        strong_alert=alert,
        components=components,
        match_reasons=reasons,
        resume_tips=tips,
        category=category,
        signals=signals,
        hard_excluded=hard_excluded,
    )
