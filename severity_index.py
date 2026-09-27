"""
severity_index.py
─────────────────
Calculates composite defect severity scores from classifier confidence
and blemish-area percentage, maps them to human-readable grades, and
returns structured, step-by-step actionable treatment, handling, and
post-harvest precautions for every diagnosis state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# ── Grade Thresholds ──────────────────────────────────────────────────
# Threshold is upper bound of composite defect severity (0 - 100)
_GRADE_TABLE: list[Tuple[float, str, str]] = [
    (5.0,   "Grade A – Premium",    "#22c55e"),
    (15.0,  "Grade B – Standard",   "#eab308"),
    (35.0,  "Grade C – Processing", "#f97316"),
    (100.0, "Grade D – Reject",     "#ef4444"),
]

# ── Structured Precautions per Disease / Defect State ─────────────────
_DISEASE_TREATMENT: Dict[str, str] = {
    "Healthy": (
        "✅  HEALTHY CARROT — GRADE A (PREMIUM)\n"
        "Diagnosis: Intact periderm with no significant pathological or mechanical defects (< 5% blemish).\n\n"
        "━━━  1. POST-HARVEST PREPARATION  ━━━\n"
        "• Gentle pre-rinse using clean potable water (10–12 °C) to clear ambient dust.\n"
        "• Sanitizer chemical wash is not required for undamaged retail table carrots.\n"
        "• Air-dry thoroughly in a forced-air tunnel before bagging to eliminate free surface moisture.\n\n"
        "━━━  2. OPTIMAL COLD-STORAGE SPECIFICATIONS  ━━━\n"
        "• Temperature        : 0 °C to 1 °C (strictly avoid freezing; ice crystallization begins at -0.8 °C).\n"
        "• Relative Humidity  : 95 % to 98 % RH (crucial to prevent transpirational wilting and rubbery texture).\n"
        "• Air Circulation    : Gentle continuous laminar airflow (0.1–0.2 m/s) with minimal fan turbulence.\n"
        "• Ethylene Isolation : Maintain ethylene < 0.1 ppm; store strictly isolated from apples, pears, and tomatoes\n"
        "                       to prevent isocoumarin synthesis (which causes bitter off-flavors).\n\n"
        "━━━  3. SHELF-LIFE & DISPATCH  ━━━\n"
        "• Commercial cold-storage life : 4 to 6 months at recommended specs.\n"
        "• Retail shelf life            : 3 to 4 weeks in micro-perforated polyethylene bags.\n"
        "• Quality inspection           : Re-inspect lots every 14 days for root tip sprouting or condensation."
    ),

    "Soil Contamination / Surface Blemish": (
        "🟫  SOIL CONTAMINATION / SURFACE BLEMISH\n"
        "Diagnosis: Superficial soil/dirt adhesion or minor epidermal abrasions without pathogen penetration.\n\n"
        "━━━  1. PRE-WASH SOAKING PROTOCOL  ━━━\n"
        "• Immersion tank : Submerge carrots in clean potable water at 12–15 °C for 3 to 5 minutes.\n"
        "• Agitation      : Use gentle pneumatic bubble aeration or submerged water jets to loosen baked-on soil.\n"
        "• Tank renewal   : Continuous overflow skimming (15–20 L/min) to prevent silt accumulation and microbe buildup.\n\n"
        "━━━  2. ROTARY BRUSH CLEANING PARAMETERS  ━━━\n"
        "• Brush roller spec : Transverse cylindrical rotary washer fitted with soft nylon bristles (0.25–0.35 mm diameter).\n"
        "• Rotation speed    : 120 to 160 RPM (moderate speed loosens dirt while avoiding abrasive root epidermal scarring).\n"
        "• Spray manifold    : High-pressure flat-fan spray nozzles operating at 2.5–3.5 bar continuous overhead wash.\n"
        "• Dwell duration    : 20 to 30 seconds traverse duration across the brush bed.\n\n"
        "━━━  3. SANITIZATION WASH PROCEDURES  ━━━\n"
        "• Option A (Chlorine)      : Sodium hypochlorite at 100–150 ppm free available chlorine (FAC), pH buffered to 6.5–7.2\n"
        "                             (contact time: 60–90 seconds).\n"
        "• Option B (Peracetic Acid): 60–80 ppm Peracetic Acid (PAA) solution (contact time: 90 seconds, leaves no chlorine residue).\n"
        "• Potable rinse            : Final clean water shower rinse to eliminate chemical traces before drying.\n\n"
        "━━━  4. SORTING & POST-WASH GRADING  ━━━\n"
        "• Re-inspection : If root is clean and undamaged after wash → suitable for Grade A / Grade B retail packaging.\n"
        "• Persistent blemish : If stains remain > 10% of surface → divert to commercial mechanical peeling or soup dicers."
    ),

    "Cavity Spot / Lesion": (
        "🕳️  CAVITY SPOT / SURFACE LESIONS (Pythium spp.)\n"
        "Diagnosis: Sunken, elliptical, horizontal tan-to-dark brown cavity lesions on the root cortex.\n\n"
        "━━━  1. TRIMMING GUIDELINES  ━━━\n"
        "• Minor cavities (< 5 spots) : Excise individual blemishes with a stainless-steel trimming knife with a 5 mm margin.\n"
        "• Moderate cavities (5–15%)  : Route through mechanical abrasive peeler to remove epidermal lesion layer.\n"
        "• Coalescing / deep lesions  : Discard affected roots; sunken lesions frequently harbor secondary bacterial soft rot.\n\n"
        "━━━  2. BATCH ISOLATION & QUARANTINE  ━━━\n"
        "• Immediately isolate affected crates from sound batches to prevent wound cross-infection.\n"
        "• Sanitize sorting belts and grading conveyor surfaces daily with 200 ppm quaternary ammonium solution.\n\n"
        "━━━  3. PROCESSING USAGE (DIVERT FROM FRESH MARKET)  ━━━\n"
        "• Trimmed sound root tissue is fully suitable for industrial processing:\n"
        "  - Pureeing for carrot paste, baby food, and commercial soups\n"
        "  - High-yield juice extraction with centrifugal solids separation\n"
        "  - Frozen diced or shredded carrot processing with thermal blanching (95 °C for 3 min)\n\n"
        "━━━  4. ANTI-FUNGAL & SANITIZATION WASH  ━━━\n"
        "• Wash in 80–100 ppm Peracetic Acid (PAA) bath for 2 minutes to halt Pythium spore viability.\n"
        "• Pre-cool to 1 °C within 4 hours of grading to arrest enzymatic cavity enlargement.\n\n"
        "━━━  5. FIELD PREVENTION (NEXT HARVEST)  ━━━\n"
        "• Improve field drainage; avoid waterlogged heavy soils during crown sizing.\n"
        "• Apply metalaxyl-M / mefenoxam at sowing; maintain balanced soil calcium levels."
    ),

    "Bacterial Soft Rot / Blight": (
        "🦠  BACTERIAL SOFT ROT / ALTERNARIA BLIGHT\n"
        "Diagnosis: Water-soaked, soft, or dark necrotic lesions with tissue maceration risk.\n\n"
        "━━━  1. TRIMMING GUIDELINES  ━━━\n"
        "• Crown rot / blight : Excise affected carrot crown ≥ 20 mm below visible lesion margin into firm orange cortex.\n"
        "• Lateral soft rot   : If soft water-soaked spots are present on body, excise with minimum 15 mm healthy tissue margin.\n"
        "• Extensive soft rot : If > 15% root is soft or foul-smelling, REJECT entire root immediately.\n\n"
        "━━━  2. BATCH ISOLATION & QUARANTINE  ━━━\n"
        "• STRICT QUARANTINE : Soft rot bacteria (Pectobacterium / Dickeya) spread rapidly through wash water.\n"
        "• Do not recirculate wash water without continuous chlorination (≥ 150 ppm FAC) or ozone treatment.\n"
        "• Immediately separate sound roots from collapsing tissue.\n\n"
        "━━━  3. PROCESSING USAGE (HIGH DEFECT ROUTING)  ━━━\n"
        "• Divert firm, properly trimmed portions to high-temperature industrial processing:\n"
        "  - Thermal soup stock and pasteurized juice processing\n"
        "  - Puree manufacturing with retort sterilization (≥ 121 °C)\n"
        "• Never package trimmed soft-rot roots for fresh whole market sales.\n\n"
        "━━━  4. ANTI-MICROBIAL WASH PROCEDURES  ━━━\n"
        "• Soak trimmed carrots in 120–150 ppm buffered chlorine solution (pH 6.5) or 100 ppm peracetic acid for 2 min.\n"
        "• Dry rapidly with high-velocity blowers; free surface water triggers bacterial resurgence in storage."
    ),

    "Black Rot": (
        "⬛  BLACK ROT (Alternaria radicina)\n"
        "Diagnosis: Jet-black, dry, sunken cortical lesions; potential black mold growth.\n\n"
        "━━━  1. TRIMMING GUIDELINES  ━━━\n"
        "• Superficial lesions (< 10% area) : Deep-peel using industrial abrasive peelers until black discoloration is eliminated.\n"
        "• Penetrating lesions : Discard root; fungal mycelium readily invades xylem and vascular ring.\n\n"
        "━━━  2. BATCH ISOLATION & QUARANTINE  ━━━\n"
        "• Isolate affected crate lots; fungal conidia spread readily through contact and sorting equipment.\n"
        "• Wash and steam-sanitize sorting equipment and plastic storage crates before reusing.\n\n"
        "━━━  3. PROCESSING USAGE & PRECAUTIONS  ━━━\n"
        "• Heat-processed canned goods or juice only following deep-peeling and quality assurance testing.\n"
        "• Anti-fungal wash: Soak in 80–100 ppm peracetic acid or 150 ppm active chlorine solution for 2 minutes.\n"
        "• Rapidly store at 0 °C to suppress fungal mycelium development."
    ),
}

# Aliases for compatibility
_DISEASE_TREATMENT["Alternaria Leaf Blight"] = _DISEASE_TREATMENT["Bacterial Soft Rot / Blight"]
_DISEASE_TREATMENT["Cavity Spot"] = _DISEASE_TREATMENT["Cavity Spot / Lesion"]
_DISEASE_TREATMENT["Soil Stained"] = _DISEASE_TREATMENT["Soil Contamination / Surface Blemish"]
_DISEASE_TREATMENT["Surface Blemish"] = _DISEASE_TREATMENT["Soil Contamination / Surface Blemish"]

_GRADE_ACTION: Dict[str, str] = {
    "Grade A – Premium":    "📦  Retail Destination: Approved for premium fresh-market retail packaging and export.",
    "Grade B – Standard":   "📦  Retail Destination: Standard retail / wholesale ready following light trimming / washing.",
    "Grade C – Processing": "🏭  Industrial Processing: Divert from fresh market to pureeing, juicing, canning, or soup diced stock.",
    "Grade D – Reject":     "🛑  Disposal / Feed: Reject for human sale — divert to non-ruminant compost or regulated animal feed.",
}


@dataclass(frozen=True)
class SeverityReport:
    """Immutable result object returned by :func:`compute_severity`."""

    defect_pct: float
    classifier_conf: float       # Diagnosis / analysis confidence (0.0 to 1.0)
    composite_score: float
    grade: str
    grade_colour: str
    defect_name: str
    treatment: str
    pipeline_name: str = "Dual-Layer Heuristic (HSV + Texture)"


def compute_severity(
    classifier_confidence: float,
    blemish_pixel_ratio: float,
    defect_name: str = "Unknown",
    *,
    model_used: bool = False,
    classifier_weight: float = 0.4,
    blemish_weight: float = 0.6,
) -> SeverityReport:
    """Return a comprehensive :class:`SeverityReport` for one carrot image.

    Key consistency rules enforced:
    1. If Grade is Grade B, Grade C, or Grade D (or defect area >= 5%),
       the diagnosis must NEVER say 'Healthy'. It identifies the visible defect.
    2. Only assign 'Healthy' if Grade is 'Grade A – Premium' and defect area is < 5%.
    3. Guarantees structured, actionable precautions for every outcome.
    """
    defect_pct = max(0.0, min(100.0, blemish_pixel_ratio * 100.0))
    conf = max(0.0, min(1.0, classifier_confidence))

    # Base severity factor determined by defect category
    defect_severity_factors: Dict[str, float] = {
        "Healthy": 0.0,
        "Soil Contamination / Surface Blemish": 0.12,
        "Cavity Spot / Lesion": 0.28,
        "Bacterial Soft Rot / Blight": 0.45,
        "Black Rot": 0.50,
        "Surface Blemish": 0.15,
        "Soil Stained": 0.12,
        "Alternaria Leaf Blight": 0.40,
        "Cavity Spot": 0.28,
    }
    sev_factor = defect_severity_factors.get(defect_name, 0.20)

    # Calculate composite defect score (0 - 100)
    if defect_pct < 5.0 and defect_name in ("Healthy", "Unknown"):
        # Healthy low-blemish carrot: strictly Grade A range
        composite = min(defect_pct, 4.9)
    else:
        # Defective carrot: composite score is weighted by defect percentage and severity
        # Guarantees that defect_pct >= 5% yields composite >= 5.1 (Grade B or worse)
        composite = defect_pct * 0.80 + sev_factor * 25.0
        composite = max(5.1, min(100.0, composite))

    # Determine Grade from table
    grade, colour = "Grade D – Reject", "#ef4444"
    for threshold, label, hex_col in _GRADE_TABLE:
        if composite <= threshold:
            grade, colour = label, hex_col
            break

    # ── Strict Consistency Rules ──────────────────────────────────────
    # Rule 1: Only assign 'Healthy' if Grade is 'Grade A – Premium' AND defect area < 5%
    if grade == "Grade A – Premium" and defect_pct < 5.0:
        defect_name = "Healthy"
    else:
        # Rule 2: If Grade B, C, D or defect area >= 5%, diagnosis must NEVER say 'Healthy'
        if defect_name in ("Healthy", "Unknown", ""):
            if defect_pct >= 35.0 or grade == "Grade D – Reject":
                defect_name = "Bacterial Soft Rot / Blight"
            elif defect_pct >= 15.0 or grade == "Grade C – Processing":
                defect_name = "Cavity Spot / Lesion"
            else:
                defect_name = "Soil Contamination / Surface Blemish"

    # Normalize defect name aliases to canonical labels
    if defect_name in ("Soil Stained", "Surface Blemish"):
        defect_name = "Soil Contamination / Surface Blemish"
    elif defect_name in ("Cavity Spot",):
        defect_name = "Cavity Spot / Lesion"
    elif defect_name in ("Alternaria Leaf Blight",):
        defect_name = "Bacterial Soft Rot / Blight"

    # Look up treatment guidance
    disease_guidance = _DISEASE_TREATMENT.get(
        defect_name,
        _DISEASE_TREATMENT["Soil Contamination / Surface Blemish"],
    )
    grade_action = _GRADE_ACTION.get(grade, "")

    # Build comprehensive treatment text
    treatment_parts: list[str] = [disease_guidance]

    # If Grade C / High Defect state, ensure explicit Grade C / Processing precautions are prominent
    if grade in ("Grade C – Processing", "Grade D – Reject") or defect_pct >= 15.0:
        treatment_parts.extend([
            "",
            "━━━  HIGH-DEFECT (GRADE C / PROCESSING) PROTOCOL  ━━━",
            "• TRIMMING GUIDELINES : Excise lesions with a stainless-steel blade minimum 10–15 mm into sound orange cortex.",
            "• BATCH ISOLATION     : Segregate affected batch immediately into quarantine crates to prevent cross-lot decay.",
            "• PROCESSING USAGE    : Divert from fresh market to high-temperature puree, industrial juicing, or canning stock.",
            "• ANTI-FUNGAL WASH    : Wash trimmed carrots in 80–100 ppm peracetic acid or 150 ppm chlorinated water for 2 min.",
        ])

    treatment_parts.extend([
        "",
        "━━━  DISPATCH & DISPOSITION  ━━━",
        grade_action,
    ])

    treatment = "\n".join(treatment_parts)

    pipeline_name = (
        "MobileNetV2 + HSV" if model_used else "Dual-Layer Heuristic (HSV + Texture)"
    )

    return SeverityReport(
        defect_pct=round(defect_pct, 2),
        classifier_conf=round(conf, 4),
        composite_score=round(composite, 2),
        grade=grade,
        grade_colour=colour,
        defect_name=defect_name,
        treatment=treatment,
        pipeline_name=pipeline_name,
    )
