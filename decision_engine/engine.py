"""
AI-Powered Restaurant Revenue Concierge Agent (rule engine + optional local Llama via Ollama).

HOW IT WORKS
------------
1. The RULE ENGINE decides. It reads the slot (occupancy, peak/off-peak), recent cancellations,
   how soon service starts and how fast the slot is filling, turns each into points, adds them up,
   and maps the total to a decision:

       total 0-2  -> NOTIFY            (soft priority notice, $0 cost)
       total 3-4  -> LOW_INCENTIVE     (culinary perk, no discount)
       total 5+   -> HIGH_INCENTIVE    (chef's tasting, or a small capped discount as a last resort)

   The same facts always give the same answer, so every decision can be explained and audited.

2. SAFETY RULES can only make a decision weaker, never stronger:
   - No discount-level incentive in a peak slot unless occupancy is below the low-occupancy threshold.
   - A completely full slot sends nothing.
   - Discounts are always capped (settings value, never above 12%, strictly below 15%).

3. OLLAMA (optional) only writes the email subject and body. It cannot change the decision or the
   offer. If Ollama is offline, or its text breaks a brand rule, the built-in templates are used.

KEPT COMPATIBLE WITH THE REST OF THE PROJECT
--------------------------------------------
- Same public functions: is_ollama_available(), run_concierge_agent(), _generate_fallback_decision().
- Same database calls and the same return keys as before. New keys are only added
  (score, score_breakdown, rules_applied, send, minutes_until_service).
- Decision values are unchanged: NOTIFY / LOW_INCENTIVE / HIGH_INCENTIVE.
"""

import json
import os
import re
from datetime import date, datetime, time as dtime
from typing import Any, Dict, List, Optional, Tuple

import requests
import database

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

TOTAL_SEATS = 20

# Score thresholds (the numbers that decide NOTIFY / LOW / HIGH)
LOW_SCORE = 3
HIGH_SCORE = 5

# Scoring inputs
URGENT_MINUTES = 120          # service starts within 2 hours -> urgency point (if not already well booked)
URGENT_BELOW_OCCUPANCY = 60.0
FILLING_BOOKINGS_PER_HOUR = 2  # 2+ new bookings in the last hour -> slot is filling by itself

# Offer economics (estimated food and beverage cost in USD)
COSTS = {
    "NONE": 0.0,
    "FREE_BEVERAGE": 7.5,
    "FREE_DESSERT": 8.0,
    "COMPLIMENTARY_APPETIZER": 10.0,
    "CHEF_SPECIAL_TASTING": 14.0,
    "DISCOUNT": 0.0,
}
AVG_CHECK_USD = 80.0           # used to estimate what a percentage discount costs

HIGH_TIERS = ("platinum", "gold")
TIER_WEIGHT = {"platinum": 4, "gold": 3, "silver": 2}

DECISION_LABELS = {
    "NOTIFY": "NOTIFY (soft priority notice)",
    "LOW_INCENTIVE": "LOW_INCENTIVE",
    "HIGH_INCENTIVE": "HIGH_INCENTIVE",
}


def is_ollama_available(host: str = OLLAMA_HOST) -> Tuple[bool, List[str]]:
    """Check if the local Ollama server is running and return available models."""
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return True, models
        return False, []
    except Exception:
        return False, []


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _setting(settings: Dict[str, Any], key: str, default: float) -> float:
    """Reads settings in the shape the database returns: {"key": {"value": ...}}."""
    try:
        return float((settings.get(key) or {}).get("value", default))
    except (TypeError, ValueError):
        return float(default)


def _max_discount(settings: Dict[str, Any]) -> float:
    """Brand guardrail: the cap from settings, never above 12%, always strictly below 15%."""
    cap = min(_setting(settings, "max_discount_percent", 10.0), 12.0)
    if cap >= 15.0:
        cap = 10.0
    return max(cap, 0.0)


def _fmt_pct(value: float) -> str:
    return "%g" % value


def _tier_of(customer: Optional[Dict[str, Any]]) -> str:
    return str((customer or {}).get("loyalty_tier") or "").strip().lower()


def _parse_time(value: Any) -> Optional[dtime]:
    if isinstance(value, dtime):
        return value
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
            try:
                return datetime.strptime(text, fmt).time()
            except ValueError:
                continue
    return None


def _minutes_until_service(slot: Dict[str, Any], now: Optional[datetime] = None) -> Optional[int]:
    """Minutes from now until the slot starts (0 if already started). None if it cannot be worked out."""
    try:
        now = now or datetime.now()
        slot_day = slot.get("slot_date")
        if isinstance(slot_day, datetime):
            slot_day = slot_day.date()
        elif isinstance(slot_day, str):
            slot_day = datetime.strptime(slot_day.strip()[:10], "%Y-%m-%d").date()
        if not isinstance(slot_day, date):
            return None
        start = _parse_time(slot.get("start_time"))
        if start is None:
            return None
        delta = datetime.combine(slot_day, start) - now
        return max(0, int(delta.total_seconds() // 60))
    except Exception:
        return None


def _recent_bookings(slot_id: int, db_path: Optional[str]) -> int:
    """
    New bookings in the last hour. Optional: if your database module has
    get_recent_booking_count(slot_id, hours=1, db_path=None) it is used, otherwise this is 0.
    """
    fn = getattr(database, "get_recent_booking_count", None)
    if not callable(fn):
        return 0
    try:
        return int(fn(slot_id, hours=1, db_path=db_path))
    except Exception:
        return 0


# --------------------------------------------------------------------------
# THE DECISION ENGINE (pure rules, no AI)
# --------------------------------------------------------------------------
def score_slot(
    slot: Dict[str, Any],
    trend: Dict[str, Any],
    settings: Dict[str, Any],
    minutes_until: Optional[int] = None,
    recent_bookings: int = 0,
) -> Dict[str, Any]:
    """Turns the facts into points and a decision. Does not touch the database."""
    occ = float(slot.get("occupancy_percent") or 0.0)
    booked = int(slot.get("booked_seats") or 0)
    available = slot.get("available_seats")
    available = TOTAL_SEATS - booked if available is None else int(available)
    canc = int(trend.get("cancellation_count") or 0)
    is_peak = int(slot.get("is_peak") or 0) == 1
    low_thresh = _setting(settings, "low_occupancy_threshold", 50.0)

    # ---- points ----
    pts_occ = 0 if occ >= 85 else 1 if occ >= 60 else 2 if occ >= 40 else 3
    pts_canc = 0 if canc == 0 else 1 if canc <= 2 else 2
    pts_time = 0 if is_peak else 2
    pts_pace = -1 if recent_bookings >= FILLING_BOOKINGS_PER_HOUR else 0
    urgent = (minutes_until is not None and minutes_until <= URGENT_MINUTES and occ < URGENT_BELOW_OCCUPANCY)
    pts_urg = 1 if urgent else 0

    breakdown = [
        {"label": "Occupancy", "detail": "%.0f%% booked" % occ, "points": pts_occ},
        {"label": "Cancellations", "detail": "%d recent" % canc, "points": pts_canc},
        {"label": "Time slot", "detail": "peak" if is_peak else "off-peak", "points": pts_time},
        {"label": "Filling pace", "detail": "%d new in last hour" % recent_bookings, "points": pts_pace},
        {"label": "Urgency",
         "detail": "service in %d min" % minutes_until if minutes_until is not None else "start time unknown",
         "points": pts_urg},
    ]
    total = max(0, pts_occ + pts_canc + pts_time + pts_pace + pts_urg)

    base = "HIGH_INCENTIVE" if total >= HIGH_SCORE else "LOW_INCENTIVE" if total >= LOW_SCORE else "NOTIFY"
    decision = base
    rules: List[str] = []

    # ---- safety rules (can only weaken the decision) ----
    peak_capped = False
    if decision == "HIGH_INCENTIVE" and is_peak and occ >= low_thresh:
        decision = "LOW_INCENTIVE"
        peak_capped = True
        rules.append(
            "Peak-slot rule: no high incentive in a peak window unless occupancy is below %.0f%%, so it was lowered to a gentle perk."
            % low_thresh)

    full = available <= 0
    if full:
        decision = "NOTIFY"
        rules.append("Slot is completely full, so there is nothing to fill and no message is sent.")

    return {
        "total": total, "breakdown": breakdown, "base_decision": base, "decision": decision,
        "rules": rules, "peak_capped": peak_capped, "full": full,
        "occupancy": occ, "booked": booked, "available": available, "cancellations": canc,
        "cancelled_seats": int(trend.get("cancelled_seats") or 0), "is_peak": is_peak,
        "minutes_until": minutes_until, "recent_bookings": recent_bookings,
    }


def _pick_customer(eligible: List[Dict[str, Any]], decision: str) -> Optional[Dict[str, Any]]:
    """
    Best-ranked eligible member. For a high incentive the best tier goes first (stable sort,
    so ties keep the order the database returned). Otherwise the database order is kept.
    """
    if not eligible:
        return None
    if decision != "HIGH_INCENTIVE":
        return eligible[0]
    ranked = sorted(
        eligible,
        key=lambda c: (TIER_WEIGHT.get(_tier_of(c), 1), int(c.get("total_visits") or 0)),
        reverse=True)
    return ranked[0]


def _choose_offer(scored: Dict[str, Any], customer: Optional[Dict[str, Any]], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Picks the offer. Culinary perks first, small discounts only as a last resort."""
    decision = scored["decision"]
    cap = _max_discount(settings)
    crit_thresh = _setting(settings, "critical_occupancy_threshold", 25.0)
    high_tier = _tier_of(customer) in HIGH_TIERS
    discount = 0.0

    if scored["full"] or decision == "NOTIFY":
        offer_type = "NONE"
    elif decision == "LOW_INCENTIVE":
        if scored["cancellations"] >= 1 or high_tier:
            offer_type = "COMPLIMENTARY_APPETIZER"
        else:
            offer_type = "FREE_BEVERAGE"
    else:  # HIGH_INCENTIVE
        if high_tier:
            offer_type = "CHEF_SPECIAL_TASTING"
        elif scored["cancellations"] >= 3:
            offer_type = "COMPLIMENTARY_APPETIZER"
            discount = min(5.0, cap)
        elif scored["occupancy"] <= crit_thresh and cap > 0:
            offer_type = "DISCOUNT"
            discount = min(10.0, cap)
        else:
            offer_type = "CHEF_SPECIAL_TASTING"

    cost = COSTS[offer_type] + AVG_CHECK_USD * discount / 100.0
    return {"offer_type": offer_type, "discount_percent": round(discount, 2), "estimated_cost": round(cost, 2)}


def _build_reasoning(scored: Dict[str, Any], plan: Dict[str, Any], customer: Optional[Dict[str, Any]]) -> str:
    peak_txt = "peak" if scored["is_peak"] else "off-peak"
    who = "a loyalty member"
    if customer:
        who = "%s member %s" % (customer.get("loyalty_tier", "loyalty"), customer.get("name", "guest"))

    s1 = "%.1f%% occupancy (%d/%d seats) in %s %s window with %d recent cancellation%s" % (
        scored["occupancy"], scored["booked"], TOTAL_SEATS, "a" if scored["is_peak"] else "an", peak_txt,
        scored["cancellations"], "" if scored["cancellations"] == 1 else "s")
    if scored["minutes_until"] is not None:
        s1 += ", service starts in %d min" % scored["minutes_until"]
    if scored["recent_bookings"]:
        s1 += ", %d new booking%s in the last hour" % (scored["recent_bookings"], "" if scored["recent_bookings"] == 1 else "s")
    s1 += "."

    s2 = "Score %d maps to %s." % (scored["total"], DECISION_LABELS[scored["decision"]])

    kind = plan["offer_type"]
    if scored["full"]:
        s3 = "Every seat is taken, so nothing is sent."
    elif kind == "NONE":
        s3 = "Demand is healthy, so a zero-cost priority notice for %s is enough and protects price perception." % who
    elif kind == "FREE_BEVERAGE":
        s3 = "A complimentary sommelier pairing for %s nudges a booking without touching menu prices." % who
    elif kind == "COMPLIMENTARY_APPETIZER":
        extra = " plus a modest %s%% courtesy" % _fmt_pct(plan["discount_percent"]) if plan["discount_percent"] > 0 else ""
        s3 = "A signature starter%s for %s fills seats without a heavy price cut." % (extra, who)
    elif kind == "CHEF_SPECIAL_TASTING":
        s3 = "An exclusive chef's tasting for %s backfills seats while keeping the luxury positioning." % who
    elif kind == "DISCOUNT":
        s3 = "Occupancy is critical, so a small %s%% courtesy (inside the brand cap) is used as a last resort for %s." % (
            _fmt_pct(plan["discount_percent"]), who)
    else:
        s3 = "Offer selected for %s." % who

    text = " ".join([s1, s2, s3])
    if scored["peak_capped"]:
        text += " The peak-slot safety rule lowered the incentive."
    return text


def _compose_copy(offer_type: str, discount: float, customer: Optional[Dict[str, Any]], slot: Dict[str, Any]) -> Tuple[str, str]:
    """Built-in email templates (used whenever Ollama is offline or its text is rejected)."""
    name = (customer or {}).get("name", "Valued Guest")
    tier = (customer or {}).get("loyalty_tier", "valued")
    when = str(slot.get("start_time", "this evening"))
    sign = "The Concierge Team"
    pct = _fmt_pct(discount)

    if offer_type == "CHEF_SPECIAL_TASTING":
        subject = "Private Chef's Tasting Invitation for %s" % name
        body = ("Dear %s,\n\nOur Executive Chef has prepared an exclusive seasonal tasting for service at %s. "
                "In honor of your %s patronage, we have reserved an intimate table for you, accompanied by a "
                "complimentary chef's amuse-bouche and tasting course.\n\nWarmest regards,\n%s") % (name, when, tier, sign)
    elif offer_type == "COMPLIMENTARY_APPETIZER":
        subject = "A Signature Starter, Courtesy of the House"
        extra = ", alongside a modest %s%% courtesy" % pct if discount > 0 else ""
        body = ("Dear %s,\n\nA table has opened for service at %s. We invite you to join us and enjoy our signature "
                "Truffle Arancini on the house%s.\n\nBest regards,\n%s") % (name, when, extra, sign)
    elif offer_type == "FREE_BEVERAGE":
        subject = "Complimentary Sommelier Selection with Your Reservation"
        body = ("Dear %s,\n\nJoin us for a relaxed dining experience at %s. As a valued %s member, enjoy a complimentary "
                "glass from our sommelier's reserve when you reserve your table.\n\nSincerely,\n%s") % (name, when, tier, sign)
    elif offer_type == "FREE_DESSERT":
        subject = "A Chef's Dessert, Reserved for You"
        body = ("Dear %s,\n\nReserve a table for service at %s and our pastry chef will plate a complimentary dessert "
                "for you.\n\nWarm regards,\n%s") % (name, when, sign)
    elif offer_type == "DISCOUNT":
        subject = "A Courtesy for Your Next Visit"
        body = ("Dear %s,\n\nWe have a table available for service at %s and would be delighted to welcome you with a "
                "%s%% courtesy on your visit.\n\nBest regards,\n%s") % (name, when, pct, sign)
    else:
        subject = "Table Availability for %s" % when
        body = ("Dear %s,\n\nWe are holding limited seating for service at %s. As a %s member, you have first priority "
                "to book before tables are committed.\n\nWarm regards,\n%s") % (name, when, tier, sign)
    return subject, body


def _generate_fallback_decision(
    slot: Dict[str, Any],
    trend: Dict[str, Any],
    eligible_customers: List[Dict[str, Any]],
    settings: Dict[str, Any],
    minutes_until: Optional[int] = None,
    recent_bookings: int = 0,
) -> Dict[str, Any]:
    """
    The deterministic decision (kept under its old name so existing imports keep working).
    Always returns a complete decision with template email text.
    """
    scored = score_slot(slot, trend, settings, minutes_until=minutes_until, recent_bookings=recent_bookings)
    customer = _pick_customer(eligible_customers, scored["decision"])
    plan = _choose_offer(scored, customer, settings)
    subject, body = _compose_copy(plan["offer_type"], plan["discount_percent"], customer, slot)

    return {
        "decision": scored["decision"],
        "reasoning": _build_reasoning(scored, plan, customer),
        "target_customer_id": customer["customer_id"] if customer else None,
        "offer_type": plan["offer_type"],
        "discount_percent": plan["discount_percent"],
        "estimated_cost": plan["estimated_cost"],
        "email_subject": subject,
        "email_body": body,
        "engine_used": "Rule engine (template email)",
        "score": scored["total"],
        "score_breakdown": scored["breakdown"],
        "rules_applied": scored["rules"],
        "send": (not scored["full"]) and customer is not None,
        "_customer": customer,
    }


# --------------------------------------------------------------------------
# Optional: let local Llama write the email (it can never change the decision)
# --------------------------------------------------------------------------
EMAIL_SYSTEM_PROMPT = """You write short, elegant invitation emails for a prestigious, boutique 20-seat fine dining restaurant.

The decision and the offer are ALREADY FINAL. You only write the wording.
Rules:
- Do not change, add or remove any perk. Mention only the offer you are given.
- Never mention a discount or any percentage unless discount_percent is greater than 0, and then use exactly that number.
- Never use pushy sales language. Keep the tone warm, refined and personal.
- Body: 3 to 5 short sentences, ending with a sign-off from "The Concierge Team".

Respond ONLY with a JSON object: {"email_subject": "<subject line>", "email_body": "<email text>"}"""


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"^```(?:json)?\s*", "", (raw or "").strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else None
            except Exception:
                return None
    return None


def _copy_is_brand_safe(subject: str, body: str, discount: float) -> bool:
    """Rejects AI text that mentions a percentage other than the approved discount."""
    if len(subject) < 3 or len(subject) > 120 or len(body) < 40:
        return False
    allowed = {float(discount)} if discount > 0 else set()
    for number in re.findall(r"(\d+(?:\.\d+)?)\s*%", subject + " " + body):
        if float(number) not in allowed:
            return False
    return True


def _llm_email(
    slot: Dict[str, Any],
    plan: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
    decision: str,
    ollama_url: str,
    model: str,
) -> Optional[Tuple[str, str]]:
    facts = {
        "decision": decision,
        "offer_type": plan["offer_type"],
        "discount_percent": plan["discount_percent"],
        "service_time": str(slot.get("start_time", "")),
        "guest_name": (customer or {}).get("name", "Valued Guest"),
        "loyalty_tier": (customer or {}).get("loyalty_tier", "valued"),
        "preferred_time": (customer or {}).get("preferred_time"),
    }
    prompt = "Write the email for this final offer:\n%s\n\nRespond with the JSON object only." % json.dumps(facts, indent=2)
    try:
        response = requests.post(
            "%s/api/generate" % ollama_url.rstrip("/"),
            json={"model": model, "system": EMAIL_SYSTEM_PROMPT, "prompt": prompt, "stream": False, "format": "json"},
            timeout=30.0)
        if response.status_code != 200:
            return None
        data = _extract_json(response.json().get("response", ""))
        if not data:
            return None
        subject = str(data.get("email_subject", "")).strip()
        body = str(data.get("email_body", "")).strip()
        if _copy_is_brand_safe(subject, body, plan["discount_percent"]):
            return subject, body
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------
# Main entry point (same name and arguments as before, plus two optional ones)
# --------------------------------------------------------------------------
def run_concierge_agent(
    slot_id: int,
    lookback_hours: int = 2,
    preferred_model: str = OLLAMA_MODEL,
    ollama_url: str = OLLAMA_HOST,
    db_path: Optional[str] = None,
    now: Optional[datetime] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Execute the Concierge Agent on a specific 20-seat time slot.
    Gathers slot context, applies the scoring rules, optionally lets local Llama write the email,
    records the audit log, and stages the offer for manager approval.

    now      - pass your own clock here if your demo uses a fake time (default: the real time).
    use_llm  - set False to skip Ollama completely and always use the built-in email templates.
    """
    # 1. Fetch real-time database context
    slot = database.get_slot_details(slot_id, db_path=db_path)
    if not slot:
        raise ValueError(f"Slot ID {slot_id} not found.")

    trend = database.get_cancellation_trend(slot_id, hours=lookback_hours, db_path=db_path)
    eligible_customers = database.get_eligible_customers(slot_id, db_path=db_path)
    settings = database.get_all_settings(db_path=db_path)

    warnings: List[str] = []
    if not eligible_customers:
        all_opted = database.get_customers(marketing_only=True, db_path=db_path)
        eligible_customers = list(all_opted[:5]) if all_opted else []
    if not eligible_customers:
        warnings.append("No marketing-consented customers available, so no offer was created.")

    minutes_until = _minutes_until_service(slot, now)
    recent = _recent_bookings(slot_id, db_path)

    # 2. The rule engine decides (always)
    result = _generate_fallback_decision(
        slot, trend, eligible_customers, settings, minutes_until=minutes_until, recent_bookings=recent)
    engine_used = result["engine_used"]

    # 3. Optional: Ollama writes the email only. The decision and offer stay exactly as scored.
    if use_llm and result["send"] and result["_customer"] is not None:
        plan = {"offer_type": result["offer_type"], "discount_percent": result["discount_percent"]}
        written = _llm_email(slot, plan, result["_customer"], result["decision"], ollama_url, preferred_model)
        if written:
            result["email_subject"], result["email_body"] = written
            engine_used = "Rule engine + Local Ollama email (%s)" % preferred_model
        else:
            engine_used = "Rule engine (Ollama offline or text rejected: template email)"

    # 4. Final brand guardrail on the discount (belt and braces)
    max_discount = _max_discount(settings)
    result["discount_percent"] = min(float(result["discount_percent"]), max_discount)
    if result["discount_percent"] >= 15.0:
        result["discount_percent"] = 10.0

    # 5. Resolve the target customer (must have marketing consent)
    target_cust = None
    target_cid = result["target_customer_id"]
    if result["send"] and target_cid is not None:
        target_cust = database.get_customer_by_id(int(target_cid), db_path=db_path)
        if not target_cust or target_cust.get("marketing_consent") != 1:
            target_cust = eligible_customers[0]
            target_cid = target_cust["customer_id"]
    elif result["_customer"] is not None:
        target_cust = result["_customer"]

    # 6. Save the offer (pending manager review), unless there is nothing to send
    created_offer = None
    if result["send"] and target_cid is not None:
        try:
            created_offer = database.create_offer(
                customer_id=target_cid,
                slot_id=slot_id,
                decision=result["decision"],
                offer_type=result["offer_type"],
                discount_percent=result["discount_percent"],
                estimated_cost=float(result["estimated_cost"]),
                email_subject=result["email_subject"],
                email_body=result["email_body"],
                status="PENDING",
                db_path=db_path,
            )
            warnings.extend((created_offer or {}).get("warnings", []))
        except Exception as e:
            warnings.append(f"Offer creation notice: {str(e)}")
    elif not result["send"] and result["rules_applied"]:
        warnings.extend(result["rules_applied"][-1:])

    # 7. Log the decision to the audit trail
    log_id = database.log_agent_decision(
        slot_id=slot_id,
        occupancy_percent=slot["occupancy_percent"],
        cancellation_count=trend["cancellation_count"],
        minutes_until_service=minutes_until if minutes_until is not None else 90,
        decision=result["decision"],
        reasoning=result.get("reasoning", "Agent decision generated."),
        incentive_type=result.get("offer_type"),
        estimated_incentive_cost=float(result.get("estimated_cost", 0.0)),
        expected_occupancy_percent=slot.get("expected_occupancy_percent"),
        warnings="; ".join(warnings) if warnings else None,
        db_path=db_path,
    )

    return {
        "slot": slot,
        "trend": trend,
        "decision": result["decision"],
        "reasoning": result["reasoning"],
        "offer_type": result["offer_type"],
        "discount_percent": result["discount_percent"],
        "estimated_cost": result.get("estimated_cost", 0.0),
        "target_customer": target_cust,
        "email_subject": result["email_subject"],
        "email_body": result["email_body"],
        "engine_used": engine_used,
        "offer_id": created_offer.get("offer_id") if created_offer else None,
        "log_id": log_id,
        "warnings": warnings,
        # new, additive fields
        "score": result["score"],
        "score_breakdown": result["score_breakdown"],
        "rules_applied": result["rules_applied"],
        "send": result["send"],
        "minutes_until_service": minutes_until,
    }