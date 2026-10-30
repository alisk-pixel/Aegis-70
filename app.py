# MIT License
#
# Copyright (c) 2025 Masterpiece Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
app.py - Masterpiece: Dual-Mode Execution Runtime & Evaluator Interface

Execution Modes:
  1. CLI Terminal Pipeline:
       python app.py                       # Execute bedroom desk crisis scenario
       python app.py --scenario all        # Execute all verification scenarios
       python app.py --offline             # Execute strictly with on-device heuristics
       python app.py --ledger | --export | --purge
  2. Streamlit Web Dashboard:
       streamlit run app.py                # Launch interactive hackathon evaluator UI
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import (
    STATES,
    MasterpieceConfig,
    NebiusConfig,
    load_config,
    resolve_system_credentials,
)
from engine import (
    BehaviorEngine,
    FusedObservation,
    PrivacyViolation,
    RiskAssessment,
    SensorEvent,
    SensorFusion,
    StateStore,
    Tier,
    audit_payload,
)

WIDTH = 78
USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
TIER_COLORS = {"GREEN": "32", "YELLOW": "33", "ORANGE": "38;5;208", "RED": "31"}


def paint(text: str, tier_name: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{TIER_COLORS.get(tier_name, '0')}m{text}\033[0m"


# ---------------------------------------------------------------------------
# Permitted Skill Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    steps: Tuple[Tuple[str, str], ...]


SKILLS: Dict[str, Skill] = {
    "desk_shutdown_ritual": Skill(
        "desk_shutdown_ritual",
        "Close current session loops and execute workstation power-down.",
        (
            ("notify", "Write tomorrow's initial objective in one sentence, then halt"),
            ("display", "Dim workstation to 10% and arm 60-second lock barrier"),
            ("lights", "Hallway path illuminated to 20% warm amber"),
        ),
    ),
    "leave_the_room_reset": Skill(
        "leave_the_room_reset",
        "Physical room exit protocol to break sedentary cognitive fixation.",
        (
            ("chime", "Soft two-tone auditory cue"),
            ("lights", "Hallway and kitchen lighting transition to warm 40%"),
            ("timer", "5-minute reset timer armed; re-verify presence on return"),
        ),
    ),
    "breathing_reset": Skill(
        "breathing_reset",
        "Two minutes of paced somatic regulation to break screen fixation.",
        (
            ("display", "Overlay box-breathing cadence guide (4-4-4-4 rhythm)"),
            ("chime", "Soft acoustic inhale/exhale timing cues"),
            ("timer", "2-minute somatic regulation sequence"),
        ),
    ),
    "reach_out_text": Skill(
        "reach_out_text",
        "Draft an accountability message to a trusted contact; human executes send.",
        (
            ("notify", "Draft armed: 'Hey - hitting late-night friction, open to a quick check-in?'"),
            ("timer", "10-minute accountability follow-up window"),
        ),
    ),
    "sleep_handoff": Skill(
        "sleep_handoff",
        "Execute sleep transition sequence and remove mobile distraction vectors.",
        (
            ("lights", "Bedroom lighting dimmed to 15% warm spectrum"),
            ("display", "Night-shift spectrum active; display locks in 10 minutes"),
            ("notify", "Relocate phone to physical charger across the bedroom"),
        ),
    ),
}

ACTUATOR_TEMPLATES = {
    "lights": "smart-lights -> {}",
    "display": "workstation-display -> {}",
    "chime": "acoustic-speaker -> {}",
    "notify": "local-notification -> {}",
    "timer": "edge-timer -> {}",
}


@dataclass
class SkillExecution:
    skill: str
    timestamp: datetime
    actions: List[str]
    simulated: bool = True


class LocalActuators:
    """Local edge actuation interface. Simulated for evaluation; maps to Home Assistant/MQTT."""

    def execute(self, skill_name: str, now: datetime) -> SkillExecution:
        skill = SKILLS[skill_name]
        actions = [ACTUATOR_TEMPLATES[actuator].format(argument) for actuator, argument in skill.steps]
        return SkillExecution(skill=skill_name, timestamp=now, actions=actions)


# ---------------------------------------------------------------------------
# Strict Contract Plan Validation
# ---------------------------------------------------------------------------
def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def validate_plan(raw: Dict[str, Any], allowed_skills: Sequence[str]) -> Dict[str, Any]:
    required = ("acknowledgement", "directive", "skill", "steps", "follow_up_minutes", "tone", "reflection_question")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Plan validation failed: missing keys: {missing}")
    skill = str(raw["skill"]).strip()
    if skill not in allowed_skills:
        raise ValueError(f"Plan violation: skill {skill!r} not in permitted registry")
    steps_raw = raw["steps"]
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("Plan validation failed: steps must be a non-empty list")
    steps = [_clip(step, 160) for step in steps_raw[:3] if str(step).strip()]
    if not steps:
        raise ValueError("Plan validation failed: steps contained no usable text")
    try:
        follow_up = int(raw["follow_up_minutes"])
    except (TypeError, ValueError) as exc:
        raise ValueError("follow_up_minutes must be an integer") from exc
    follow_up = max(5, min(30, follow_up))
    tone = str(raw["tone"]).strip().lower()
    if tone not in {"gentle", "steady", "firm"}:
        tone = "steady"
    acknowledgement = _clip(raw["acknowledgement"], 400)
    directive = _clip(raw["directive"], 240)
    reflection = _clip(raw["reflection_question"], 240)
    if not acknowledgement or not directive or not reflection:
        raise ValueError("acknowledgement, directive, and reflection_question must be non-empty strings")
    return {
        "acknowledgement": acknowledgement,
        "directive": directive,
        "skill": skill,
        "steps": steps,
        "follow_up_minutes": follow_up,
        "tone": tone,
        "reflection_question": reflection,
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Inference engine did not return a valid JSON object")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON payload to be a key-value dictionary")
    return parsed


# ---------------------------------------------------------------------------
# Universal Inference Client (Nebius Token Factory & Google AI Studio)
# ---------------------------------------------------------------------------
class NebiusError(RuntimeError):
    pass


SYSTEM_PROMPT_TEMPLATE = (
    "You are {name}, a private, personal-AI behavioral accountability supervisor. "
    "You receive an ABSTRACT behavioral telemetry report containing zero raw video, audio, or keystrokes. "
    "Your objective is to generate exactly ONE high-leverage intervention.\n\n"
    "CONTRACT RULES:\n"
    "1. Respond ONLY with a valid, single JSON object without commentary or markdown code fences.\n"
    "2. Schema:\n"
    "   {{\n"
    "     \"acknowledgement\": string (max 2 sentences, recognizing the behavioral loop without shaming),\n"
    "     \"directive\": string (one concrete physical action doable in under 5 minutes, max 25 words),\n"
    "     \"skill\": string (MUST be strictly chosen from allowed_skills),\n"
    "     \"steps\": array of 1-3 short strings,\n"
    "     \"follow_up_minutes\": integer between 5 and 30,\n"
    "     \"tone\": \"gentle\" | \"steady\" | \"firm\",\n"
    "     \"reflection_question\": string\n"
    "   }}\n\n"
    "VOICE & PERSONA GUIDELINES:\n"
    "- Calm, dry, respectful JARVIS-style presence. Address the user as \"{address}\".\n"
    "- Zero moralizing, zero shame. Never claim to see, hear, or watch the user.\n"
    "- High-risk states require firm, concise clarity. Lower-risk states warrant gentle nudges.\n"
    "- If relapse_streak_nights >= 3 or previous_intervention_outcome is RELAPSED, "
    "the reflection_question must encourage connecting with trusted human accountability."
)


def build_messages(payload: Dict[str, Any], cfg: MasterpieceConfig) -> List[Dict[str, str]]:
    system = SYSTEM_PROMPT_TEMPLATE.format(name=cfg.persona.assistant_name, address=cfg.persona.address_as)
    user = (
        "Behavioral Telemetry Report (JSON):\n"
        + json.dumps(payload, indent=2, sort_keys=True)
        + "\n\nGenerate the single structured intervention JSON object now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class NebiusClient:
    """Retrying wrapper over OpenAI-compatible endpoints (Nebius or Google AI Studio)."""

    def __init__(self, cfg: NebiusConfig) -> None:
        if not cfg.configured:
            raise NebiusError("No API key configured for cloud inference")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise NebiusError("The 'openai' client package is missing (run: pip install openai)") from exc

        self.cfg = cfg
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_seconds,
            max_retries=0,
        )

    def complete_plan(
        self, messages: List[Dict[str, str]], allowed_skills: Sequence[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import openai

        last_error: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_tokens,
                )
                if not response.choices:
                    raise ValueError(f"{self.cfg.provider_name} returned zero completion choices")
                text = response.choices[0].message.content or ""
                fields = validate_plan(extract_json_object(text), allowed_skills)
                usage_obj = getattr(response, "usage", None)
                usage = (
                    {
                        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                    }
                    if usage_obj
                    else {}
                )
                meta = {
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "usage": usage,
                    "model": self.cfg.model,
                    "provider": self.cfg.provider_name,
                    "attempts": attempt,
                }
                return fields, meta
            except (
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
                openai.BadRequestError,
            ) as exc:
                raise NebiusError(f"{self.cfg.provider_name} rejected request ({exc.__class__.__name__}): {exc}") from exc
            except (
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.backoff_base_seconds * (2 ** (attempt - 1)))

        raise NebiusError(f"{self.cfg.provider_name} call failed after {self.cfg.max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Edge Heuristic Fallback
# ---------------------------------------------------------------------------
def local_fallback_plan(a: RiskAssessment, cfg: MasterpieceConfig) -> Dict[str, Any]:
    address = cfg.persona.address_as
    if a.room == "BEDROOM_BED":
        skill = "sleep_handoff"
        directive = "Move phone to the physical charger across the room and close the session."
        steps = ["Lock screen immediately", "Relocate phone to distance stand", "Extinguish room lighting"]
    elif a.tier == Tier.RED:
        skill = "leave_the_room_reset"
        directive = "Stand up now, leave the bedroom workspace, and drink a glass of water."
        steps = ["Push back chair and stand", "Walk to kitchen area", "Drink 250ml water before re-evaluating"]
    else:
        skill = "desk_shutdown_ritual"
        directive = "Write tomorrow's primary objective in one sentence and close laptop display."
        steps = ["Write single task line", "Close workstation lid", "Set lighting to warm sleep path"]

    return {
        "acknowledgement": (
            f"{address}, the behavioral pattern indicates the last {a.idle_minutes:.0f} minutes "
            "have transitioned into sedentary drift. That is the biological loop, not your capability."
        ),
        "directive": directive,
        "skill": skill,
        "steps": steps,
        "follow_up_minutes": 10,
        "tone": "firm" if a.tier == Tier.RED else "steady",
        "reflection_question": "What is one decisive physical action you would respect having completed tonight?",
    }


# ---------------------------------------------------------------------------
# CLI JARVIS Protocol Terminal Renderer
# ---------------------------------------------------------------------------
def _wrap(label: str, text: str) -> str:
    prefix = f" [{label}]".ljust(15)
    lines = textwrap.wrap(text, width=WIDTH - len(prefix)) or [""]
    return "\n".join([prefix + lines[0]] + [" " * len(prefix) + line for line in lines[1:]])


def render_protocol(
    a: RiskAssessment,
    fields: Dict[str, Any],
    execution: SkillExecution,
    source: str,
    model: str,
    latency_ms: int,
    usage: Dict[str, int],
    cfg: MasterpieceConfig,
) -> str:
    title = f" {cfg.persona.assistant_name.upper()} INTERVENTION PROTOCOL"
    status = f"{a.timestamp.strftime('%H:%M')} | {a.tier.name} | {a.score:5.1f}/100 "
    pad = max(1, WIDTH - 2 - len(title) - len(status))
    horizon = cfg.markov.horizon_steps * cfg.markov.step_minutes
    lines = [
        "╔" + "═" * (WIDTH - 2) + "╗",
        "║" + title + " " * pad + paint(status, a.tier.name) + "║",
        "╚" + "═" * (WIDTH - 2) + "╝",
        _wrap(
            "PRESENCE",
            f"{a.room} · {a.state} · idle {a.idle_minutes:.0f} min · isolated "
            f"{a.isolation_minutes:.0f} min · sensor confidence {a.confidence:.2f}",
        ),
        _wrap(
            "ASSESSMENT",
            f"Risk {a.score:.1f}/100 ({a.tier.name}) · P(trap in {horizon}m) = "
            f"{a.trap_probability:.2f} · relapse streak {a.relapse_streak} night(s), x{a.relapse_multiplier:.2f}",
        ),
        _wrap(cfg.persona.assistant_name.upper(), fields["acknowledgement"]),
        _wrap("DIRECTIVE", fields["directive"]),
        _wrap("SKILL", f"{execution.skill} - {len(execution.actions)} local action(s) executed [simulated]"),
    ]
    for action in execution.actions:
        lines.append(" " * 15 + "• " + action)
    for index, step in enumerate(fields["steps"], start=1):
        lines.append(_wrap("STEP " + str(index), step))
    lines.append(_wrap("CHECK-IN", f"Presence re-check in {fields['follow_up_minutes']}m (tone: {fields['tone']})."))
    lines.append(_wrap("REFLECT", fields["reflection_question"]))
    token_text = ""
    if usage:
        token_text = f" · {usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} tokens"
    lines.append(_wrap("SOURCE", f"{source} · {model} · {latency_ms} ms{token_text}"))
    lines.append("─" * WIDTH)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation Scenarios (Sensors & Fault Ingestion)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    start: str
    end: str
    room: str
    engaged: bool
    camera: bool = True
    mmwave: bool = True


@dataclass(frozen=True)
class Ghost:
    at: str
    room: str
    source: str = "mmwave"
    confidence: float = 0.58
    motion: float = 0.60


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    narrative: str
    segments: Tuple[Segment, ...]
    end: str
    ghosts: Tuple[Ghost, ...] = ()


CRISIS_SEGMENTS: Tuple[Segment, ...] = (
    Segment("20:30", "20:50", "LIVING_ROOM", False),
    Segment("20:50", "22:30", "BEDROOM_DESK", True),
    Segment("22:30", "22:45", "BEDROOM_DESK", True, camera=False),
    Segment("22:45", "23:00", "BEDROOM_DESK", True),
    Segment("23:00", "23:46", "BEDROOM_DESK", False),
)

SCENARIOS: Dict[str, Scenario] = {
    "crisis": Scenario(
        "crisis",
        "11:45 PM Bedroom Desk Crisis",
        "Stationary at bedroom desk, 45 minutes idle, prolonged nocturnal isolation. "
        "Includes transient camera dropout (22:30-22:45) and ghost radar anomaly in living room.",
        CRISIS_SEGMENTS,
        "23:45",
        (Ghost("22:10", "LIVING_ROOM", "mmwave", 0.62, 0.55),),
    ),
    "recovery": Scenario(
        "recovery",
        "Intervention Follow-up Recovery",
        "Identical late-night drift, but following intervention protocol, user complies "
        "by walking to kitchen and living room.",
        CRISIS_SEGMENTS
        + (
            Segment("23:46", "23:52", "BEDROOM_DESK", False),
            Segment("23:52", "24:05", "KITCHEN", True),
            Segment("24:05", "24:25", "LIVING_ROOM", False),
        ),
        "24:25",
        (Ghost("22:10", "LIVING_ROOM", "mmwave", 0.62, 0.55),),
    ),
    "relapse": Scenario(
        "relapse",
        "Intervention Ignored (Escalation)",
        "User disregards initial directive and remains sedentary. Follow-up fails and risk escalates.",
        CRISIS_SEGMENTS + (Segment("23:46", "24:21", "BEDROOM_DESK", False),),
        "24:20",
        (Ghost("22:10", "LIVING_ROOM", "mmwave", 0.62, 0.55),),
    ),
    "dropout": Scenario(
        "dropout",
        "Total Sensor Blackout Resilience",
        "Hardware power interruption silences all sensors for 14 minutes mid-session. System refuses to hallucinate.",
        (
            Segment("22:40", "23:00", "BEDROOM_DESK", True),
            Segment("23:00", "23:20", "BEDROOM_DESK", False),
            Segment("23:34", "23:50", "BEDROOM_DESK", False),
        ),
        "23:49",
    ),
    "false_positive": Scenario(
        "false_positive",
        "Ghost Reflection Immunity",
        "User actively building in living room while bedroom radar sensor encounters multi-path ghost reflections.",
        (Segment("22:30", "23:50", "LIVING_ROOM", True),),
        "23:49",
        (
            Ghost("23:05", "BEDROOM_DESK"),
            Ghost("23:06", "BEDROOM_DESK"),
            Ghost("23:07", "BEDROOM_DESK"),
        ),
    ),
}
ALL_ORDER = ("false_positive", "dropout", "crisis", "recovery", "relapse")


def _at(day: date, hhmm: str) -> datetime:
    hour_text, minute_text = hhmm.split(":")
    return datetime.combine(day, datetime.min.time()) + timedelta(hours=int(hour_text), minutes=int(minute_text))


def build_events(day: date, scenario: Scenario, rng: random.Random) -> List[SensorEvent]:
    events: List[SensorEvent] = []
    for segment in scenario.segments:
        current, stop = _at(day, segment.start), _at(day, segment.end)
        while current < stop:
            motion = rng.uniform(0.35, 0.55) if segment.engaged else rng.uniform(0.02, 0.10)
            if segment.mmwave:
                events.append(
                    SensorEvent(
                        current, "mmwave", segment.room, True, round(motion, 3),
                        round(rng.uniform(0.86, 0.97), 3), None
                    )
                )
            if segment.camera:
                events.append(
                    SensorEvent(
                        current, "camera", segment.room, True, round(motion, 3),
                        round(rng.uniform(0.88, 0.98), 3), segment.engaged
                    )
                )
            current += timedelta(minutes=1)
    for ghost in scenario.ghosts:
        events.append(
            SensorEvent(_at(day, ghost.at), ghost.source, ghost.room, True, ghost.motion, ghost.confidence, None)
        )
    events.sort(key=lambda event: event.timestamp)
    return events


def bootstrap_history(engine: BehaviorEngine, day: date, rng: random.Random) -> None:
    profile: Dict[str, Dict[str, Tuple[float, ...]]] = {
        "DAYTIME": {
            "DESK_ENGAGED": (0.78, 0.12, 0.02, 0.06, 0.02),
            "DESK_IDLE": (0.30, 0.48, 0.06, 0.14, 0.02),
            "BED_SCREEN": (0.10, 0.12, 0.60, 0.15, 0.03),
            "COMMON_AREA": (0.25, 0.08, 0.07, 0.52, 0.08),
            "AWAY": (0.15, 0.00, 0.00, 0.15, 0.70),
        },
        "EVENING": {
            "DESK_ENGAGED": (0.62, 0.22, 0.08, 0.07, 0.01),
            "DESK_IDLE": (0.14, 0.60, 0.18, 0.07, 0.01),
            "BED_SCREEN": (0.04, 0.14, 0.70, 0.11, 0.01),
            "COMMON_AREA": (0.20, 0.15, 0.20, 0.43, 0.02),
            "AWAY": (0.10, 0.05, 0.10, 0.30, 0.45),
        },
        "LATE_NIGHT": {
            "DESK_ENGAGED": (0.50, 0.35, 0.10, 0.03, 0.02),
            "DESK_IDLE": (0.04, 0.80, 0.14, 0.01, 0.01),
            "BED_SCREEN": (0.02, 0.14, 0.82, 0.01, 0.01),
            "COMMON_AREA": (0.08, 0.22, 0.38, 0.30, 0.02),
            "AWAY": (0.05, 0.05, 0.30, 0.55, 0.05),
        },
    }
    phases = (("DAYTIME", 36, "DESK_ENGAGED"), ("EVENING", 20, "COMMON_AREA"), ("LATE_NIGHT", 20, "DESK_ENGAGED"))
    for _ in range(14):
        for window, steps, state in phases:
            for _ in range(steps):
                nxt = rng.choices(STATES, weights=profile[window][state])[0]
                engine.markov.observe_transition(window, state, nxt)
                state = nxt
    for back in (3, 2, 1):
        engine.store.data["nights"].append(
            {
                "night": (day - timedelta(days=back)).isoformat(),
                "peak_tier": "RED",
                "interventions": 2,
                "relapsed": True,
                "unresolved": 0,
                "synthetic": True,
            }
        )
    engine.persist()


# ---------------------------------------------------------------------------
# CLI Helpers
# ---------------------------------------------------------------------------
def format_matrix(engine: BehaviorEngine, window: str) -> str:
    short = {"DESK_ENGAGED": "ENGAGED", "DESK_IDLE": "IDLE", "BED_SCREEN": "BED", "COMMON_AREA": "COMMON", "AWAY": "AWAY"}
    matrix = engine.markov.matrix(window)
    lines = [" " * 14 + "".join(f"{short[s]:>9}" for s in STATES)]
    for i, state in enumerate(STATES):
        lines.append(f"{short[state]:<14}" + "".join(f"{matrix[i][j]:>9.2f}" for j in range(len(STATES))))
    return "\n".join(lines)


def print_ledger(store: StateStore) -> None:
    ledger = store.data["privacy_ledger"]
    print(f"PRIVACY AUDIT LEDGER - {len(ledger)} remote call(s) logged at {store.path}")
    for entry in ledger:
        print(f"  {entry['sent_at']}  {entry['status']:<10} {entry['model']}  sha256={entry['sha256'][:16]}…")
        print("    payload: " + json.dumps(entry["payload"], sort_keys=True))


def handle_intervention(
    a: RiskAssessment,
    reason: str,
    cfg: MasterpieceConfig,
    engine: BehaviorEngine,
    store: StateStore,
    nebius: Optional[NebiusClient],
    actuators: LocalActuators,
) -> None:
    print(paint(f"  >>> INTERVENTION TRIGGERED: {reason}", a.tier.name))
    payload = engine.build_cloud_payload(a, list(SKILLS))
    source, model, latency_ms, usage = "LOCAL_FALLBACK", "local-heuristics", 0, {}
    fields: Optional[Dict[str, Any]] = None

    if nebius is not None:
        try:
            audit_payload(payload)
            messages = build_messages(payload, cfg)
            fields, meta = nebius.complete_plan(messages, list(SKILLS))
            source, model = meta["provider"], meta["model"]
            latency_ms, usage = meta["latency_ms"], meta["usage"]
            store.log_cloud_call(payload, model, "OK", latency_ms, usage)
        except PrivacyViolation as exc:
            print(f"  [!] Privacy guard intervention blocked outbound call: {exc}")
        except NebiusError as exc:
            print(f"  [!] Cloud endpoint unavailable ({exc}); engaging local fallback protocol")
            store.log_cloud_call(payload, cfg.nebius.model, "FAILED", 0, {})

    if fields is None:
        fields = local_fallback_plan(a, cfg)

    execution = actuators.execute(fields["skill"], a.timestamp)
    engine.record_intervention(a, fields, source, model)
    store.log_skill(
        {
            "timestamp": a.timestamp.isoformat(),
            "skill": execution.skill,
            "actions": execution.actions,
            "simulated": execution.simulated,
        }
    )
    print(render_protocol(a, fields, execution, source, model, latency_ms, usage, cfg))
    engine.persist()


def run_scenario(
    scenario: Scenario,
    day: date,
    cfg: MasterpieceConfig,
    engine: BehaviorEngine,
    store: StateStore,
    nebius: Optional[NebiusClient],
    actuators: LocalActuators,
    rng: random.Random,
) -> None:
    print("\n" + "═" * WIDTH)
    print(f" SCENARIO [{scenario.key}] {scenario.title}")
    print(_wrap("STORY", scenario.narrative))
    print("═" * WIDTH)
    print(" time  | room         | state        | idle   | iso     | conf | risk         | P(trap) | flags")

    engine.begin_session()
    fusion = SensorFusion(cfg.thresholds)
    events = build_events(day, scenario, rng)
    now = events[0].timestamp.replace(second=0, microsecond=0)
    stop = _at(day, scenario.end)
    index = 0
    last_flags: Tuple[str, ...] = ()
    last_now = now
    interventions_fired = 0

    while now <= stop:
        while index < len(events) and events[index].timestamp <= now:
            fusion.ingest(events[index])
            index += 1
        obs = fusion.snapshot(now)
        last_now = now
        if obs is not None:
            engine.observe(obs)
            outcome = engine.evaluate_followup(obs)
            a = engine.assess(obs)
            flags_tuple = tuple(a.flags)
            show = (
                now.minute % 15 == 0
                or a.tier_changed
                or (bool(flags_tuple) and flags_tuple != last_flags)
            )
            if show:
                tier_text = paint(f"{a.score:5.1f} {a.tier.name:<6}", a.tier.name)
                print(
                    f" {now.strftime('%H:%M')} | {a.room:<12} | {a.state:<12} | {a.idle_minutes:4.0f}m  | "
                    f"{a.isolation_minutes:5.0f}m  | {a.confidence:.2f} | {tier_text} | {a.trap_probability:.2f}    | "
                    f"{','.join(a.flags)}"
                )
            last_flags = flags_tuple
            if outcome:
                print(f"       ↳ Follow-up evaluated for '{outcome['skill']}': {outcome['outcome']}")
            go, reason = engine.should_intervene(a)
            if go:
                handle_intervention(a, reason, cfg, engine, store, nebius, actuators)
                interventions_fired += 1
        now += timedelta(minutes=1)

    record = engine.close_night(last_now)
    engine.persist()
    print(
        f" RESULT [{scenario.key}] peak={record['peak_tier']} interventions={interventions_fired} "
        f"relapsed={record['relapsed']} unresolved_followups={record['unresolved']}"
    )


# ---------------------------------------------------------------------------
# Streamlit Interactive Web Application
# ---------------------------------------------------------------------------
def run_streamlit_app() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("[!] Streamlit is required to launch the UI dashboard. Run: pip install streamlit")
        sys.exit(1)

    st.set_page_config(
        page_title="Masterpiece: Contextual Presence & Behavioral Supervision",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .risk-red { color: #ff4b4b; font-weight: bold; }
        .risk-orange { color: #ffa500; font-weight: bold; }
        .risk-yellow { color: #ffeb3b; font-weight: bold; }
        .risk-green { color: #4caf50; font-weight: bold; }
        .intervention-card {
            border: 2px solid #3366ff;
            border-radius: 8px;
            padding: 18px;
            background-color: rgba(30, 60, 120, 0.15);
            margin-top: 15px;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛡️ Masterpiece: Behavioral Supervision System")
    st.caption(
        "Edge Heuristic Markov Model + Universal OpenAI Router (Nebius Token Factory & Google AI Studio)"
    )

    # --- Sidebar Configuration Layer ---
    st.sidebar.header("Inference Gateway")
    default_key, detected_prov = resolve_system_credentials()

    api_key_input = st.sidebar.text_input(
        "API Key",
        value=default_key,
        type="password",
        help="Paste a Nebius Token Factory key OR a Google AI Studio key (starts with AIza...).",
    )

    # Resolve active configuration
    cfg = load_config(api_key_override=api_key_input if api_key_input else None)

    st.sidebar.caption(f"Active Provider: **{cfg.nebius.provider_name}**")
    base_url_input = st.sidebar.text_input("Base URL", value=cfg.nebius.base_url)
    model_input = st.sidebar.text_input("Model Identifier", value=cfg.nebius.model)

    # Reload with any manual overrides
    cfg = load_config(
        api_key_override=api_key_input,
        base_url_override=base_url_input,
        model_override=model_input,
    )

    offline_mode = st.sidebar.checkbox("Force Offline Mode (Edge Heuristics Only)", value=not bool(api_key_input))

    st.sidebar.markdown("---")
    st.sidebar.header("Simulation Settings")
    scenario_choice = st.sidebar.selectbox("Evaluation Scenario", list(SCENARIOS.keys()), index=0)
    seed_input = st.sidebar.number_input("RNG Determinism Seed", min_value=1, max_value=9999, value=7)
    warm_start = st.sidebar.checkbox("Ingest 14-Day Sticky Drift Profile", value=True)

    store = StateStore(
        cfg.storage.state_path,
        cfg.storage.max_ledger_entries,
        cfg.storage.max_interventions,
        cfg.storage.max_nights,
    )
    engine = BehaviorEngine(cfg, store)
    actuators = LocalActuators()
    rng = random.Random(seed_input)
    day = date.today()

    if warm_start and engine.is_cold_start():
        bootstrap_history(engine, day, rng)

    nebius_client: Optional[NebiusClient] = None
    if not offline_mode and cfg.nebius.configured:
        try:
            nebius_client = NebiusClient(cfg.nebius)
        except Exception as exc:
            st.sidebar.error(f"Inference setup error: {exc}")

    tab_sim, tab_matrix, tab_ledger, tab_state = st.tabs(
        ["Simulation & Protocol Execution", "Habit Markov Engine", "Zero-Knowledge Privacy Ledger", "Local State Vault"]
    )

    with tab_sim:
        st.subheader(f"Scenario: {SCENARIOS[scenario_choice].title}")
        st.info(SCENARIOS[scenario_choice].narrative)

        if st.button("Execute Pipeline Trace", type="primary"):
            sim_scenario = SCENARIOS[scenario_choice]
            events = build_events(day, sim_scenario, rng)
            fusion = SensorFusion(cfg.thresholds)
            engine.begin_session()

            now = events[0].timestamp.replace(second=0, microsecond=0)
            stop = _at(day, sim_scenario.end)
            index = 0

            trace_rows = []
            intervention_blocks = []

            with st.spinner(f"Running edge state engine & querying {cfg.nebius.provider_name}..."):
                while now <= stop:
                    while index < len(events) and events[index].timestamp <= now:
                        fusion.ingest(events[index])
                        index += 1
                    obs = fusion.snapshot(now)
                    if obs is not None:
                        engine.observe(obs)
                        outcome = engine.evaluate_followup(obs)
                        a = engine.assess(obs)

                        if now.minute % 15 == 0 or a.tier_changed or a.flags:
                            trace_rows.append(
                                {
                                    "Time": now.strftime("%H:%M"),
                                    "Room": a.room,
                                    "State": a.state,
                                    "Idle (m)": int(a.idle_minutes),
                                    "Iso (m)": int(a.isolation_minutes),
                                    "Confidence": round(a.confidence, 2),
                                    "Score": round(a.score, 1),
                                    "Tier": a.tier.name,
                                    "P(Trap)": round(a.trap_probability, 2),
                                    "Flags": ", ".join(a.flags) if a.flags else "NORMAL",
                                }
                            )

                        go, reason = engine.should_intervene(a)
                        if go:
                            payload = engine.build_cloud_payload(a, list(SKILLS))
                            source, model, latency_ms, usage = "LOCAL_FALLBACK", "local-heuristics", 0, {}
                            fields = None

                            if nebius_client is not None:
                                try:
                                    audit_payload(payload)
                                    messages = build_messages(payload, cfg)
                                    fields, meta = nebius_client.complete_plan(messages, list(SKILLS))
                                    source, model = meta["provider"], meta["model"]
                                    latency_ms, usage = meta["latency_ms"], meta["usage"]
                                    store.log_cloud_call(payload, model, "OK", latency_ms, usage)
                                except Exception as exc:
                                    st.warning(f"Cloud call bypassed ({exc}); engaged edge fallback.")
                                    store.log_cloud_call(payload, cfg.nebius.model, "FAILED", 0, {})

                            if fields is None:
                                fields = local_fallback_plan(a, cfg)

                            execution = actuators.execute(fields["skill"], a.timestamp)
                            engine.record_intervention(a, fields, source, model)
                            store.log_skill(
                                {
                                    "timestamp": a.timestamp.isoformat(),
                                    "skill": execution.skill,
                                    "actions": execution.actions,
                                    "simulated": execution.simulated,
                                }
                            )
                            engine.persist()

                            intervention_blocks.append(
                                {
                                    "assessment": a,
                                    "fields": fields,
                                    "execution": execution,
                                    "source": source,
                                    "model": model,
                                    "latency_ms": latency_ms,
                                    "usage": usage,
                                    "reason": reason,
                                }
                            )

                    now += timedelta(minutes=1)

            st.success("Pipeline trace execution complete.")

            col1, col2, col3, col4 = st.columns(4)
            final_assessment = a if "a" in locals() and a is not None else None
            if final_assessment:
                col1.metric("Peak Tier", engine._eng.get("tier", "GREEN"))
                col2.metric("Final Risk Score", f"{final_assessment.score:.1f}/100")
                col3.metric("Trap Forecast P(Trap)", f"{final_assessment.trap_probability:.2f}")
                col4.metric("Interventions Fired", len(intervention_blocks))

            if intervention_blocks:
                st.markdown("### Active Interventions Dispatched")
                for item in intervention_blocks:
                    a_item = item["assessment"]
                    f_item = item["fields"]
                    e_item = item["execution"]
                    st.markdown(
                        f"""
                        <div class="intervention-card">
                            <h4>🚨 {cfg.persona.assistant_name.upper()} INTERVENTION PROTOCOL ({a_item.timestamp.strftime('%H:%M')} | {a_item.tier.name})</h4>
                            <p><strong>Acknowledgement:</strong> <em>"{f_item['acknowledgement']}"</em></p>
                            <p><strong>Directive:</strong> <strong>{f_item['directive']}</strong></p>
                            <p><strong>Skill Executed:</strong> <code>{e_item.skill}</code> (Tone: <em>{f_item['tone']}</em>)</p>
                            <p><strong>Physical Steps:</strong></p>
                            <ul>
                                {"".join(f"<li>{s}</li>" for s in f_item['steps'])}
                            </ul>
                            <p><strong>Follow-up Window:</strong> {f_item['follow_up_minutes']} minutes</p>
                            <p><strong>Somatic Reflection:</strong> <em>"{f_item['reflection_question']}"</em></p>
                            <small>Source: {item['source']} | Model: {item['model']} | Latency: {item['latency_ms']}ms</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("### Telemetry Stream Log")
            st.dataframe(trace_rows, use_container_width=True)

    with tab_matrix:
        st.subheader("Learned Late-Night Markov Habit Dynamics")
        mat = engine.markov.matrix("LATE_NIGHT")
        import pandas as pd

        df = pd.DataFrame(mat, index=STATES, columns=STATES)
        st.dataframe(df.style.format("{:.2%}"), use_container_width=True)
        st.caption(
            f"Prior Drift Metric: {engine.markov.drift_from_prior('LATE_NIGHT'):.4f} | "
            f"Expected Desk Idle Dwell: {engine.markov.expected_dwell_minutes('LATE_NIGHT', 'DESK_IDLE'):.1f} min"
        )

    with tab_ledger:
        st.subheader("Edge Privacy Ledger (Audited Outbound Telemetry)")
        st.caption("Verifiable proof: Zero audio, video, or keystroke telemetry ever departs local hardware.")
        ledger_entries = store.data.get("privacy_ledger", [])
        if ledger_entries:
            st.json(ledger_entries)
        else:
            st.info("No outbound cloud transactions logged. System is operating purely on edge heuristics.")

    with tab_state:
        st.subheader("Encrypted Local State Vault")
        st.json(store.data)
        if st.button("Purge All Local State"):
            store.purge()
            st.success("Local state securely expunged.")


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Entrypoint
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masterpiece: Contextual Presence & Behavioral Supervision System")
    parser.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="crisis")
    parser.add_argument("--offline", action="store_true", help="Skip cloud call; execute edge heuristics protocol")
    parser.add_argument("--no-seed", action="store_true", help="Skip 14-day simulated behavioral profile bootstrap")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for deterministic evaluation")
    parser.add_argument("--purge", action="store_true", help="Erase all local Masterpiece state and exit")
    parser.add_argument("--export", action="store_true", help="Export full on-device state as JSON and exit")
    parser.add_argument("--ledger", action="store_true", help="Inspect cloud privacy audit ledger and exit")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Streamlit evaluation dashboard")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    is_streamlit = False
    try:
        import streamlit as st
        is_streamlit = getattr(st, "_is_running_with_streamlit", False)
    except Exception:
        pass

    if is_streamlit or ("--ui" in (argv or sys.argv)):
        run_streamlit_app()
        return 0

    args = parse_args(argv)
    cfg = load_config()
    store = StateStore(
        cfg.storage.state_path,
        cfg.storage.max_ledger_entries,
        cfg.storage.max_interventions,
        cfg.storage.max_nights,
    )

    if args.purge:
        store.purge()
        print(f"All Masterpiece data stored at {cfg.storage.state_path} has been purged.")
        return 0
    if args.export:
        print(json.dumps(store.data, indent=2, sort_keys=True))
        return 0
    if args.ledger:
        print_ledger(store)
        return 0

    engine = BehaviorEngine(cfg, store)
    actuators = LocalActuators()
    rng = random.Random(args.seed)
    day = date.today()

    nebius: Optional[NebiusClient] = None
    if not args.offline:
        try:
            nebius = NebiusClient(cfg.nebius)
        except NebiusError as exc:
            print(f"[!] Cloud endpoint bypassed: {exc}. Operating with edge fallback protocol.")

    print("═" * WIDTH)
    print(" MASTERPIECE · Contextual Presence & Behavioral Supervision System")
    print("═" * WIDTH)
    print(_wrap("EDGE", "Pure-Python Markov habit transition engine + sensor fusion (local)"))
    mode = f"{cfg.nebius.model} via {cfg.nebius.provider_name}" if nebius else "OFFLINE (local edge fallback)"
    print(_wrap("CLOUD", mode))
    print(_wrap("DATA", f"State isolated locally at {cfg.storage.state_path} (purge via --purge)"))

    if not args.no_seed and engine.is_cold_start():
        before = engine.markov.matrix("LATE_NIGHT")[1][1]
        bootstrap_history(engine, day, rng)
        after = engine.markov.matrix("LATE_NIGHT")[1][1]
        print(
            _wrap(
                "WARM START",
                f"14-day habit profile loaded. Late-night P(IDLE->IDLE): {before:.2f} -> {after:.2f}; "
                "3 prior relapsed sessions registered.",
            )
        )

    keys = list(ALL_ORDER) if args.scenario == "all" else [args.scenario]
    for key in keys:
        run_scenario(SCENARIOS[key], day, cfg, engine, store, nebius, actuators, rng)

    print("\n" + "═" * WIDTH)
    print(" LEARNED EDGE MARKOV DYNAMICS")
    print("═" * WIDTH)
    print(" Late-night transition matrix (rows = from, columns = to):")
    print(format_matrix(engine, "LATE_NIGHT"))
    print(
        f"\n Drift from prior: {engine.markov.drift_from_prior('LATE_NIGHT'):.3f} · "
        f"Expected desk idle dwell: {engine.markov.expected_dwell_minutes('LATE_NIGHT', 'DESK_IDLE'):.0f} min"
    )
    print(
        f" Privacy ledger calls: {len(store.data['privacy_ledger'])} · Skills executed: {len(store.data['skill_log'])} · "
        f"Nights tracked: {len(store.data['nights'])}"
    )
    print(" Inspect: python app.py --ledger | --export    Purge: python app.py --purge")
    print(" Streamlit UI: streamlit run app.py or python app.py --ui")
    engine.persist()
    return 0


if __name__ == "__main__":
    sys.exit(main())
