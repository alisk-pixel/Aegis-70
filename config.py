# MIT License
#
# Copyright (c) 2025 AEGIS-70 Contributors
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
config.py - AEGIS-70: Autonomous Edge Guardian & Behavioral Supervision System

Single source of truth for:
  * Universal OpenAI router (Nebius Token Factory & Google AI Studio auto-swap)
  * Behavioral thresholds (isolation limits, sleep windows, risk multipliers)
  * Markov chain vocabulary, priors, and learning parameters
  * On-device state storage configuration
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------
STATES: Tuple[str, ...] = ("DESK_ENGAGED", "DESK_IDLE", "BED_SCREEN", "COMMON_AREA", "AWAY")
TRAP_STATES: Tuple[str, ...] = ("DESK_IDLE", "BED_SCREEN")
ROOMS: Tuple[str, ...] = ("BEDROOM_DESK", "BEDROOM_BED", "LIVING_ROOM", "KITCHEN", "OUTSIDE")
ISOLATION_ROOMS: Tuple[str, ...] = ("BEDROOM_DESK", "BEDROOM_BED")
WINDOWS: Tuple[str, ...] = ("DAYTIME", "EVENING", "LATE_NIGHT")

# Transition priors per time window.
# Rows: Current State, Columns: Next State (ordered by STATES tuple).
TRANSITION_PRIORS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "DAYTIME": {
        "DESK_ENGAGED": (0.80, 0.08, 0.01, 0.07, 0.04),
        "DESK_IDLE": (0.35, 0.40, 0.05, 0.15, 0.05),
        "BED_SCREEN": (0.10, 0.10, 0.55, 0.20, 0.05),
        "COMMON_AREA": (0.25, 0.05, 0.05, 0.55, 0.10),
        "AWAY": (0.15, 0.00, 0.00, 0.15, 0.70),
    },
    "EVENING": {
        "DESK_ENGAGED": (0.65, 0.15, 0.04, 0.12, 0.04),
        "DESK_IDLE": (0.20, 0.50, 0.12, 0.15, 0.03),
        "BED_SCREEN": (0.05, 0.10, 0.65, 0.18, 0.02),
        "COMMON_AREA": (0.20, 0.08, 0.12, 0.55, 0.05),
        "AWAY": (0.10, 0.00, 0.05, 0.25, 0.60),
    },
    "LATE_NIGHT": {
        "DESK_ENGAGED": (0.50, 0.33, 0.10, 0.05, 0.02),
        "DESK_IDLE": (0.04, 0.76, 0.16, 0.03, 0.01),
        "BED_SCREEN": (0.02, 0.16, 0.80, 0.01, 0.01),
        "COMMON_AREA": (0.08, 0.22, 0.35, 0.33, 0.02),
        "AWAY": (0.05, 0.05, 0.30, 0.50, 0.10),
    },
}

PROVIDER_DEFAULTS = {
    "nebius": {
        "name": "Nebius Token Factory",
        "base_url": "https://api.tokenfactory.nebius.com/v1/",
        "model": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    },
    "google": {
        "name": "Google AI Studio",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
    },
}


# ---------------------------------------------------------------------------
# Environment parsing helpers
# ---------------------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw is not None and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _env_time(name: str, default: time) -> time:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        hour_text, minute_text = raw.strip().split(":")
        return time(int(hour_text), int(minute_text))
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must look like HH:MM, got {raw!r}") from exc


# ---------------------------------------------------------------------------
# Provider Detection & Universal Key Resolution
# ---------------------------------------------------------------------------
def detect_provider_from_key(key: str) -> str:
    """Detects provider by key signature: Google keys start with AIza."""
    if key.strip().startswith("AIza"):
        return "google"
    return "nebius"


def resolve_system_credentials() -> Tuple[str, str]:
    """
    Checks environment variables in strict priority:
      1. NEBIUS_API_KEY -> Nebius Token Factory (Primary Hackathon Provider)
      2. GEMINI_API_KEY or GOOGLE_API_KEY -> Google AI Studio
      3. OPENAI_API_KEY -> signature-based auto-detection
    """
    nebius_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if nebius_key:
        return nebius_key, "nebius"

    gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")).strip()
    if gemini_key:
        return gemini_key, "google"

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return openai_key, detect_provider_from_key(openai_key)

    return "", "nebius"


# ---------------------------------------------------------------------------
# Configuration sections
# ---------------------------------------------------------------------------
@dataclass
class NebiusConfig:
    """Universal OpenAI-compatible API Configuration for AEGIS-70."""

    api_key: str = ""
    provider: str = "nebius"
    provider_name: str = "Nebius Token Factory"
    base_url: str = ""
    model: str = ""
    temperature: float = field(default_factory=lambda: _env_float("AEGIS_TEMPERATURE", 0.35))
    top_p: float = field(default_factory=lambda: _env_float("AEGIS_TOP_P", 0.90))
    max_tokens: int = field(default_factory=lambda: _env_int("AEGIS_MAX_TOKENS", 700))
    timeout_seconds: float = field(default_factory=lambda: _env_float("AEGIS_TIMEOUT_SECONDS", 25.0))
    max_retries: int = field(default_factory=lambda: _env_int("AEGIS_MAX_RETRIES", 3))
    backoff_base_seconds: float = field(default_factory=lambda: _env_float("AEGIS_BACKOFF_SECONDS", 1.5))

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key, self.provider = resolve_system_credentials()
        else:
            self.provider = detect_provider_from_key(self.api_key)

        # Allow judges' environment variables to swap to Nebius automatically
        if os.getenv("NEBIUS_API_KEY", "").strip():
            self.provider = "nebius"
            if not self.api_key:
                self.api_key = os.getenv("NEBIUS_API_KEY", "").strip()

        env_base = os.getenv("NEBIUS_BASE_URL", "").strip() or os.getenv("LLM_BASE_URL", "").strip()
        env_model = os.getenv("NEBIUS_MODEL", "").strip() or os.getenv("LLM_MODEL", "").strip()

        if self.provider == "google":
            self.provider_name = PROVIDER_DEFAULTS["google"]["name"]
            self.base_url = env_base or PROVIDER_DEFAULTS["google"]["base_url"]
            self.model = env_model or PROVIDER_DEFAULTS["google"]["model"]
        else:
            self.provider_name = PROVIDER_DEFAULTS["nebius"]["name"]
            self.base_url = env_base or PROVIDER_DEFAULTS["nebius"]["base_url"]
            self.model = env_model or PROVIDER_DEFAULTS["nebius"]["model"]

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())


@dataclass
class ThresholdConfig:
    """Environment thresholds: sleep schedules, idleness limits, risk weightings."""

    sleep_window_start: time = field(default_factory=lambda: _env_time("SLEEP_WINDOW_START", time(23, 0)))
    sleep_window_end: time = field(default_factory=lambda: _env_time("SLEEP_WINDOW_END", time(4, 0)))
    evening_start: time = field(default_factory=lambda: _env_time("EVENING_START", time(18, 0)))
    pre_window_ramp_minutes: int = 60
    post_window_tail_minutes: int = 120

    idle_grace_minutes: float = 10.0
    idle_midpoint_minutes: float = 28.0
    idle_scale_minutes: float = 8.5
    isolation_limit_minutes: float = field(
        default_factory=lambda: _env_float("ISOLATION_LIMIT_MINUTES", 120.0)
    )
    isolation_scale_minutes: float = 25.0
    idle_intervention_minutes: float = field(
        default_factory=lambda: _env_float("IDLE_INTERVENTION_MINUTES", 40.0)
    )

    location_risk: Dict[str, float] = field(
        default_factory=lambda: {
            "BEDROOM_DESK": 0.60,
            "BEDROOM_BED": 0.85,
            "LIVING_ROOM": 0.10,
            "KITCHEN": 0.05,
            "OUTSIDE": 0.00,
        }
    )
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "location": 0.10,
            "time": 0.10,
            "idle": 0.55,
            "isolation": 0.10,
            "markov": 0.15,
        }
    )
    engagement_dampener: float = 0.65
    relapse_multiplier_per_night: float = 0.06
    relapse_multiplier_cap: float = 1.25
    tier_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"YELLOW": 30.0, "ORANGE": 52.0, "RED": 82.0}
    )
    deescalate_confirmations: int = 5
    cooldown_minutes: Dict[str, int] = field(default_factory=lambda: {"ORANGE": 30, "RED": 15})
    max_interventions_per_night: int = 6

    min_event_confidence: float = 0.35
    min_action_confidence: float = 0.55
    room_change_confirmations: int = 2
    room_change_sustain_seconds: int = 120
    engaged_motion_threshold: float = 0.30
    sensor_fresh_seconds: int = 90
    sensor_dropout_seconds: int = 180
    sensor_blind_seconds: int = 600
    single_sensor_confidence_factor: float = 0.75


@dataclass
class MarkovConfig:
    step_minutes: int = 15
    horizon_steps: int = 3
    prior_strength: float = 6.0
    decay: float = 0.985
    learn_min_confidence: float = 0.60
    max_gap_steps: int = 3


@dataclass
class StorageConfig:
    data_dir: Path = field(default_factory=lambda: Path(_env_str("AEGIS_DATA_DIR", "./aegis_data")))
    state_filename: str = "state.json"
    max_ledger_entries: int = 200
    max_interventions: int = 200
    max_nights: int = 90

    @property
    def state_path(self) -> Path:
        return self.data_dir / self.state_filename


@dataclass
class PersonaConfig:
    assistant_name: str = field(default_factory=lambda: _env_str("AEGIS_NAME", "AEGIS"))
    address_as: str = field(default_factory=lambda: _env_str("AEGIS_ADDRESS_AS", "Builder"))


@dataclass
class MasterpieceConfig:
    nebius: NebiusConfig = field(default_factory=NebiusConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    markov: MarkovConfig = field(default_factory=MarkovConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)


def validate_config(cfg: MasterpieceConfig) -> None:
    t = cfg.thresholds
    expected_weights = {"location", "time", "idle", "isolation", "markov"}
    if set(t.weights) != expected_weights:
        raise ValueError(f"weights must define exactly {sorted(expected_weights)}")
    if abs(sum(t.weights.values()) - 1.0) > 1e-6:
        raise ValueError("risk weights must sum to 1.0")
    thresholds = [t.tier_thresholds[name] for name in ("YELLOW", "ORANGE", "RED")]
    if not (0 < thresholds[0] < thresholds[1] < thresholds[2] <= 100):
        raise ValueError("tier thresholds must satisfy 0 < YELLOW < ORANGE < RED <= 100")
    for room in ROOMS:
        if room not in t.location_risk:
            raise ValueError(f"location_risk is missing room {room}")
    for tier in ("ORANGE", "RED"):
        if tier not in t.cooldown_minutes:
            raise ValueError(f"cooldown_minutes is missing tier {tier}")
    if not (0.0 < t.engagement_dampener <= 1.0):
        raise ValueError("engagement_dampener must be in (0, 1]")
    if t.sensor_fresh_seconds >= t.sensor_dropout_seconds or t.sensor_dropout_seconds >= t.sensor_blind_seconds:
        raise ValueError("sensor_fresh < sensor_dropout < sensor_blind must hold")
    m = cfg.markov
    if m.step_minutes <= 0 or m.horizon_steps <= 0 or not (0.0 < m.decay <= 1.0):
        raise ValueError("invalid Markov step/horizon/decay settings")
    for window in WINDOWS:
        if window not in TRANSITION_PRIORS:
            raise ValueError(f"missing transition priors for window {window}")
        for state in STATES:
            row = TRANSITION_PRIORS[window].get(state)
            if row is None or len(row) != len(STATES):
                raise ValueError(f"prior row {window}/{state} must have {len(STATES)} entries")
            if abs(sum(row) - 1.0) > 1e-6:
                raise ValueError(f"prior row {window}/{state} must sum to 1.0")
    n = cfg.nebius
    if n.max_retries < 1:
        raise ValueError("AEGIS_MAX_RETRIES must be >= 1")
    if n.configured and not (n.base_url.startswith("https://") or n.base_url.startswith("http://")):
        raise ValueError("API Base URL must start with http:// or https://")


def load_config(
    api_key_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> MasterpieceConfig:
    cfg = MasterpieceConfig()
    if api_key_override is not None:
        cfg.nebius.api_key = api_key_override.strip()
        cfg.nebius.__post_init__()
    if base_url_override is not None and base_url_override.strip():
        cfg.nebius.base_url = base_url_override.strip()
    if model_override is not None and model_override.strip():
        cfg.nebius.model = model_override.strip()

    validate_config(cfg)
    return cfg
