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
engine.py - Edge Mathematical Runtime for Masterpiece

Pure-Python implementation containing:
  * SensorFusion: Multimodal temporal fusion (mmWave + Edge CV).
  * HabitMarkov: Heuristic, online-learning Markov chain with Dirichlet priors.
  * BehaviorEngine: Behavioral Risk Score, hysteretic FSM, and intervention gating.
  * StateStore: Zero-cloud, atomic, user-owned JSON persistence and privacy ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from config import (
    ISOLATION_ROOMS,
    STATES,
    TRANSITION_PRIORS,
    TRAP_STATES,
    WINDOWS,
    MarkovConfig,
    MasterpieceConfig,
    ThresholdConfig,
)


# ---------------------------------------------------------------------------
# Tiers and Privacy Guard
# ---------------------------------------------------------------------------
class Tier(IntEnum):
    GREEN = 0
    YELLOW = 1
    ORANGE = 2
    RED = 3


PRIVACY_ALLOWLIST = frozenset(
    {
        "schema",
        "local_time",
        "window",
        "room",
        "behavioral_state",
        "idle_minutes",
        "isolation_minutes",
        "risk_score",
        "risk_tier",
        "trap_probability",
        "trap_horizon_minutes",
        "risk_components",
        "relapse_streak_nights",
        "previous_intervention_outcome",
        "sensor_confidence",
        "sensor_flags",
        "allowed_skills",
        "user_urgent_profile",  # <-- Added to permit custom crisis prompts
    }
)


class PrivacyViolation(RuntimeError):
    """Raised when an outbound payload contains fields outside the privacy allowlist."""


def audit_payload(payload: Dict[str, Any]) -> None:
    unknown = set(payload) - PRIVACY_ALLOWLIST
    if unknown:
        raise PrivacyViolation(f"Privacy guard blocked outbound payload. Disallowed keys: {sorted(unknown)}")


# ---------------------------------------------------------------------------
# Numerical Helpers
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _minutes_of_day(ts: datetime) -> float:
    return ts.hour * 60 + ts.minute + ts.second / 60.0


def _cfg_minutes(t: Any) -> float:
    return float(t.hour * 60 + t.minute)


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def night_key(ts: datetime) -> str:
    """A night period runs noon-to-noon; 02:00 belongs to the previous day's evening."""
    return (ts - timedelta(hours=12)).date().isoformat()


def window_for(ts: datetime, cfg: ThresholdConfig) -> str:
    now_m = _minutes_of_day(ts)
    start_m = _cfg_minutes(cfg.sleep_window_start)
    end_m = _cfg_minutes(cfg.sleep_window_end)
    span = (end_m - start_m) % 1440
    if (now_m - start_m) % 1440 < span:
        return "LATE_NIGHT"
    eve_m = _cfg_minutes(cfg.evening_start)
    if (now_m - eve_m) % 1440 < (start_m - eve_m) % 1440:
        return "EVENING"
    return "DAYTIME"


def time_risk(ts: datetime, cfg: ThresholdConfig) -> float:
    now_m = _minutes_of_day(ts)
    start_m = _cfg_minutes(cfg.sleep_window_start)
    end_m = _cfg_minutes(cfg.sleep_window_end)
    span = (end_m - start_m) % 1440
    if (now_m - start_m) % 1440 < span:
        return 1.0
    to_start = (start_m - now_m) % 1440
    if 0 < to_start <= cfg.pre_window_ramp_minutes:
        progress = 1.0 - to_start / cfg.pre_window_ramp_minutes
        return 0.35 + 0.55 * progress
    after_end = (now_m - end_m) % 1440
    if after_end < cfg.post_window_tail_minutes:
        return 0.60 - 0.45 * (after_end / cfg.post_window_tail_minutes)
    return 0.10


def classify_state(room: str, idle_minutes: float, cfg: ThresholdConfig) -> str:
    if room == "BEDROOM_DESK":
        return "DESK_ENGAGED" if idle_minutes < cfg.idle_grace_minutes else "DESK_IDLE"
    if room == "BEDROOM_BED":
        return "BED_SCREEN"
    if room == "OUTSIDE":
        return "AWAY"
    return "COMMON_AREA"


# ---------------------------------------------------------------------------
# Multimodal Sensor Fusion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorEvent:
    timestamp: datetime
    source: str  # "mmwave" | "camera"
    room: str
    presence: bool
    motion_energy: float  # Micro-motion or pose delta (0.0 to 1.0)
    confidence: float
    engaged: Optional[bool] = None  # On-device classifier output (No raw frames stored)


@dataclass(frozen=True)
class FusedObservation:
    timestamp: datetime
    room: str
    state: str
    idle_minutes: float
    isolation_minutes: float
    confidence: float
    flags: Tuple[str, ...]
    blind: bool


class SensorFusion:
    """Fuses multi-sensor feeds into a single verified room presence state."""

    def __init__(self, cfg: ThresholdConfig) -> None:
        self.cfg = cfg
        self._last_by_source: Dict[str, SensorEvent] = {}
        self._last_any: Optional[datetime] = None
        self._room: Optional[str] = None
        self._room_since: Optional[datetime] = None
        self._idle_since: Optional[datetime] = None
        self._isolation_since: Optional[datetime] = None
        self._pending_room: Optional[str] = None
        self._pending_count = 0
        self._pending_sources: Set[str] = set()
        self._pending_first: Optional[datetime] = None
        self._flags: Set[str] = set()

    def _clear_pending(self) -> None:
        self._pending_room = None
        self._pending_count = 0
        self._pending_sources = set()
        self._pending_first = None

    def _engagement(self, event: SensorEvent) -> Optional[bool]:
        if event.engaged is not None:
            return event.engaged
        camera = self._last_by_source.get("camera")
        camera_fresh = (
            camera is not None
            and (event.timestamp - camera.timestamp).total_seconds() <= self.cfg.sensor_fresh_seconds
        )
        if camera_fresh:
            return None
        return event.motion_energy >= self.cfg.engaged_motion_threshold

    def _enter_room(self, event: SensorEvent, engagement: Optional[bool]) -> None:
        previous = self._room
        self._room = event.room
        self._room_since = event.timestamp
        self._idle_since = None if engagement is True else event.timestamp
        if event.room in ISOLATION_ROOMS:
            if previous not in ISOLATION_ROOMS:
                self._isolation_since = event.timestamp
        else:
            self._isolation_since = None
        self._clear_pending()

    def _register_candidate(self, event: SensorEvent) -> bool:
        if self._pending_room == event.room:
            self._pending_count += 1
            self._pending_sources.add(event.source)
        else:
            self._pending_room = event.room
            self._pending_count = 1
            self._pending_sources = {event.source}
            self._pending_first = event.timestamp
        assert self._pending_first is not None
        sustained = (event.timestamp - self._pending_first).total_seconds() >= self.cfg.room_change_sustain_seconds
        enough = self._pending_count >= self.cfg.room_change_confirmations
        return enough and (len(self._pending_sources) >= 2 or sustained)

    def ingest(self, event: SensorEvent) -> None:
        if event.confidence < self.cfg.min_event_confidence:
            self._flags.add("LOW_CONFIDENCE_DROPPED")
            return
        if (
            self._last_any is not None
            and (event.timestamp - self._last_any).total_seconds() > self.cfg.sensor_blind_seconds
        ):
            self._flags.add("IDLE_CLOCK_RESET_AFTER_BLIND")
            self._idle_since = None
            self._room_since = event.timestamp
            self._clear_pending()

        engagement = self._engagement(event)
        self._last_by_source[event.source] = event
        if self._last_any is None or event.timestamp > self._last_any:
            self._last_any = event.timestamp
        if not event.presence:
            return

        if self._room is None:
            self._enter_room(event, engagement)
            return

        if event.room == self._room:
            if self._pending_count > 0:
                self._flags.add("FALSE_POSITIVE_REJECTED")
            self._clear_pending()
            if engagement is True:
                self._idle_since = None
            elif engagement is False and self._idle_since is None:
                self._idle_since = event.timestamp
            return

        if self._register_candidate(event):
            self._enter_room(event, engagement)
        else:
            self._flags.add("ROOM_CHANGE_UNCONFIRMED")

    def snapshot(self, now: datetime) -> Optional[FusedObservation]:
        if self._room is None or self._last_any is None:
            return None
        flags = set(self._flags)
        self._flags.clear()

        gap = (now - self._last_any).total_seconds()
        known = list(self._last_by_source.values())
        fresh = [e for e in known if (now - e.timestamp).total_seconds() <= self.cfg.sensor_fresh_seconds]
        blind = gap > self.cfg.sensor_blind_seconds
        if blind:
            flags.add("SENSOR_BLIND")
        elif gap > self.cfg.sensor_dropout_seconds:
            flags.add("SENSOR_DROPOUT")

        if blind:
            confidence = 0.0
        elif fresh:
            confidence = sum(e.confidence for e in fresh) / len(fresh)
            if len(fresh) < len(known):
                confidence *= self.cfg.single_sensor_confidence_factor
                flags.add("SINGLE_SENSOR_MODE")
        else:
            base = sum(e.confidence for e in known) / len(known)
            confidence = base * math.exp(-(gap - self.cfg.sensor_fresh_seconds) / self.cfg.sensor_dropout_seconds)

        idle = 0.0 if self._idle_since is None else max(0.0, (now - self._idle_since).total_seconds() / 60.0)
        isolation = (
            0.0
            if self._isolation_since is None
            else max(0.0, (now - self._isolation_since).total_seconds() / 60.0)
        )
        return FusedObservation(
            timestamp=now,
            room=self._room,
            state=classify_state(self._room, idle, self.cfg),
            idle_minutes=idle,
            isolation_minutes=isolation,
            confidence=max(0.0, min(1.0, confidence)),
            flags=tuple(sorted(flags)),
            blind=blind,
        )


# ---------------------------------------------------------------------------
# Heuristic Online Markov Chain
# ---------------------------------------------------------------------------
class HabitMarkov:
    """Mathematical state engine modeling habit transitions with Bayesian priors."""

    def __init__(self, cfg: MarkovConfig, saved: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg
        self.index = {state: i for i, state in enumerate(STATES)}
        n = len(STATES)
        self.counts: Dict[str, List[List[float]]] = {w: [[0.0] * n for _ in range(n)] for w in WINDOWS}
        self.observations: Dict[str, int] = {w: 0 for w in WINDOWS}
        if saved:
            self._load(saved)

    def _load(self, saved: Dict[str, Any]) -> None:
        n = len(STATES)
        for window in WINDOWS:
            rows = saved.get("counts", {}).get(window)
            if (
                isinstance(rows, list)
                and len(rows) == n
                and all(isinstance(r, list) and len(r) == n for r in rows)
            ):
                self.counts[window] = [[float(v) for v in r] for r in rows]
            self.observations[window] = int(saved.get("observations", {}).get(window, 0))

    def to_dict(self) -> Dict[str, Any]:
        return {"counts": self.counts, "observations": self.observations}

    def total_observations(self) -> int:
        return sum(self.observations.values())

    def observe_transition(self, window: str, src: str, dst: str, weight: float = 1.0) -> None:
        i, j = self.index[src], self.index[dst]
        row = self.counts[window][i]
        for k in range(len(row)):
            row[k] *= self.cfg.decay
        row[j] += weight
        self.observations[window] += 1

    def matrix(self, window: str) -> List[List[float]]:
        rows: List[List[float]] = []
        for i, state in enumerate(STATES):
            prior = TRANSITION_PRIORS[window][state]
            raw = [self.counts[window][i][j] + self.cfg.prior_strength * prior[j] for j in range(len(STATES))]
            total = sum(raw)
            rows.append([v / total for v in raw])
        return rows

    def trap_probability(self, window: str, start_state: str, steps: int) -> float:
        mat = self.matrix(window)
        n = len(STATES)
        vec = [0.0] * n
        vec[self.index[start_state]] = 1.0
        for _ in range(steps):
            vec = [sum(vec[i] * mat[i][j] for i in range(n)) for j in range(n)]
        return sum(vec[self.index[s]] for s in TRAP_STATES)

    def expected_dwell_minutes(self, window: str, state: str) -> float:
        p_stay = self.matrix(window)[self.index[state]][self.index[state]]
        return self.cfg.step_minutes / max(1e-6, 1.0 - p_stay)

    def drift_from_prior(self, window: str) -> float:
        mat = self.matrix(window)
        total = 0.0
        for i, state in enumerate(STATES):
            prior = TRANSITION_PRIORS[window][state]
            total += 0.5 * sum(abs(mat[i][j] - prior[j]) for j in range(len(STATES)))
        return total / len(STATES)


# ---------------------------------------------------------------------------
# Local-First Encrypted State Store
# ---------------------------------------------------------------------------
class StateStore:
    VERSION = 1

    def __init__(self, path: Path, max_ledger: int, max_interventions: int, max_nights: int) -> None:
        self.path = Path(path)
        self.max_ledger = max_ledger
        self.max_interventions = max_interventions
        self.max_nights = max_nights
        self.data: Dict[str, Any] = self._default()
        self.load()

    @classmethod
    def _default(cls) -> Dict[str, Any]:
        return {
            "version": cls.VERSION,
            "markov": None,
            "engine": {
                "tier": "GREEN",
                "deescalate_counter": 0,
                "last_state": None,
                "last_obs_ts": None,
                "last_intervention_ts": None,
                "last_intervention_tier": None,
                "pending_followup": None,
                "last_outcome": None,
                "session_peak": 0,
                "session_seq": 0,
            },
            "nights": [],
            "interventions": [],
            "skill_log": [],
            "privacy_ledger": [],
        }

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict) or loaded.get("version") != self.VERSION:
                raise ValueError("Incompatible state version")
        except (json.JSONDecodeError, ValueError, OSError):
            quarantine = self.path.with_name(f"{self.path.name}.corrupted-{int(time.time())}")
            try:
                os.replace(self.path, quarantine)
            except OSError:
                pass
            return
        defaults = self._default()
        for key in defaults:
            if key == "engine" and isinstance(loaded.get("engine"), dict):
                defaults["engine"].update(loaded["engine"])
            elif key in loaded:
                defaults[key] = loaded[key]
        self.data = defaults

    def save(self) -> None:
        self.data["privacy_ledger"] = self.data["privacy_ledger"][-self.max_ledger:]
        self.data["interventions"] = self.data["interventions"][-self.max_interventions:]
        self.data["skill_log"] = self.data["skill_log"][-self.max_interventions:]
        self.data["nights"] = self.data["nights"][-self.max_nights:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), prefix=".state-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def purge(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self.data = self._default()

    def log_cloud_call(
        self,
        payload: Dict[str, Any],
        model: str,
        status: str,
        latency_ms: int,
        usage: Dict[str, int],
    ) -> None:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.data["privacy_ledger"].append(
            {
                "sent_at": datetime.now().isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "model": model,
                "status": status,
                "latency_ms": latency_ms,
                "usage": usage,
                "payload": payload,
            }
        )

    def log_skill(self, record: Dict[str, Any]) -> None:
        self.data["skill_log"].append(record)


# ---------------------------------------------------------------------------
# Behavioral Engine & Assessment Model
# ---------------------------------------------------------------------------
@dataclass
class RiskAssessment:
    timestamp: datetime
    room: str
    state: str
    window: str
    idle_minutes: float
    isolation_minutes: float
    confidence: float
    components: Dict[str, float]
    score: float
    raw_tier: Tier
    previous_tier: Tier
    tier: Tier
    trap_probability: float
    relapse_streak: int
    relapse_multiplier: float
    dampener: float
    flags: List[str] = field(default_factory=list)
    actionable: bool = True

    @property
    def tier_changed(self) -> bool:
        return self.tier != self.previous_tier


class BehaviorEngine:
    """Core state engine that assesses risk, manages hysteresis, and handles interventions."""

    def __init__(self, config: MasterpieceConfig, store: StateStore) -> None:
        self.config = config
        self.cfg = config.thresholds
        self.mcfg = config.markov
        self.store = store
        self.markov = HabitMarkov(self.mcfg, store.data.get("markov"))

    @property
    def _eng(self) -> Dict[str, Any]:
        return self.store.data["engine"]

    def is_cold_start(self) -> bool:
        return self.markov.total_observations() == 0 and not self.store.data["nights"]

    def begin_session(self) -> None:
        self._eng.update(
            {
                "tier": "GREEN",
                "deescalate_counter": 0,
                "last_state": None,
                "last_obs_ts": None,
                "last_intervention_ts": None,
                "last_intervention_tier": None,
                "pending_followup": None,
                "last_outcome": None,
                "session_peak": 0,
                "session_seq": int(self._eng.get("session_seq", 0)) + 1,
            }
        )

    def persist(self) -> None:
        self.store.data["markov"] = self.markov.to_dict()
        self.store.save()

    def relapse_streak(self, before_night: str) -> int:
        nights = [n for n in self.store.data["nights"] if n.get("night", "") < before_night]
        streak = 0
        for night in sorted(nights, key=lambda n: n["night"], reverse=True):
            if night.get("relapsed"):
                streak += 1
            else:
                break
        return streak

    def relapse_multiplier(self, before_night: str) -> float:
        streak = self.relapse_streak(before_night)
        return min(self.cfg.relapse_multiplier_cap, 1.0 + self.cfg.relapse_multiplier_per_night * streak)

    def observe(self, obs: FusedObservation) -> bool:
        if obs.blind or obs.confidence < self.mcfg.learn_min_confidence:
            return False
        last_ts = _parse_dt(self._eng.get("last_obs_ts"))
        last_state = self._eng.get("last_state")
        if last_ts is None or last_state is None:
            self._eng["last_obs_ts"] = obs.timestamp.isoformat()
            self._eng["last_state"] = obs.state
            return False
        elapsed = (obs.timestamp - last_ts).total_seconds() / 60.0
        step = self.mcfg.step_minutes
        if elapsed < step - 0.5:
            return False
        recorded = False
        if elapsed <= step * self.mcfg.max_gap_steps:
            window = window_for(last_ts, self.cfg)
            self.markov.observe_transition(window, last_state, obs.state)
            recorded = True
        self._eng["last_obs_ts"] = obs.timestamp.isoformat()
        self._eng["last_state"] = obs.state
        return recorded

    def _tier_for_score(self, score: float) -> Tier:
        thresholds = self.cfg.tier_thresholds
        if score >= thresholds["RED"]:
            return Tier.RED
        if score >= thresholds["ORANGE"]:
            return Tier.ORANGE
        if score >= thresholds["YELLOW"]:
            return Tier.YELLOW
        return Tier.GREEN

    def assess(self, obs: FusedObservation) -> RiskAssessment:
        cfg = self.cfg
        window = window_for(obs.timestamp, cfg)
        trap_p = self.markov.trap_probability(window, obs.state, self.mcfg.horizon_steps)
        idle_factor = 1.0 if obs.room in ISOLATION_ROOMS else 0.25
        isolation_risk = (
            0.0
            if obs.isolation_minutes <= 0
            else _sigmoid((obs.isolation_minutes - cfg.isolation_limit_minutes) / cfg.isolation_scale_minutes)
        )
        components = {
            "location": cfg.location_risk.get(obs.room, 0.30),
            "time": time_risk(obs.timestamp, cfg),
            "idle": idle_factor * _sigmoid((obs.idle_minutes - cfg.idle_midpoint_minutes) / cfg.idle_scale_minutes),
            "isolation": isolation_risk,
            "markov": trap_p,
        }
        base = sum(cfg.weights[name] * components[name] for name in cfg.weights)
        dampener = cfg.engagement_dampener if obs.state == "DESK_ENGAGED" else 1.0
        key = night_key(obs.timestamp)
        streak = self.relapse_streak(key)
        multiplier = self.relapse_multiplier(key)
        score = max(0.0, min(100.0, 100.0 * base * dampener * multiplier))

        flags = list(obs.flags)
        raw_tier = self._tier_for_score(score)
        if obs.confidence < cfg.min_action_confidence and raw_tier >= Tier.ORANGE:
            raw_tier = Tier.YELLOW
            flags.append("TIER_CAPPED_LOW_CONFIDENCE")

        current = Tier[self._eng["tier"]]
        counter = int(self._eng.get("deescalate_counter", 0))
        if obs.blind:
            new_tier = current
            flags.append("HOLD_SENSOR_BLIND")
        elif raw_tier > current:
            new_tier = raw_tier
            counter = 0
        elif raw_tier < current:
            counter += 1
            if counter >= cfg.deescalate_confirmations:
                new_tier = Tier(int(current) - 1)
                counter = 0
            else:
                new_tier = current
                flags.append("TIER_HOLD_HYSTERESIS")
        else:
            new_tier = current
            counter = 0

        self._eng["tier"] = new_tier.name
        self._eng["deescalate_counter"] = counter
        if not obs.blind:
            self._eng["session_peak"] = max(int(self._eng.get("session_peak", 0)), int(new_tier))

        return RiskAssessment(
            timestamp=obs.timestamp,
            room=obs.room,
            state=obs.state,
            window=window,
            idle_minutes=obs.idle_minutes,
            isolation_minutes=obs.isolation_minutes,
            confidence=obs.confidence,
            components=components,
            score=score,
            raw_tier=raw_tier,
            previous_tier=current,
            tier=new_tier,
            trap_probability=trap_p,
            relapse_streak=streak,
            relapse_multiplier=multiplier,
            dampener=dampener,
            flags=flags,
            actionable=(not obs.blind) and obs.confidence >= cfg.min_action_confidence,
        )

    def _session_night_interventions(self, key: str) -> int:
        seq = self._eng.get("session_seq")
        return sum(
            1
            for item in self.store.data["interventions"]
            if item.get("session") == seq and item.get("night") == key
        )

    def should_intervene(self, a: RiskAssessment) -> Tuple[bool, str]:
        cfg = self.cfg
        if not a.actionable:
            return False, "Sensors unreliable - intervention gated"
        if a.tier < Tier.ORANGE:
            return False, "Risk tier below ORANGE activation threshold"
        if a.idle_minutes < cfg.idle_intervention_minutes:
            return False, f"Patience gate: idle {a.idle_minutes:.0f}m < {cfg.idle_intervention_minutes:.0f}m"
        if self._eng.get("pending_followup"):
            return False, "Awaiting follow-up resolution on active intervention"
        key = night_key(a.timestamp)
        if self._session_night_interventions(key) >= cfg.max_interventions_per_night:
            return False, "Nightly anti-nag intervention cap reached"
        last_ts = _parse_dt(self._eng.get("last_intervention_ts"))
        last_tier_name = self._eng.get("last_intervention_tier")
        if last_ts is not None and last_tier_name:
            elapsed = (a.timestamp - last_ts).total_seconds() / 60.0
            cooldown = cfg.cooldown_minutes.get(a.tier.name, 30)
            if elapsed < cooldown and a.tier <= Tier[last_tier_name]:
                return False, f"Cooldown active: {elapsed:.0f}m < {cooldown}m"
        return True, f"{a.tier.name} tier with sustained drift ({a.idle_minutes:.0f}m idle)"

    def record_intervention(self, a: RiskAssessment, fields: Dict[str, Any], source: str, model: str) -> str:
        seq = int(self._eng.get("session_seq", 0))
        count = sum(1 for i in self.store.data["interventions"] if i.get("session") == seq)
        intervention_id = f"{seq}-{count + 1}"
        due = a.timestamp + timedelta(minutes=int(fields["follow_up_minutes"]))
        self.store.data["interventions"].append(
            {
                "id": intervention_id,
                "session": seq,
                "night": night_key(a.timestamp),
                "timestamp": a.timestamp.isoformat(),
                "tier": a.tier.name,
                "score": round(a.score, 1),
                "skill": fields["skill"],
                "source": source,
                "model": model,
                "follow_up_minutes": int(fields["follow_up_minutes"]),
                "outcome": None,
            }
        )
        self._eng["last_intervention_ts"] = a.timestamp.isoformat()
        self._eng["last_intervention_tier"] = a.tier.name
        self._eng["pending_followup"] = {
            "id": intervention_id,
            "issued_at": a.timestamp.isoformat(),
            "due_at": due.isoformat(),
            "skill": fields["skill"],
        }
        return intervention_id

    def evaluate_followup(self, obs: FusedObservation) -> Optional[Dict[str, Any]]:
        pending = self._eng.get("pending_followup")
        if not pending or obs.blind:
            return None
        due = _parse_dt(pending["due_at"])
        trusted = obs.confidence >= self.cfg.min_action_confidence
        outcome: Optional[str] = None
        if trusted and obs.state in ("COMMON_AREA", "AWAY"):
            outcome = "COMPLIED"
        elif trusted and obs.state == "DESK_ENGAGED":
            outcome = "REENGAGED"
        elif due is not None and obs.timestamp >= due:
            outcome = "RELAPSED"
        if outcome is None:
            return None
        for item in self.store.data["interventions"]:
            if item.get("id") == pending["id"]:
                item["outcome"] = outcome
        self._eng["pending_followup"] = None
        self._eng["last_outcome"] = outcome
        return {"outcome": outcome, "skill": pending["skill"], "issued_at": pending["issued_at"]}

    def close_night(self, ts: datetime) -> Dict[str, Any]:
        key = night_key(ts)
        seq = self._eng.get("session_seq")
        mine = [i for i in self.store.data["interventions"] if i.get("session") == seq]
        relapses = sum(1 for i in mine if i.get("outcome") == "RELAPSED")
        record = {
            "night": key,
            "peak_tier": Tier(int(self._eng.get("session_peak", 0))).name,
            "interventions": len(mine),
            "relapsed": relapses > 0,
            "unresolved": 1 if self._eng.get("pending_followup") else 0,
        }
        nights = [n for n in self.store.data["nights"] if n.get("night") != key]
        nights.append(record)
        nights.sort(key=lambda n: n["night"])
        self.store.data["nights"] = nights
        return record

    def build_cloud_payload(self, a: RiskAssessment, allowed_skills: Sequence[str]) -> Dict[str, Any]:
        return {
            "schema": "masterpiece.cloud_payload.v1",
            "local_time": a.timestamp.strftime("%H:%M"),
            "window": a.window,
            "room": a.room,
            "behavioral_state": a.state,
            "idle_minutes": int(round(a.idle_minutes)),
            "isolation_minutes": int(round(a.isolation_minutes / 5.0) * 5),
            "risk_score": round(a.score, 1),
            "risk_tier": a.tier.name,
            "trap_probability": round(a.trap_probability, 2),
            "trap_horizon_minutes": self.mcfg.horizon_steps * self.mcfg.step_minutes,
            "risk_components": {name: round(value, 2) for name, value in a.components.items()},
            "relapse_streak_nights": a.relapse_streak,
            "previous_intervention_outcome": self._eng.get("last_outcome") or "NONE",
            "sensor_confidence": round(a.confidence, 2),
            "sensor_flags": list(a.flags),
            "allowed_skills": list(allowed_skills),
        }
