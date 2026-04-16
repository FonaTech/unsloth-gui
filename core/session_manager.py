"""
core/session_manager.py
Global session registry, per-session isolation, and task queue.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.monitor import TrainingMonitor
from core.trainer import TrainingOrchestrator
from core.auto_tuner import AutoTuner


@dataclass
class SessionState:
    session_id: str
    monitor: TrainingMonitor
    orchestrator: TrainingOrchestrator
    auto_tuner: AutoTuner
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    status: str = "idle"   # idle | queued | running | finished | error


class SessionManager:
    """
    Global singleton managing all browser sessions.

    Modes:
      "singleton"   — all sessions share one monitor/orchestrator/auto_tuner
      "per_session" — each session_id gets its own independent instances

    Task queue:
      When running sessions >= max_concurrent, new training requests are queued.
      When a session finishes, the next queued session is automatically started.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.mode: str = "singleton"
        self.max_concurrent: int = 1

        # Per-session registry
        self._sessions: Dict[str, SessionState] = {}
        self._queue: List[str] = []   # session_ids waiting for a slot

        # Singleton instances (used in singleton mode)
        self._singleton_monitor = TrainingMonitor()
        self._singleton_orchestrator = TrainingOrchestrator(self._singleton_monitor)
        self._singleton_auto_tuner = AutoTuner()
        self._singleton_id = "__singleton__"
        self._singleton_state = SessionState(
            session_id=self._singleton_id,
            monitor=self._singleton_monitor,
            orchestrator=self._singleton_orchestrator,
            auto_tuner=self._singleton_auto_tuner,
        )

    # ── Public API ────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self.mode = mode

    def set_max_concurrent(self, n: int) -> None:
        with self._lock:
            self.max_concurrent = max(1, int(n))

    def get_or_create(self, session_id: str) -> SessionState:
        """Return the SessionState for this session, creating if needed."""
        if self.mode == "singleton":
            self._singleton_state.last_seen = time.time()
            return self._singleton_state

        with self._lock:
            if session_id not in self._sessions:
                monitor = TrainingMonitor()
                orchestrator = TrainingOrchestrator(monitor)
                auto_tuner = AutoTuner()
                self._sessions[session_id] = SessionState(
                    session_id=session_id,
                    monitor=monitor,
                    orchestrator=orchestrator,
                    auto_tuner=auto_tuner,
                )
            state = self._sessions[session_id]
            state.last_seen = time.time()
            return state

    def request_training(self, session_id: str) -> str:
        """
        Request a training slot for this session.
        Returns "start" if slot available, "queued" if waiting.
        """
        with self._lock:
            running = self._count_running()
            if running < self.max_concurrent:
                state = self._get_state(session_id)
                state.status = "running"
                return "start"
            else:
                if session_id not in self._queue:
                    self._queue.append(session_id)
                state = self._get_state(session_id)
                state.status = "queued"
                return "queued"

    def on_training_done(self, session_id: str) -> None:
        """Called when a session's training finishes/stops. Releases slot and starts next."""
        with self._lock:
            state = self._get_state(session_id)
            if state.status == "running":
                state.status = "finished"
            self._release_gpu_locked(session_id)
            self._start_next_queued()

    def is_queued(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._queue

    def queue_position(self, session_id: str) -> int:
        """Return 1-based queue position, or 0 if not queued."""
        with self._lock:
            try:
                return self._queue.index(session_id) + 1
            except ValueError:
                return 0

    def get_stats(self) -> dict:
        """Return current session statistics."""
        with self._lock:
            if self.mode == "singleton":
                running = 1 if self._singleton_state.status == "running" else 0
                return {
                    "mode": "singleton",
                    "online": 1,
                    "running": running,
                    "queued": 0,
                    "max_concurrent": self.max_concurrent,
                }
            online = len(self._sessions)
            running = self._count_running()
            queued = len(self._queue)
            return {
                "mode": "per_session",
                "online": online,
                "running": running,
                "queued": queued,
                "max_concurrent": self.max_concurrent,
            }

    def touch(self, session_id: str) -> None:
        """Update last_seen timestamp for a session."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state:
                state.last_seen = time.time()

    def cleanup_stale(self, max_age_seconds: int = 3600) -> None:
        """Remove sessions that haven't been seen for max_age_seconds."""
        now = time.time()
        with self._lock:
            stale = [
                sid for sid, s in self._sessions.items()
                if now - s.last_seen > max_age_seconds and s.status not in ("running", "queued")
            ]
            for sid in stale:
                self._sessions.pop(sid, None)

    # ── Internal ──────────────────────────────────────────────────

    def _get_state(self, session_id: str) -> SessionState:
        """Must be called with lock held."""
        if self.mode == "singleton":
            return self._singleton_state
        if session_id not in self._sessions:
            monitor = TrainingMonitor()
            orchestrator = TrainingOrchestrator(monitor)
            auto_tuner = AutoTuner()
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                monitor=monitor,
                orchestrator=orchestrator,
                auto_tuner=auto_tuner,
            )
        return self._sessions[session_id]

    def _count_running(self) -> int:
        """Must be called with lock held."""
        if self.mode == "singleton":
            return 1 if self._singleton_state.status == "running" else 0
        return sum(1 for s in self._sessions.values() if s.status == "running")

    def _release_gpu_locked(self, session_id: str) -> None:
        """Release GPU memory for a session. Must be called with lock held."""
        state = self._get_state(session_id)
        orch = state.orchestrator
        # Delete model references to free GPU/Metal memory
        if hasattr(orch, "_model") and orch._model is not None:
            try:
                del orch._model
            except Exception:
                pass
            orch._model = None
        if hasattr(orch, "_tokenizer") and orch._tokenizer is not None:
            try:
                del orch._tokenizer
            except Exception:
                pass
            orch._tokenizer = None
        # Clear Metal cache on Apple Silicon
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()
        except Exception:
            pass
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _start_next_queued(self) -> None:
        """Start the next queued session if a slot is available. Must be called with lock held."""
        if not self._queue:
            return
        running = self._count_running()
        while running < self.max_concurrent and self._queue:
            next_id = self._queue.pop(0)
            state = self._get_state(next_id)
            state.status = "running"
            # Signal the waiting session to start
            orch = state.orchestrator
            if hasattr(orch, "_queue_event"):
                orch._queue_event.set()
            running += 1


# Global singleton instance
session_manager = SessionManager()
