"""LewLM-owned paged KV residency accounting for first-class text serving paths."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from threading import Lock
import time
from typing import Literal, Sequence

SchedulingLane = Literal["decode", "prefill"]
PressureLevel = Literal["low", "medium", "high", "overflow", "unbounded"]


@dataclass(frozen=True, slots=True)
class PagedKVReservation:
    reservation_id: str
    model_id: str
    scheduling_lane: SchedulingLane
    prompt_pages: int
    decode_pages: int
    requested_pages: int
    reused_pages: int
    new_pages: int
    evicted_pages: int
    overflow_pages: int
    resident_pages_after: int
    active_pages_after: int
    resident_decode_pages_after: int
    resident_prefill_pages_after: int
    active_decode_pages_after: int
    active_prefill_pages_after: int
    pressure_ratio: float
    pressure_level: PressureLevel


@dataclass(slots=True)
class _ResidentKVPage:
    model_id: str
    page_key: str
    scheduling_lane: SchedulingLane
    ref_count: int
    reuse_count: int
    last_touched_at: float


@dataclass(slots=True)
class _ActiveReservation:
    model_id: str
    scheduling_lane: SchedulingLane
    prompt_page_keys: tuple[str, ...]
    decode_pages: int


class PagedKVResidencyManager:
    """Track reusable paged-KV blocks, eviction, and lane-aware residency pressure."""

    def __init__(
        self,
        *,
        page_size_tokens: int,
        max_pages: int | None,
    ) -> None:
        self.page_size_tokens = max(int(page_size_tokens or 1), 1)
        self.max_pages = max(int(max_pages), 1) if max_pages is not None else None
        self._lock = Lock()
        self._resident_pages: dict[str, _ResidentKVPage] = {}
        self._active_reservations: dict[str, _ActiveReservation] = {}
        self._active_transient_decode_pages = 0
        self._active_transient_prefill_pages = 0
        self._reservation_counter = 0
        self._total_reservations = 0
        self._decode_lane_reservations = 0
        self._prefill_lane_reservations = 0
        self._total_reused_pages = 0
        self._total_new_pages = 0
        self._total_evicted_pages = 0
        self._prefill_evicted_pages = 0
        self._decode_evicted_pages = 0
        self._decode_headroom_preservation_events = 0
        self._prefill_decode_tradeoff_events = 0
        self._overflow_events = 0
        self._overflow_pages = 0
        self._high_pressure_events = 0
        self._peak_resident_pages = 0
        self._peak_total_pages = 0
        self._peak_pressure_ratio = 0.0

    def reserve(
        self,
        *,
        model_id: str,
        prompt_tokens: Sequence[int],
        max_tokens: int,
        scheduling_lane: SchedulingLane,
    ) -> PagedKVReservation:
        lane: SchedulingLane = "prefill" if scheduling_lane == "prefill" else "decode"
        prompt_page_keys = self._page_keys_for(model_id=model_id, prompt_tokens=prompt_tokens)
        decode_pages = max((max(int(max_tokens or 0), 0) + self.page_size_tokens - 1) // self.page_size_tokens, 1)
        reused_pages = 0
        new_page_keys: list[str] = []
        evicted_pages = 0
        pressure_ratio = 0.0
        pressure_level: PressureLevel = "unbounded" if self.max_pages is None else "low"
        overflow_pages = 0
        with self._lock:
            self._reservation_counter += 1
            reservation_id = f"kv-{self._reservation_counter}"
            timestamp = time.perf_counter()
            for page_key in prompt_page_keys:
                resident_page = self._resident_pages.get(page_key)
                if resident_page is None:
                    new_page_keys.append(page_key)
                    continue
                resident_page.ref_count += 1
                resident_page.reuse_count += 1
                resident_page.scheduling_lane = lane
                resident_page.last_touched_at = timestamp
                reused_pages += 1
            needed_pages = len(new_page_keys) + decode_pages
            evicted_pages = self._evict_to_fit_locked(needed_pages=needed_pages, preferred_lane=lane)
            for page_key in new_page_keys:
                self._resident_pages[page_key] = _ResidentKVPage(
                    model_id=model_id,
                    page_key=page_key,
                    scheduling_lane=lane,
                    ref_count=1,
                    reuse_count=0,
                    last_touched_at=timestamp,
                )
            if lane == "decode":
                self._active_transient_decode_pages += decode_pages
                self._decode_lane_reservations += 1
            else:
                self._active_transient_prefill_pages += decode_pages
                self._prefill_lane_reservations += 1
            self._active_reservations[reservation_id] = _ActiveReservation(
                model_id=model_id,
                scheduling_lane=lane,
                prompt_page_keys=prompt_page_keys,
                decode_pages=decode_pages,
            )
            self._total_reservations += 1
            self._total_reused_pages += reused_pages
            self._total_new_pages += len(new_page_keys)
            total_pages = self._total_pages_locked()
            if self.max_pages is not None and total_pages > self.max_pages:
                overflow_pages = total_pages - self.max_pages
                self._overflow_events += 1
                self._overflow_pages += overflow_pages
            pressure_ratio = self._pressure_ratio_locked(total_pages=total_pages)
            pressure_level = _pressure_level(pressure_ratio=pressure_ratio, max_pages=self.max_pages, overflow=overflow_pages)
            if pressure_level in {"high", "overflow"}:
                self._high_pressure_events += 1
            self._peak_resident_pages = max(self._peak_resident_pages, len(self._resident_pages))
            self._peak_total_pages = max(self._peak_total_pages, total_pages)
            self._peak_pressure_ratio = max(self._peak_pressure_ratio, pressure_ratio)
            return PagedKVReservation(
                reservation_id=reservation_id,
                model_id=model_id,
                scheduling_lane=lane,
                prompt_pages=len(prompt_page_keys),
                decode_pages=decode_pages,
                requested_pages=len(prompt_page_keys) + decode_pages,
                reused_pages=reused_pages,
                new_pages=len(new_page_keys),
                evicted_pages=evicted_pages,
                overflow_pages=overflow_pages,
                resident_pages_after=len(self._resident_pages),
                active_pages_after=self._active_page_count_locked(),
                resident_decode_pages_after=self._resident_page_count_for_lane_locked("decode"),
                resident_prefill_pages_after=self._resident_page_count_for_lane_locked("prefill"),
                active_decode_pages_after=self._active_page_count_for_lane_locked("decode"),
                active_prefill_pages_after=self._active_page_count_for_lane_locked("prefill"),
                pressure_ratio=pressure_ratio,
                pressure_level=pressure_level,
            )

    def release(self, reservation: PagedKVReservation | None) -> None:
        if reservation is None:
            return
        with self._lock:
            active = self._active_reservations.pop(reservation.reservation_id, None)
            if active is None:
                return
            if active.scheduling_lane == "decode":
                self._active_transient_decode_pages = max(0, self._active_transient_decode_pages - active.decode_pages)
            else:
                self._active_transient_prefill_pages = max(0, self._active_transient_prefill_pages - active.decode_pages)
            timestamp = time.perf_counter()
            for page_key in active.prompt_page_keys:
                resident_page = self._resident_pages.get(page_key)
                if resident_page is None:
                    continue
                resident_page.ref_count = max(0, resident_page.ref_count - 1)
                resident_page.last_touched_at = timestamp
            self._evict_to_fit_locked(needed_pages=0, preferred_lane="decode")

    def unregister_model(self, model_id: str) -> None:
        with self._lock:
            reservation_ids = [
                reservation_id
                for reservation_id, reservation in self._active_reservations.items()
                if reservation.model_id == model_id
            ]
            for reservation_id in reservation_ids:
                active = self._active_reservations.pop(reservation_id)
                if active.scheduling_lane == "decode":
                    self._active_transient_decode_pages = max(0, self._active_transient_decode_pages - active.decode_pages)
                else:
                    self._active_transient_prefill_pages = max(0, self._active_transient_prefill_pages - active.decode_pages)
            for page_key in [page_key for page_key, page in self._resident_pages.items() if page.model_id == model_id]:
                self._resident_pages.pop(page_key, None)

    def snapshot(self) -> dict[str, int | float | str]:
        with self._lock:
            total_pages = self._total_pages_locked()
            pressure_ratio = self._pressure_ratio_locked(total_pages=total_pages)
            return {
                "page_size_tokens": self.page_size_tokens,
                **({"max_pages": self.max_pages} if self.max_pages is not None else {}),
                "resident_pages": len(self._resident_pages),
                "active_pages": self._active_page_count_locked(),
                "active_decode_pages": self._active_page_count_for_lane_locked("decode"),
                "active_prefill_pages": self._active_page_count_for_lane_locked("prefill"),
                "resident_decode_pages": self._resident_page_count_for_lane_locked("decode"),
                "resident_prefill_pages": self._resident_page_count_for_lane_locked("prefill"),
                "decode_lane_reservations": self._decode_lane_reservations,
                "prefill_lane_reservations": self._prefill_lane_reservations,
                "reused_pages": self._total_reused_pages,
                "new_pages": self._total_new_pages,
                "evicted_pages": self._total_evicted_pages,
                "prefill_evicted_pages": self._prefill_evicted_pages,
                "decode_evicted_pages": self._decode_evicted_pages,
                "decode_headroom_preservation_events": self._decode_headroom_preservation_events,
                "prefill_decode_tradeoff_events": self._prefill_decode_tradeoff_events,
                "overflow_events": self._overflow_events,
                "overflow_pages": self._overflow_pages,
                "high_pressure_events": self._high_pressure_events,
                "peak_resident_pages": self._peak_resident_pages,
                "peak_total_pages": self._peak_total_pages,
                "pressure_ratio": pressure_ratio,
                "peak_pressure_ratio": round(self._peak_pressure_ratio, 4),
                "pressure_level": _pressure_level(
                    pressure_ratio=pressure_ratio,
                    max_pages=self.max_pages,
                    overflow=max(total_pages - self.max_pages, 0) if self.max_pages is not None else 0,
                ),
            }

    def _evict_to_fit_locked(self, *, needed_pages: int, preferred_lane: SchedulingLane) -> int:
        if self.max_pages is None:
            return 0
        evicted_pages = 0
        decode_evicted = 0
        prefill_evicted = 0
        target_total_pages = self._total_pages_locked() + max(needed_pages, 0)
        if target_total_pages <= self.max_pages:
            return 0
        candidates = sorted(
            (
                page
                for page in self._resident_pages.values()
                if page.ref_count <= 0
            ),
            key=lambda page: (
                0 if page.scheduling_lane == "prefill" else 1,
                page.last_touched_at,
                page.page_key,
            ),
        )
        for resident_page in candidates:
            if self._total_pages_locked() + max(needed_pages, 0) <= self.max_pages:
                break
            removed = self._resident_pages.pop(resident_page.page_key, None)
            if removed is None:
                continue
            evicted_pages += 1
            if removed.scheduling_lane == "decode":
                decode_evicted += 1
            else:
                prefill_evicted += 1
        if evicted_pages:
            self._total_evicted_pages += evicted_pages
            self._decode_evicted_pages += decode_evicted
            self._prefill_evicted_pages += prefill_evicted
            if preferred_lane == "decode" and prefill_evicted > 0:
                self._decode_headroom_preservation_events += 1
            if preferred_lane == "prefill" and decode_evicted > 0:
                self._prefill_decode_tradeoff_events += 1
        return evicted_pages

    def _page_keys_for(self, *, model_id: str, prompt_tokens: Sequence[int]) -> tuple[str, ...]:
        if not prompt_tokens:
            return ()
        page_keys: list[str] = []
        normalized_tokens = [int(token) for token in prompt_tokens]
        for offset in range(0, len(normalized_tokens), self.page_size_tokens):
            digest = hashlib.sha256()
            digest.update(model_id.encode("utf-8"))
            for token in normalized_tokens[offset : offset + self.page_size_tokens]:
                digest.update(int(token).to_bytes(8, "little", signed=True))
            page_keys.append(digest.hexdigest())
        return tuple(page_keys)

    def _total_pages_locked(self) -> int:
        return len(self._resident_pages) + self._active_transient_decode_pages + self._active_transient_prefill_pages

    def _active_page_count_locked(self) -> int:
        active_prompt_pages = sum(1 for page in self._resident_pages.values() if page.ref_count > 0)
        return active_prompt_pages + self._active_transient_decode_pages + self._active_transient_prefill_pages

    def _active_page_count_for_lane_locked(self, lane: SchedulingLane) -> int:
        active_prompt_pages = sum(
            1
            for page in self._resident_pages.values()
            if page.ref_count > 0 and page.scheduling_lane == lane
        )
        transient_pages = self._active_transient_decode_pages if lane == "decode" else self._active_transient_prefill_pages
        return active_prompt_pages + transient_pages

    def _resident_page_count_for_lane_locked(self, lane: SchedulingLane) -> int:
        return sum(1 for page in self._resident_pages.values() if page.scheduling_lane == lane)

    def _pressure_ratio_locked(self, *, total_pages: int | None = None) -> float:
        if self.max_pages is None:
            return 0.0
        normalized_total_pages = self._total_pages_locked() if total_pages is None else total_pages
        if self.max_pages <= 0:
            return 0.0
        return round(normalized_total_pages / self.max_pages, 4)


def _pressure_level(
    *,
    pressure_ratio: float,
    max_pages: int | None,
    overflow: int,
) -> PressureLevel:
    if max_pages is None:
        return "unbounded"
    if overflow > 0 or pressure_ratio > 1.0:
        return "overflow"
    if pressure_ratio >= 0.85:
        return "high"
    if pressure_ratio >= 0.6:
        return "medium"
    return "low"
