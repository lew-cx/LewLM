"""Persisted response caching for deterministic runtime capabilities."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import TypeVar

from pydantic import BaseModel

from lewlm.core.contracts import (
    AudioSpeechResponse,
    AudioTranscriptionResponse,
    EmbeddingResponse,
    RerankResponse,
)
from lewlm.storage.metadata import MetadataStore


ResponseT = TypeVar("ResponseT", AudioTranscriptionResponse, EmbeddingResponse, RerankResponse)


class RuntimeResponseCache:
    """Reuse deterministic runtime responses across process restarts."""

    def __init__(self, *, metadata_store: MetadataStore) -> None:
        self.metadata_store = metadata_store

    def get_embedding_response(self, *, model_id: str, inputs: list[str]) -> EmbeddingResponse | None:
        return self._lookup_response(
            capability="embeddings",
            payload={"model_id": model_id, "inputs": inputs},
            response_type=EmbeddingResponse,
        )

    def get_embedding_response_by_cache_key(self, cache_key: str) -> EmbeddingResponse | None:
        return self._lookup_response_by_cache_key(cache_key, response_type=EmbeddingResponse)

    def put_embedding_response(self, *, model_id: str, inputs: list[str], response: EmbeddingResponse) -> None:
        self._store_response(
            capability="embeddings",
            payload={"model_id": model_id, "inputs": inputs},
            model_id=model_id,
            response=response,
        )

    def embedding_cache_key(self, *, model_id: str, inputs: list[str]) -> str:
        return self._cache_key(capability="embeddings", payload={"model_id": model_id, "inputs": inputs})

    def get_rerank_response(
        self,
        *,
        model_id: str,
        query: str,
        documents: list[str],
        top_n: int | None,
    ) -> RerankResponse | None:
        return self._lookup_response(
            capability="rerank",
            payload={"model_id": model_id, "query": query, "documents": documents, "top_n": top_n},
            response_type=RerankResponse,
        )

    def get_rerank_response_by_cache_key(self, cache_key: str) -> RerankResponse | None:
        return self._lookup_response_by_cache_key(cache_key, response_type=RerankResponse)

    def put_rerank_response(
        self,
        *,
        model_id: str,
        query: str,
        documents: list[str],
        top_n: int | None,
        response: RerankResponse,
    ) -> None:
        self._store_response(
            capability="rerank",
            payload={"model_id": model_id, "query": query, "documents": documents, "top_n": top_n},
            model_id=model_id,
            response=response,
        )

    def rerank_cache_key(
        self,
        *,
        model_id: str,
        query: str,
        documents: list[str],
        top_n: int | None,
    ) -> str:
        return self._cache_key(
            capability="rerank",
            payload={"model_id": model_id, "query": query, "documents": documents, "top_n": top_n},
        )

    def get_audio_transcription_response(
        self,
        *,
        model_id: str,
        audio_bytes: bytes,
        file_name: str,
        language: str | None,
        prompt: str | None,
    ) -> AudioTranscriptionResponse | None:
        return self._lookup_response(
            capability="audio_transcription",
            payload=self._audio_transcription_payload(
                model_id=model_id,
                audio_bytes=audio_bytes,
                file_name=file_name,
                language=language,
                prompt=prompt,
            ),
            response_type=AudioTranscriptionResponse,
        )

    def get_audio_transcription_response_by_cache_key(self, cache_key: str) -> AudioTranscriptionResponse | None:
        return self._lookup_response_by_cache_key(cache_key, response_type=AudioTranscriptionResponse)

    def put_audio_transcription_response(
        self,
        *,
        model_id: str,
        audio_bytes: bytes,
        file_name: str,
        language: str | None,
        prompt: str | None,
        response: AudioTranscriptionResponse,
    ) -> None:
        self._store_response(
            capability="audio_transcription",
            payload=self._audio_transcription_payload(
                model_id=model_id,
                audio_bytes=audio_bytes,
                file_name=file_name,
                language=language,
                prompt=prompt,
            ),
            model_id=model_id,
            response=response,
        )

    def audio_transcription_cache_key(
        self,
        *,
        model_id: str,
        audio_bytes: bytes,
        file_name: str,
        language: str | None,
        prompt: str | None,
    ) -> str:
        return self._cache_key(
            capability="audio_transcription",
            payload=self._audio_transcription_payload(
                model_id=model_id,
                audio_bytes=audio_bytes,
                file_name=file_name,
                language=language,
                prompt=prompt,
            ),
        )

    def get_audio_speech_response(
        self,
        *,
        model_id: str,
        input_text: str,
        voice: str | None,
        audio_format: str,
    ) -> AudioSpeechResponse | None:
        return self.get_audio_speech_response_by_cache_key(
            self.audio_speech_cache_key(
                model_id=model_id,
                input_text=input_text,
                voice=voice,
                audio_format=audio_format,
            ),
        )

    def get_audio_speech_response_by_cache_key(self, cache_key: str) -> AudioSpeechResponse | None:
        record = self._get_cache_record(cache_key)
        if record is None:
            return None
        payload = dict(record["response_payload"])
        audio_base64 = payload.pop("audio_base64")
        return AudioSpeechResponse.model_validate(
            {
                **payload,
                "audio_bytes": base64.b64decode(audio_base64),
            },
        )

    def put_audio_speech_response(
        self,
        *,
        model_id: str,
        input_text: str,
        voice: str | None,
        audio_format: str,
        response: AudioSpeechResponse,
    ) -> None:
        self._store_response_payload(
            cache_key=self.audio_speech_cache_key(
                model_id=model_id,
                input_text=input_text,
                voice=voice,
                audio_format=audio_format,
            ),
            capability="audio_speech",
            model_id=model_id,
            response_payload={
                "model_id": response.model_id,
                "audio_base64": base64.b64encode(response.audio_bytes).decode("ascii"),
                "media_type": response.media_type,
                "voice": response.voice,
                "duration_seconds": response.duration_seconds,
            },
        )

    def audio_speech_cache_key(
        self,
        *,
        model_id: str,
        input_text: str,
        voice: str | None,
        audio_format: str,
    ) -> str:
        return self._cache_key(
            capability="audio_speech",
            payload={
                "model_id": model_id,
                "input_text": input_text,
                "voice": voice,
                "audio_format": audio_format,
            },
        )

    def cache_stats(self) -> dict[str, int]:
        return {
            "runtime_response_count": self.metadata_store.runtime_response_cache_count(),
            "runtime_response_bytes": self.metadata_store.runtime_response_cache_size_bytes(),
            "runtime_cache_hits": self.metadata_store.get_counter("runtime_cache_hits"),
            "runtime_cache_misses": self.metadata_store.get_counter("runtime_cache_misses"),
        }

    def _lookup_response(
        self,
        *,
        capability: str,
        payload: dict[str, object],
        response_type: type[ResponseT],
    ) -> ResponseT | None:
        return self._lookup_response_by_cache_key(
            self._cache_key(capability=capability, payload=payload),
            response_type=response_type,
        )

    def _lookup_response_by_cache_key(self, cache_key: str, *, response_type: type[ResponseT]) -> ResponseT | None:
        record = self._get_cache_record(cache_key)
        if record is None:
            return None
        return response_type.model_validate(record["response_payload"])

    def _store_response(
        self,
        *,
        capability: str,
        payload: dict[str, object],
        model_id: str,
        response: BaseModel,
    ) -> None:
        self._store_response_payload(
            cache_key=self._cache_key(capability=capability, payload=payload),
            capability=capability,
            model_id=model_id,
            response_payload=response.model_dump(mode="json"),
        )

    def _store_response_payload(
        self,
        *,
        cache_key: str,
        capability: str,
        model_id: str,
        response_payload: dict[str, object],
    ) -> None:
        self.metadata_store.upsert_runtime_response_cache_entry(
            cache_key=cache_key,
            capability=capability,
            model_id=model_id,
            response_payload=response_payload,
        )

    @staticmethod
    def _cache_key(*, capability: str, payload: dict[str, object]) -> str:
        serialized = json.dumps(
            {"capability": capability, **payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256()
        digest.update(serialized.encode("utf-8"))
        return digest.hexdigest()

    def _get_cache_record(self, cache_key: str) -> dict[str, object] | None:
        record = self.metadata_store.get_runtime_response_cache_entry(cache_key)
        if record is None:
            self.metadata_store.increment_counter("runtime_cache_misses")
            return None
        self.metadata_store.increment_counter("runtime_cache_hits")
        return record

    @staticmethod
    def _audio_transcription_payload(
        *,
        model_id: str,
        audio_bytes: bytes,
        file_name: str,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, object]:
        digest = hashlib.sha256()
        digest.update(audio_bytes)
        return {
            "model_id": model_id,
            "file_name": file_name,
            "language": language,
            "prompt": prompt,
            "audio_sha256": digest.hexdigest(),
            "audio_input_bytes": len(audio_bytes),
        }
