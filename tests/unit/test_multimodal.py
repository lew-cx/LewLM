from __future__ import annotations

from lewlm.documents.ingest.models import DocumentChunk, DocumentSourceType, IngestedDocumentSource
from lewlm.core.contracts import AudioTranscriptionResponse, AudioTranscriptionSegment
from lewlm.core.multimodal import _merge_audio_transcription_chunks, _plan_audio_transcription_chunks, _rank_retrieval_candidates


def test_audio_transcription_chunk_plan_splits_long_wav(long_sample_audio_bytes: bytes) -> None:
    chunk_plan = _plan_audio_transcription_chunks(long_sample_audio_bytes)

    assert chunk_plan.is_chunked is True
    assert chunk_plan.chunk_count == 2
    assert chunk_plan.duration_seconds == 2.0
    assert chunk_plan.chunks[0].start_seconds == 0.0
    assert chunk_plan.chunks[0].end_seconds == 1.0
    assert chunk_plan.chunks[1].start_seconds == 1.0
    assert chunk_plan.chunks[1].end_seconds == 2.0
    assert all(chunk.audio_bytes.startswith(b"RIFF") for chunk in chunk_plan.chunks)


def test_audio_transcription_chunk_merge_offsets_segment_times(long_sample_audio_bytes: bytes) -> None:
    chunk_plan = _plan_audio_transcription_chunks(long_sample_audio_bytes)
    responses = [
        AudioTranscriptionResponse(
            model_id="audio-model",
            text="chunk one",
            language="en",
            duration_seconds=1.0,
            segments=[AudioTranscriptionSegment(start_seconds=0.0, end_seconds=1.0, text="chunk one")],
        ),
        AudioTranscriptionResponse(
            model_id="audio-model",
            text="chunk two",
            language="en",
            duration_seconds=1.0,
            segments=[AudioTranscriptionSegment(start_seconds=0.0, end_seconds=1.0, text="chunk two")],
        ),
    ]

    merged = _merge_audio_transcription_chunks(
        model_id="audio-model",
        chunk_plan=chunk_plan,
        responses=responses,
    )

    assert merged.text == "chunk one\nchunk two"
    assert merged.language == "en"
    assert merged.duration_seconds == 2.0
    assert [(segment.start_seconds, segment.end_seconds) for segment in merged.segments] == [
        (0.0, 1.0),
        (1.0, 2.0),
    ]


def test_rank_retrieval_candidates_prefers_rerank_then_embedding_scores() -> None:
    source = IngestedDocumentSource(
        source_id="source-1",
        path="/tmp/source.md",
        source_type=DocumentSourceType.MARKDOWN,
        source_name="source.md",
        source_label="source.md",
    )
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            text="local model routing",
            source_id="source-1",
            section_id="section-1",
            source_label="source.md",
            section_label="source.md / Section 1",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            text="local model notes",
            source_id="source-1",
            section_id="section-2",
            source_label="source.md",
            section_label="source.md / Section 2",
        ),
        DocumentChunk(
            chunk_id="chunk-3",
            text="remote api",
            source_id="source-1",
            section_id="section-3",
            source_label="source.md",
            section_label="source.md / Section 3",
        ),
    ]

    ranked = _rank_retrieval_candidates(
        candidate_chunks=chunks,
        candidate_sources=[source],
        embedding_scores={0: 0.7, 1: 0.9, 2: 0.4},
        rerank_scores={0: 0.8, 1: 0.8, 2: 0.2},
    )

    assert [item.chunk.chunk_id for item in ranked] == ["chunk-2", "chunk-1", "chunk-3"]
    assert all(item.source is not None for item in ranked)
    assert ranked[0].score == 0.8
    assert ranked[0].embedding_score == 0.9
