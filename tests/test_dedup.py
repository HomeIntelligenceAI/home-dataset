"""MinHash near-duplicate detection.

The dedup rate decides the model size (see R0), so this has to be right in both
directions: missing duplicates inflates the token budget, and over-merging
silently deletes real training text.
"""

from __future__ import annotations

import numpy as np
import pytest

from home_dataset.dedup import MinHasher, deduplicate_near, shingle

BASE = (
    "తెలుగు భాష భారతదేశంలోని ద్రావిడ భాషలలో ఒకటి. ఇది ఆంధ్రప్రదేశ్ "
    "మరియు తెలంగాణ రాష్ట్రాలలో అధికార భాషగా ఉంది. తెలుగు లిపి "
    "బ్రాహ్మీ లిపి నుండి ఉద్భవించింది."
)
NEAR = BASE + " ఈ పేజీ చివరిగా నవీకరించబడింది."   # same text + a footer
OTHER = (
    "కంప్యూటర్ సైన్స్ అనేది గణన మరియు సమాచార ప్రాసెసింగ్ అధ్యయనం. "
    "ఇది అల్గారిథమ్‌లు, డేటా నిర్మాణాలు మరియు ప్రోగ్రామింగ్ భాషలను కలిగి ఉంటుంది."
)


class TestShingle:
    def test_identical_text_shingles_identically(self) -> None:
        assert shingle(BASE) == shingle(BASE)

    def test_near_duplicates_share_most_shingles(self) -> None:
        a, b = shingle(BASE), shingle(NEAR)
        jaccard = len(a & b) / len(a | b)
        assert jaccard > 0.6

    def test_unrelated_text_shares_almost_none(self) -> None:
        a, b = shingle(BASE), shingle(OTHER)
        assert len(a & b) / len(a | b) < 0.05

    def test_short_documents_still_produce_a_shingle(self) -> None:
        """Documents shorter than the window must not vanish silently."""
        assert len(shingle("ఒక రెండు", n=5)) == 1


class TestMinHasher:
    def test_signature_has_the_requested_width(self) -> None:
        assert len(MinHasher(num_perm=64).signature(shingle(BASE))) == 64

    def test_signatures_are_reproducible_across_instances(self) -> None:
        """A dedup pass that differs between runs cannot gate anything."""
        a = MinHasher(seed=7).signature(shingle(BASE))
        b = MinHasher(seed=7).signature(shingle(BASE))
        assert np.array_equal(a, b)

    def test_estimated_similarity_tracks_true_jaccard(self) -> None:
        hasher = MinHasher(num_perm=256)
        a, b = shingle(BASE), shingle(NEAR)
        true = len(a & b) / len(a | b)
        estimated = hasher.similarity(hasher.signature(a), hasher.signature(b))
        assert abs(estimated - true) < 0.12

    def test_identical_input_estimates_one(self) -> None:
        hasher = MinHasher()
        sig = hasher.signature(shingle(BASE))
        assert hasher.similarity(sig, sig) == 1.0

    def test_unrelated_input_estimates_near_zero(self) -> None:
        hasher = MinHasher(num_perm=256)
        est = hasher.similarity(
            hasher.signature(shingle(BASE)), hasher.signature(shingle(OTHER))
        )
        assert est < 0.1


class TestDeduplicate:
    def test_exact_duplicates_are_removed(self) -> None:
        kept, stats = deduplicate_near([BASE, BASE, BASE, OTHER])
        assert stats.exact_duplicates == 2
        assert len(kept) == 2

    def test_near_duplicates_are_removed(self) -> None:
        kept, stats = deduplicate_near([BASE, NEAR, OTHER], threshold=0.6)
        assert stats.near_duplicates == 1
        assert len(kept) == 2

    def test_distinct_documents_are_all_kept(self) -> None:
        """Over-merging deletes real training text -- the costlier failure."""
        docs = [BASE, OTHER, BASE.replace("తెలుగు", "కన్నడ") + " వేరే విషయం " * 20]
        kept, stats = deduplicate_near(docs, threshold=0.8)
        assert stats.near_duplicates == 0
        assert len(kept) == 3

    def test_the_first_of_each_cluster_survives(self) -> None:
        kept, _ = deduplicate_near([BASE, NEAR], threshold=0.6)
        assert kept == [BASE]

    def test_removed_fraction_is_reported(self) -> None:
        _, stats = deduplicate_near([BASE, BASE, OTHER, OTHER])
        assert stats.documents_in == 4
        assert stats.removed_fraction == pytest.approx(0.5)

    def test_empty_input_is_handled(self) -> None:
        kept, stats = deduplicate_near([])
        assert kept == []
        assert stats.removed_fraction == 0.0

    def test_a_higher_threshold_removes_less(self) -> None:
        strict = deduplicate_near([BASE, NEAR, OTHER], threshold=0.95)[1]
        loose = deduplicate_near([BASE, NEAR, OTHER], threshold=0.5)[1]
        assert strict.near_duplicates <= loose.near_duplicates

    def test_results_are_deterministic(self) -> None:
        docs = [BASE, NEAR, OTHER, BASE + " extra"]
        first = deduplicate_near(docs, seed=99)[0]
        second = deduplicate_near(docs, seed=99)[0]
        assert first == second
