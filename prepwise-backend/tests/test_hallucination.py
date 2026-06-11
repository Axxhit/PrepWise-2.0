import pytest
import time


class TestHallucination:

    def test_fake_company_not_fabricated(self, generate):
        """Agent should not invent experience at a company not in resume."""
        time.sleep(13)
        data = generate("experience at Google DeepMind")
        text = data["with_rag"].lower()

        # should say not enough info, not fabricate answers
        assert any(phrase in text for phrase in [
            "not enough information",
            "no information",
            "not mentioned",
            "not found",
            "does not mention",
            "no experience",
            "cannot find",
            "isn't mentioned",
            "is not mentioned",
        ]), f"Model may have hallucinated. Got: {text[:300]}"

    def test_fake_skill_not_fabricated(self, generate):
        """Resume has no Kubernetes — agent should not invent it."""
        time.sleep(13)
        data = generate("Kubernetes and Docker container orchestration experience")
        text = data["with_rag"].lower()

        assert any(phrase in text for phrase in [
            "not enough information",
            "no information",
            "not mentioned",
            "not found",
            "does not mention",
            "no experience",
            "cannot find",
        ]), f"Model may have hallucinated Kubernetes experience. Got: {text[:300]}"

    def test_fake_degree_not_fabricated(self, generate):
        """Resume has B.Tech, not PhD — should not invent PhD."""
        time.sleep(13)
        data = generate("PhD research experience and publications")
        text = data["with_rag"].lower()

        assert any(phrase in text for phrase in [
            "not enough information",
            "no information",
            "not mentioned",
            "phd is not",
            "no phd",
            "does not mention",
            "cannot find",
        ]), f"Model may have fabricated PhD. Got: {text[:300]}"

    def test_real_skill_is_grounded(self, generate):
        """Positive control — Python IS in resume, should generate questions."""
        time.sleep(13)
        data = generate("Python programming")
        text = data["with_rag"].lower()

        # should NOT say not enough information for a real skill
        assert "not enough information" not in text
        assert len(text) > 100  # should generate actual questions

    def test_chunks_used_are_relevant(self, generate):
        """Chunks returned should contain resume content, not empty."""
        time.sleep(13)
        data = generate("machine learning")
        chunks = data["chunks_used"]

        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        assert any(kw in combined for kw in [
            "random forest", "bari", "accuracy", "android", "python", "ml"
        ]), f"Chunks don't seem relevant. Got: {combined[:300]}"