from pydoc import text

import pytest
import time


class TestQuestionGeneration:
    @pytest.mark.slow
    def test_generates_three_questions(self, generate):
        data = generate("machine learning")
        text = data["with_rag"]
    # count numbered questions more reliably
        assert text.count("1.") >= 1
        assert text.count("2.") >= 1
        assert text.count("3.") >= 1

    def test_rag_response_longer_than_no_rag(self, generate):
        data = generate("machine learning")
        # with_rag has resume context — should produce more specific output
        assert len(data["with_rag"]) > 0
        assert len(data["without_rag"]) > 0

    def test_with_rag_mentions_resume_keywords(self, generate):
        data = generate("machine learning project")
        text = data["with_rag"].lower()
        assert any(
            kw in text
            for kw in ["bari", "random forest", "accuracy", "android", "classifier"]
        )

    def test_without_rag_is_more_generic(self, generate):
        data = generate("machine learning project")
        text = data["without_rag"].lower()
        # without context it should NOT mention specific project names
        assert "bari" not in text

    def test_chunks_returned_with_response(self, generate):
        data = generate("Python")
        assert "chunks_used" in data
        assert isinstance(data["chunks_used"], list)

    @pytest.mark.slow
    def test_fullstack_questions_mention_web_tech(self, generate):
        time.sleep(13)
        data = generate("full stack development")
        text = data["with_rag"].lower()
        assert any(
            kw in text
            for kw in ["next", "firebase", "react", "node", "api", "frontend", "web", "javascript"]
        )

    def test_internship_questions_reference_experience(self, generate):
        time.sleep(13)
        data = generate("internship experience")
        text = data["with_rag"].lower()
        assert any(
            kw in text
            for kw in ["sap", "kanchan", "sales", "dashboard", "automation", "intern", "process", "kpi"]
        )

    def test_response_has_required_keys(self, generate):
        data = generate("Python")
        assert "with_rag" in data
        assert "without_rag" in data
        assert "chunks_used" in data
        assert "topic" in data

    def test_topic_echoed_in_response(self, generate):
        data = generate("transformer architecture")
        assert data["topic"] == "transformer architecture"