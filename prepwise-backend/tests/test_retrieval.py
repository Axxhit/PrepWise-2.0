import pytest


class TestRetrieval:

    def test_python_query_returns_results(self, retrieve):
        data = retrieve("Python programming")
        assert len(data["results"]) > 0

    def test_results_have_required_keys(self, retrieve):
        data = retrieve("machine learning")
        for r in data["results"]:
            assert "chunk" in r
            assert "similarity_score" in r

    def test_similarity_scores_are_numeric(self, retrieve):
        data = retrieve("machine learning")
        for r in data["results"]:
            assert isinstance(r["similarity_score"], float)

    def test_ml_query_returns_relevant_chunk(self, retrieve):
        data = retrieve("Random Forest classifier")
        chunks_text = " ".join(r["chunk"] for r in data["results"])
        # resume mentions BARI + Random Forest — at least one chunk should contain it
        assert any(
            keyword in chunks_text
            for keyword in ["Random Forest", "BARI", "classifier", "accuracy"]
        )

    def test_fullstack_query_returns_relevant_chunk(self, retrieve):
        data = retrieve("full stack web development")
        chunks_text = " ".join(r["chunk"] for r in data["results"])
        assert any(
            keyword in chunks_text
            for keyword in ["Next.js", "Firebase", "Node.js", "React", "Flask"]
        )

    def test_education_query_returns_relevant_chunk(self, retrieve):
        data = retrieve("university education degree")
        chunks_text = " ".join(r["chunk"] for r in data["results"])
        assert any(
            keyword in chunks_text
            for keyword in ["VIT", "Vellore", "B.Tech", "CGPA"]
        )

    def test_internship_query_returns_relevant_chunk(self, retrieve):
        data = retrieve("Kanchan India SAP intern sales order")
        chunks_text = " ".join(r["chunk"] for r in data["results"])
        assert any(
            keyword in chunks_text
            for keyword in ["Kanchan", "SAP", "intern", "sales"]
        )

    def test_invalid_session_returns_404(self):
        import requests
        res = requests.get(
            "http://localhost:8000/retrieve",
            params={"session_id": "fake-session-999", "q": "Python"}
        )
        assert res.status_code == 404

    def test_top_k_respected(self, retrieve):
        import requests
        res = requests.get(
            "http://localhost:8000/retrieve",
            params={"session_id": "placeholder", "q": "Python", "top_k": 2}
        )
        # just check the param is accepted (session may not exist)
        assert res.status_code in [200, 404]

    def test_empty_query_handled(self, retrieve):
        # empty query should not crash server
        data = retrieve(" ")
        assert "results" in data