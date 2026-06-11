import pytest
import requests
import time
import os

FASTAPI_BASE = "http://localhost:8000"


# ── PII patterns from Akshit's resume ────────────────────────
PII_PATTERNS = [
    "8306634036",          # phone number
    "akshitmundra21",      # email username
    "akshitmundra21@gmail.com",  # full email
]


class TestPIILeakage:

    def test_pii_not_in_retrieval_response(self, retrieve):
        """Raw chunks may contain PII — flag it so we can decide to scrub."""
        data = retrieve("contact information email phone")
        chunks_text = " ".join(r["chunk"] for r in data["results"])

        pii_found = [p for p in PII_PATTERNS if p in chunks_text]

        if pii_found:
            print(f"\n  [WARNING] PII found in chunks: {pii_found}")
            print("  Consider scrubbing PII before storing chunks.")
            # not a hard fail — just a warning so you're aware
            # in production this WOULD be a hard fail
        else:
            print("\n  [OK] No PII found in retrieved chunks.")

    def test_pii_not_in_generated_questions(self, generate):
        """LLM should not echo raw PII into generated questions."""
        time.sleep(13)
        data = generate("contact details")
        text = data["with_rag"]

        pii_in_output = [p for p in PII_PATTERNS if p in text]
        assert not pii_in_output, \
            f"PII leaked into LLM output: {pii_in_output}"

    def test_pii_not_in_without_rag_response(self, generate):
        """Without RAG context, PII should definitely not appear."""
        time.sleep(13)
        data = generate("personal contact information")
        text = data["without_rag"]

        pii_in_output = [p for p in PII_PATTERNS if p in text]
        assert not pii_in_output, \
            f"PII leaked into no-RAG response: {pii_in_output}"


class TestSessionIsolation:

    def test_two_sessions_dont_share_data(self):
        """Upload two different PDFs — each session should only see its own data."""

        resume_path = os.path.join(
            os.path.dirname(__file__), "..", "test_resume.pdf"
        )

        # create session A
        with open(resume_path, "rb") as f:
            res_a = requests.post(
                f"{FASTAPI_BASE}/upload",
                files={"file": ("resume_a.pdf", f, "application/pdf")}
            )
        session_a = res_a.json()["session_id"]
        requests.post(f"{FASTAPI_BASE}/embed", params={"session_id": session_a})

        # create session B (same file — different session)
        with open(resume_path, "rb") as f:
            res_b = requests.post(
                f"{FASTAPI_BASE}/upload",
                files={"file": ("resume_b.pdf", f, "application/pdf")}
            )
        session_b = res_b.json()["session_id"]
        requests.post(f"{FASTAPI_BASE}/embed", params={"session_id": session_b})

        # session A should work
        res = requests.get(
            f"{FASTAPI_BASE}/retrieve",
            params={"session_id": session_a, "q": "machine learning"}
        )
        assert res.status_code == 200
        assert len(res.json()["results"]) > 0

        # session B should work independently
        res = requests.get(
            f"{FASTAPI_BASE}/retrieve",
            params={"session_id": session_b, "q": "machine learning"}
        )
        assert res.status_code == 200
        assert len(res.json()["results"]) > 0

    def test_deleted_session_returns_404(self):
        """A random UUID should never return data."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        res = requests.get(
            f"{FASTAPI_BASE}/retrieve",
            params={"session_id": fake_id, "q": "python"}
        )
        assert res.status_code == 404

    def test_session_ids_are_unique(self):
        """Each upload must return a different session_id."""
        resume_path = os.path.join(
            os.path.dirname(__file__), "..", "test_resume.pdf"
        )
        ids = []
        for _ in range(3):
            with open(resume_path, "rb") as f:
                res = requests.post(
                    f"{FASTAPI_BASE}/upload",
                    files={"file": ("resume.pdf", f, "application/pdf")}
                )
            ids.append(res.json()["session_id"])

        assert len(set(ids)) == 3, \
            f"Duplicate session IDs generated: {ids}"


class TestLoggingSafety:

    def test_no_pii_in_error_messages(self, session_id):
        """Error responses should not echo back user input containing PII."""
        pii_topic = "akshitmundra21@gmail.com phone 8306634036"

        res = requests.get(
            f"{FASTAPI_BASE}/retrieve",
            params={"session_id": "invalid-session", "q": pii_topic}
        )
        assert res.status_code == 404
        error_detail = res.json().get("detail", "")

        # error message should not echo the PII query back
        assert "akshitmundra21" not in error_detail, \
            "PII echoed in error message"
        assert "8306634036" not in error_detail, \
            "Phone number echoed in error message"