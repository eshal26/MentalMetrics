import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from main import app  # noqa: E402
from job_store import jobs  # noqa: E402


class ApiTests(unittest.TestCase):
    def setUp(self):
        jobs.clear()
        self.client = TestClient(app)

    def tearDown(self):
        jobs.clear()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_analyze_rejects_non_edf_upload(self):
        response = self.client.post(
            "/api/analyze",
            files={"file": ("sample.txt", b"demo", "text/plain")},
            data={"subject_id": "EEG-01"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Only .edf files are supported")

    def test_analyze_creates_job_and_starts_runner(self):
        with patch("routes.start_job_thread") as start_job_thread:
            response = self.client.post(
                "/api/analyze",
                files={"file": ("sample.edf", b"edf-bytes", "application/octet-stream")},
                data={"subject_id": "EEG-01"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("job_id", payload)

        job = jobs[payload["job_id"]]
        self.assertEqual(job["subject_id"], "EEG-01")
        self.assertEqual(job["status"], "running")
        self.assertTrue(os.path.exists(job["edf_path"]))

        start_job_thread.assert_called_once()

        os.remove(job["edf_path"])
        os.rmdir(os.path.dirname(job["edf_path"]))

    def test_results_returns_completed_job_payload(self):
        jobs["done-job"] = {
            "status": "done",
            "progress": 100,
            "logs": [],
            "result": {"subject_id": "EEG-01"},
            "report": {"summary": "ok"},
            "error": None,
            "subject_id": "EEG-01",
            "edf_path": "",
        }

        response = self.client.get("/api/results/done-job")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"result": {"subject_id": "EEG-01"}, "report": {"summary": "ok"}},
        )

    def test_pdf_uses_pdf_service_for_completed_job(self):
        jobs["done-job"] = {
            "status": "done",
            "progress": 100,
            "logs": [],
            "result": {"subject_id": "EEG-01"},
            "report": {"summary": "ok"},
            "error": None,
            "subject_id": "EEG-01",
            "edf_path": "",
        }

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.write(b"%PDF-1.4\n")
        temp_pdf.close()

        try:
            with patch("routes.build_pdf_response") as build_pdf_response:
                from fastapi.responses import FileResponse

                build_pdf_response.return_value = FileResponse(
                    temp_pdf.name,
                    media_type="application/pdf",
                    filename="report.pdf",
                )
                response = self.client.get("/api/pdf/done-job")
        finally:
            os.remove(temp_pdf.name)

        self.assertEqual(response.status_code, 200)
        build_pdf_response.assert_called_once_with(
            {"subject_id": "EEG-01"},
            {"summary": "ok"},
        )


if __name__ == "__main__":
    unittest.main()
