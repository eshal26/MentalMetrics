import asyncio
import json
import os
import tempfile
import threading
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from config import API_PREFIX
from db import get_subject_history
from job_runner import run_job
from job_store import create_job, require_job, save_job
from pdf_service import build_pdf_response


router = APIRouter(prefix=API_PREFIX)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    subject_id: str = Form(...),
):
    if not file.filename.endswith(".edf"):
        raise HTTPException(400, "Only .edf files are supported")

    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp()
    edf_path = os.path.join(tmp_dir, f"{job_id}.edf")

    contents = await file.read()
    with open(edf_path, "wb") as handle:
        handle.write(contents)

    save_job(job_id, create_job(subject_id, edf_path))
    start_job_thread(job_id, edf_path, subject_id)
    return {"job_id": job_id}


def start_job_thread(job_id: str, edf_path: str, subject_id: str) -> None:
    thread = threading.Thread(target=_run_job, args=(job_id, edf_path, subject_id))
    thread.daemon = True
    thread.start()


def _run_job(job_id: str, edf_path: str, subject_id: str):
    job = require_job(job_id)
    run_job(job_id, job, edf_path, subject_id)


@router.get("/stream/{job_id}")
async def stream(job_id: str):
    require_job(job_id)

    async def event_generator():
        last_log_idx = 0
        while True:
            job = require_job(job_id)
            logs = job["logs"]

            for log in logs[last_log_idx:]:
                yield f"data: {json.dumps(log)}\n\n"
                last_log_idx += 1

            if job["status"] == "done":
                payload = json.dumps(
                    {
                        "progress": 100,
                        "message": "done",
                        "result": job["result"],
                        "report": job["report"],
                    }
                )
                yield f"data: {payload}\n\n"
                break

            if job["status"] == "error":
                payload = json.dumps(
                    {
                        "progress": job["progress"],
                        "message": f"error: {job['error']}",
                    }
                )
                yield f"data: {payload}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/results/{job_id}")
def get_results(job_id: str):
    job = require_job(job_id)
    if job["status"] != "done":
        raise HTTPException(400, f"Job status: {job['status']}")
    return {"result": job["result"], "report": job["report"]}


@router.get("/history/{subject_id}")
def subject_history(subject_id: str, limit: int = 20):
    if limit < 1 or limit > 100:
        raise HTTPException(400, "limit must be between 1 and 100")
    analyses = get_subject_history(subject_id, limit=limit)
    return {
        "subject_id": subject_id,
        "count": len(analyses),
        "analyses": analyses,
    }


@router.get("/pdf/{job_id}")
def download_pdf(job_id: str):
    job = require_job(job_id)
    if job["status"] != "done" or not job["result"]:
        raise HTTPException(400, "Analysis not complete")
    return build_pdf_response(job["result"], job["report"])
