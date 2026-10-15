import asyncio
import logging
import time
import uuid
from collections import deque

from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder

from app.core.config import get_settings
from app.schemas.pledge_slip import DocumentInfo, OcrResult, PledgeSlipResponse
from app.services.review_service import build_review
from app.services.slip_analyzer import analyze_document

logger = logging.getLogger(__name__)

MAX_PENDING = 200        # jobs allowed to wait in the queue
KEEP_SECONDS = 120       # finished jobs are kept 2 min, then deleted


class Job:
    def __init__(self, job_id: str, validated):
        self.id = job_id
        self.validated = validated
        self.status = "queued"          # queued | processing | done | failed
        self.result = None
        self.error = None
        self.exception = None
        self.created = time.time()
        self.finished = None
        self.event = asyncio.Event()


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.pending: deque[str] = deque()
        self.queue: asyncio.Queue | None = None
        self.workers: list[asyncio.Task] = []
        self.loop = None
        self.service = None

    def _ensure_started(self, service):
        loop = asyncio.get_running_loop()
        if self.queue is None or self.loop is not loop or self.service is not service:
            for task in self.workers:
                try:
                    task.cancel()
                except RuntimeError:
                    pass
            self.workers = []
            self.pending.clear()
            self.queue = asyncio.Queue()
            self.loop = loop
            self.service = service
            count = max(1, get_settings().ocr_max_concurrency)
            for _ in range(count):
                self.workers.append(asyncio.create_task(self._worker(service)))

    def _purge(self):
        now = time.time()
        for job_id in [j.id for j in self.jobs.values()
                       if j.finished and now - j.finished > KEEP_SECONDS]:
            del self.jobs[job_id]

    def submit(self, validated, service) -> Job | None:
        self._purge()
        if len(self.pending) >= MAX_PENDING:
            return None
        self._ensure_started(service)
        job = Job(uuid.uuid4().hex, validated)
        self.jobs[job.id] = job
        self.pending.append(job.id)
        self.queue.put_nowait(job.id)
        return job

    def position(self, job_id: str) -> int:
        try:
            return list(self.pending).index(job_id) + 1
        except ValueError:
            return 0

    async def _worker(self, service):
        settings = get_settings()
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            try:
                self.pending.remove(job_id)
            except ValueError:
                pass
            job.status = "processing"
            started = time.perf_counter()
            await service.semaphore.acquire()
            try:
                analysis = await run_in_threadpool(
                    analyze_document, job.validated, service, settings
                )
                result, extracted = analysis.pipeline, analysis.extracted
                review = build_review(
                    extracted, settings.mandatory_field_list, analysis.warnings
                )
                response = PledgeSlipResponse(
                    requestId=job.id,
                    document=DocumentInfo(
                        filename=job.validated.filename,
                        fileType=job.validated.kind,
                        pageCount=job.validated.page_count,
                        pages=result.pages,
                    ),
                    extractedData=extracted,
                    ocr=OcrResult(
                        rawText="\n".join(line.text for line in result.lines),
                        lines=result.lines,
                    ),
                    review=review,
                )
                job.result = jsonable_encoder(response)
                job.status = "done"
                logger.info("Job %s done in %.2fs", job.id, time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001
                job.exception = exc
                logger.exception("Job %s failed", job.id)
                job.status = "failed"
            finally:
                service.semaphore.release()
                job.validated = None   # free the file from memory
                job.finished = time.time()
                job.event.set()


job_manager = JobManager()