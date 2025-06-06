"""Export data from Humanloop to a local directory.

This script fetches Files and Versions, Logs, and Evaluations from Humanloop.
The data is saved to an export directory (with a timestamp) every time the script is run.

Prerequisites:
- Python environment. We recommend using `uv`.
  The script specifies the required Python version and dependencies in a uv-compatible manner below.
- .env file with HUMANLOOP_API_KEY and HUMANLOOP_API_URL
- DIRECTORY_ID environment variable set to the ID of the directory to export. If not set, the root directory is used.

Usage:
- Set the DIRECTORY_ID environment variable to the ID of the directory to export.
- Run the script with `uv run export.py`.
- The data will be saved to an export directory (with a timestamp) every time the script is run.

Output:
- The script creates an `exports` directory in the current working directory (if one doesn't already exist).
- The script creates a new subdirectory within the `exports` directory, named with a timestamp. This new export directory will contain the exported data.
- The export directory contains further subdirectories, one for each File.
- Each File directory (e.g. `exports/20250605-120000/files/pr_.../`) contains:
    - `file.json` - the File metadata
    - `deployments.json` - A mapping indicating which Versions are deployed to which Environments.
    - `versions/` - a subdirectory containing Version files. Each file is a JSON representation of a Version.
    - `logs/` - a subdirectory containing JSON files with Logs. Each JSONL file contains 1000 Logs; each line contains a single Log.
    - `evaluations/` - a subdirectory containing Evaluation directories. Each Evaluation directory (`evaluations/evr_.../`) contains:
        - `evaluation.json` - the Evaluation metadata
        - `stats.json` - the Evaluation stats (e.g. number of Logs in each Run, and aggregated metrics for each Evaluator)
        - `runs/` - a subdirectory containing Run files. Each file is a JSON representation of a Run.
        - `logs/` - a subdirectory containing JSONL files. Each line corresponds to a Log, associated with a Run ID and optional Datapoint, along with the Evaluator Logs containing judgments.
    - `datapoints/`
        - `<version_id>/ 1.jsonl...` - a subdirectory containing JSONL files. Each JSONL file contains up to 1000 datapoints. Each line corresponds to a Datapoint, associated with the Version ID.

"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "tqdm",
#   "python-dotenv",
#   "loguru",
#   "tenacity",
# ]
# ///

import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict
import time

import httpx
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
)

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

load_dotenv()

HUMANLOOP_API_KEY = os.getenv("HUMANLOOP_API_KEY")
HUMANLOOP_API_URL = os.getenv("HUMANLOOP_API_URL", "https://api.humanloop.com/v5")
if HUMANLOOP_API_KEY is None:
    raise ValueError(
        "Please set the HUMANLOOP_API_KEY environment variable to your Humanloop API key."
    )
DIRECTORY_ID = os.getenv("HUMANLOOP_DIRECTORY_ID")

# Whether to export Logs for Files and Evaluations.
# Not appropriate if you have more than 1 million Logs.
# Contact us if you have more than 1 million Logs and wish to export them.
EXPORT_LOGS = os.getenv("EXPORT_LOGS", "").lower() == "true"

# Longer timeout for requests that may take longer, e.g. fetching Logs.
# httpx otherwise defaults to 5s.
LONG_REQUEST_TIMEOUT = 300


def main(
    directory_id: str, export_dir: Path | None = None, max_workers: int | None = 8
):
    start_time = time.time()
    successful_exports = 0
    failed_exports = 0
    try:
        export_dir = export_dir or get_export_dir(get_exports_dir())
        logger.info(f"Exporting to {export_dir}")
        log_path = export_dir / "export.log"
        logger.add(log_path)
        logger.info(f"Logging to {log_path}")

        logger.info(f"EXPORT_LOGS: {EXPORT_LOGS}")

        logger.info("Fetching Files...")
        files = get_files_in_directory(directory_id)
        if not files:
            logger.info(f"No files found in Directory '{directory_id}'.")
            return
        logger.info(f"Exporting {len(files)} files from Directory '{directory_id}'.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            for file in files:
                file_dir = get_file_dir(export_dir=export_dir, file_id=file["id"])
                done_file = file_dir / ".done"
                if done_file.exists():
                    logger.info(
                        f"Skipping export for File '{file['id']}' as it has already been exported."
                    )
                    continue

                future = executor.submit(
                    export_file_to_disk,
                    file_dir=file_dir,
                    file=file,
                    done_file=done_file,
                )
                future_to_file[future] = file

            # Process results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file)):
                file = future_to_file[future]
                try:
                    future.result()
                    successful_exports += 1
                except Exception as e:
                    logger.error(f"File {file['id']} generated an exception: {str(e)}")
                    failed_exports += 1
    finally:
        end_time = time.time()
        logger.info(f"Export completed in {end_time - start_time:.1f} seconds.")
        logger.info(
            f"{successful_exports} Files successfully exported; {failed_exports} Files failed."
        )


@logger.catch
def export_file_to_disk(file_dir: Path, file: "File", done_file: Path):
    write_file_to_disk(
        file_dir=file_dir,
        file=file,
    )

    export_versions_and_deployments(
        file_id=file["id"], file_dir=file_dir, file_type=file["type"]
    )

    if file["type"] in [
        "prompt",
        "tool",
        # "dataset",
        "flow",
        "agent",
        "evaluator",
    ]:
        export_evals(file_id=file["id"], file_dir=file_dir, file_type=file["type"])

        if EXPORT_LOGS:
            export_logs(
                file_id=file["id"],
                file_dir=file_dir,
            )

    # Write .done file to indicate export completion
    with done_file.open("w") as f:
        f.write("Export completed successfully.\n")


def write_file_to_disk(file_dir: Path, file: "File"):
    with (file_dir / "file.json").open("w") as f:
        f.write(json.dumps(file))


def export_versions_and_deployments(file_id: str, file_dir: Path, file_type: str):
    """Export a File from Humanloop by its ID."""
    response = request(f"/{file_type}s/{file_id}/versions")
    versions = response["records"]
    if not versions:
        logger.info(f"No versions found for {file_type.capitalize()} '{file_id}'.")
        return
    logger.info(
        f"Exporting {file_type.capitalize()} '{file_id}' with {len(versions)} versions."
    )
    write_versions_to_disk(versions=versions, file_dir=file_dir)

    if file_type == "dataset":
        # Fetch datapoints for the Dataset versions
        datapoints_dir = file_dir / "datapoints"
        datapoints_dir.mkdir(parents=True, exist_ok=True)
        for version in versions:
            version_id = version["version_id"]
            version_datapoints_dir = datapoints_dir / version_id
            version_datapoints_dir.mkdir(parents=True, exist_ok=True)
            datapoints_count = write_pages_to_jsonl(
                dir=version_datapoints_dir,
                url=f"/{file_type}s/{file_id}/datapoints",
                params={
                    "version_id": version_id,
                    "size": 1000,
                },
            )
            logger.info(
                f"Exported {datapoints_count} datapoints for version '{version_id}' of {file_type.capitalize()} '{file_id}'."
            )

    deployments = []
    for version in versions:
        environments = version["environments"]
        for environment in environments:
            deployments.append(
                {
                    "version_id": version["version_id"],
                    "environment_id": environment["id"],
                    "environment_name": environment["name"],
                    "environment_tag": environment["tag"],
                }
            )
    with (file_dir / "deployments.json").open("w") as f:
        f.write(json.dumps(deployments))


def write_versions_to_disk(versions: list[dict], file_dir: Path):
    # path = versions[0]["path"]
    versions_dir = file_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    for version in versions:
        version_id = version["version_id"]
        with (versions_dir / version_id).open("w") as f:
            f.write(json.dumps(version))


def export_evals(file_id: str, file_dir: Path, file_type: str):
    """Export Evaluations for a File from Humanloop by its ID."""
    evaluations = list(paginated_request(f"/evaluations", params={"file_id": file_id}))
    if not evaluations:
        logger.info(f"No Evaluations found for {file_type.capitalize()} '{file_id}'.")
        return
    logger.info(
        f"Exporting {len(evaluations)} Evaluations for {file_type.capitalize()} '{file_id}'."
    )

    evaluations_dir = file_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    for evaluation in evaluations:
        evaluation_id = evaluation["id"]
        evaluation_dir = evaluations_dir / evaluation_id
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        # Write Evaluation metadata to disk
        with (evaluation_dir / "evaluation.json").open("w") as f:
            f.write(json.dumps(evaluation))

        stats = request(f"/evaluations/{evaluation_id}/stats")
        with (evaluation_dir / "stats.json").open("w") as f:
            f.write(json.dumps(stats))

        runs = request(f"/evaluations/{evaluation['id']}/runs")["runs"]
        runs_dir = evaluation_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            # Write each Run to disk
            run_id = run["id"]
            with (runs_dir / run_id).open("w") as f:
                f.write(json.dumps(run))

        if EXPORT_LOGS:
            export_evaluation_logs(
                evaluation_id=evaluation_id,
                evaluation_dir=evaluation_dir,
            )
            # export_evaluation_datapoints(
            #     evaluation_id=evaluation_id,
            #     evaluation_dir=evaluation_dir,
            # )


def write_pages_to_jsonl(
    *,
    dir: Path,
    url: str,
    records_per_file: int = 1000,
    **kwargs,
):
    """Write paginated request results to JSONL files.

    Writes records to JSONL files in the specified directory.
    """
    written_records_count = 0
    files = set()

    i = 1
    file = dir / f"{i}.jsonl"
    open_file = None
    for record in paginated_request(
        url=url,
        **kwargs,
    ):
        if open_file is None:
            open_file = file.open("w")
            files.add(file)
        open_file.write(json.dumps(record) + "\n")

        if written_records_count % records_per_file == 0 and written_records_count > 0:
            i += 1
            open_file.close()
            file = dir / f"{i}.jsonl"
            open_file = None
        written_records_count += 1

    if open_file:
        open_file.close()

    logger.trace(
        f"Wrote {written_records_count} JSONL records to {len(files)} files in {dir}."
    )
    return written_records_count


def export_evaluation_logs(evaluation_id: str, evaluation_dir: Path):
    """Export Evaluation Logs for an Evaluation ID."""
    logger.info(f"Exporting Logs for Evaluation '{evaluation_dir}'.")

    evaluation_logs_dir = evaluation_dir / "logs"
    evaluation_logs_dir.mkdir(parents=True, exist_ok=True)
    exported_logs_count = write_pages_to_jsonl(
        dir=evaluation_logs_dir,
        records_per_file=1000,
        url=f"/evaluations/{evaluation_id}/logs",
        params={
            "size": 100,
        },
        timeout=LONG_REQUEST_TIMEOUT,
    )

    logger.info(
        f"Exported {exported_logs_count} Logs for Evaluation '{evaluation_id}'."
    )


def export_evaluation_datapoints(evaluation_id: str, evaluation_dir: Path):
    """Export Evaluation Logs for an Evaluation ID."""
    logger.info(f"Exporting Logs for Evaluation '{evaluation_dir}'.")

    evaluation_logs_dir = evaluation_dir / "datapoints"
    evaluation_logs_dir.mkdir(parents=True, exist_ok=True)
    exported_datapoints_count = write_pages_to_jsonl(
        dir=evaluation_logs_dir,
        records_per_file=1000,
        url=f"/evaluations/{evaluation_id}/datapoints",
        params={
            "size": 100,
        },
        timeout=LONG_REQUEST_TIMEOUT,
    )

    logger.info(
        f"Exported {exported_datapoints_count} datapoints for Evaluation '{evaluation_id}'."
    )


def export_logs(file_id: str, file_dir: Path):
    """Export Logs for a File from Humanloop by its ID."""
    logger.info(f"Exporting Logs for File '{file_id}'.")

    logs_dir = file_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    exported_logs_count = write_pages_to_jsonl(
        dir=logs_dir,
        records_per_file=1000,
        url=f"/logs",
        params={
            "file_id": file_id,
            "size": 100,
            "include_trace_children": True,
        },
        timeout=LONG_REQUEST_TIMEOUT,
    )

    logger.info(f"Exported {exported_logs_count} Logs for File '{file_id}'.")


class File(TypedDict):
    id: str
    type: str
    directory_id: str


def get_files_in_directory(directory_id: str) -> list[File]:
    """Get all Files within a Directory from Humanloop.

    Returns a list of File IDs for Files within the specified Directory.
    Includes Files in subdirectories.
    """

    directory = request(f"/directories/{directory_id}")
    path = directory["path"]

    files: list[File] = []
    for file in paginated_request(
        f"/files",
        params={
            "path": path,
        },
    ):
        files.append(File(**file))
    return files


def get_directory_structure() -> dict:
    """
    Get the directory structure from Humanloop.
    """
    # TODO: Consider making this allow API Key auth - currently only allows bearer token.
    return request(f"/directories/structure")


# Request utils


def paginated_request(url: str, **kwargs) -> Generator[dict, None, None]:
    """Generator to perform a paginated request to the Humanloop API.

    Uses parallel requests to fetch multiple pages concurrently.
    Starts with a single request and increases parallelism if more pages are needed.
    """
    params = {"page": 1, "size": 100}
    params.update(kwargs.pop("params", {}))

    # Start with just one request
    response = request(url, params=params, **kwargs)
    records = response["records"]
    if not records:
        return

    yield from records

    # If we got a full page, there might be more - start parallel fetching
    if len(records) == params["size"]:
        max_parallel_pages = 10
        current_page = 2

        with ThreadPoolExecutor(max_workers=max_parallel_pages) as executor:
            while True:
                # Create futures for next batch of pages
                futures = []
                for page in range(current_page, current_page + max_parallel_pages):
                    page_params = params.copy()
                    page_params["page"] = page
                    futures.append(
                        executor.submit(request, url, params=page_params, **kwargs)
                    )

                # Process completed futures
                any_records = False
                for future in as_completed(futures):
                    try:
                        response = future.result()
                        page_records = response["records"]
                        if page_records:
                            any_records = True
                            yield from page_records
                    except Exception as e:
                        logger.error(f"Error fetching page: {str(e)}")
                        continue

                if not any_records:
                    break

                current_page += max_parallel_pages


def request_retry_log(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logger.info(
        f"Retrying {retry_state.attempt_number}/5 after {retry_state.next_action.sleep:.2f}s delay... Error: {str(exception)}"
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=0.5, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.RequestError)),
    reraise=True,
    before_sleep=request_retry_log,
)
def request(url: str, method: str = "GET", **kwargs):
    """Performs an HTTP request to the Humanloop API with auth headers.

    Includes retry logic that will:
    - Retry up to 5 times
    - Use exponential backoff starting at 0.5 seconds, doubling each retry up to 10 seconds
    - Only retry on HTTP errors or request errors
    """
    timeout = kwargs.pop("timeout", 60)
    response = httpx.request(
        method=method,
        url=f"{HUMANLOOP_API_URL}{url}",
        headers={"X-API-KEY": HUMANLOOP_API_KEY},
        timeout=timeout,  # Set a longer timeout to avoid erroring (Defaults to 5s).
        follow_redirects=True,
        **kwargs,
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        logger.info(f"Error requesting {url}: {exc}")
        logger.info(f"Response: {response.text}")
        raise exc
    return response.json()


# Directory utils
def get_exports_dir() -> Path:
    """Get the directory where exports are stored."""
    exports_dir = Path(__file__).parent / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir


def get_export_dir(exports_dir: Path) -> Path:
    """Get a directory for this export."""
    export_dir = exports_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def get_file_dir(export_dir: Path, file_id: str) -> Path:
    """Get the directory for a specific file in the export."""
    file_dir = export_dir / "files" / file_id
    file_dir.mkdir(parents=True, exist_ok=True)
    return file_dir


if __name__ == "__main__":
    if not DIRECTORY_ID:
        logger.info("DIRECTORY_ID not set, using root directory.")
        directory_id = request("/directories/root")["id"]
    else:
        directory_id = DIRECTORY_ID

    main(
        directory_id=directory_id,
        export_dir=None,  # Pass in a directory here to continue a previous export.
    )
