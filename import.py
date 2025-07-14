"""Import data from a local directory to Humanloop.

This script imports data from a prior export to Humanloop.

Prerequisites:
- Python environment. We recommend using `uv`.
  The script specifies the required Python version and dependencies in a uv-compatible manner below.
- .env file with HUMANLOOP_API_KEY and HUMANLOOP_API_URL
- EXPORT_DIRECTORY environment variable set to the path of the directory to import.

Usage:
- Set the EXPORT_DIRECTORY environment variable to the path of the directory to import.
- Run the script with `uv run import.py`.

Output:
- The script creates an `imports` directory (in the folder containing this script)
- The script creates a new subdirectory within the `imports` directory, named with a timestamp.
  This new import directory will contain logs of the import process (in `import.log`).
- Humanloop will be populated with the data from the export.

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
from collections import Counter, deque
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


import httpx
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

load_dotenv()


HUMANLOOP_API_KEY = os.getenv("HUMANLOOP_API_KEY")
HUMANLOOP_API_URL = os.getenv("HUMANLOOP_API_URL", "https://api.humanloop.com/v5")
if HUMANLOOP_API_KEY is None:
    raise ValueError(
        "Please set the HUMANLOOP_API_KEY environment variable to your Humanloop API key."
    )

EXPORT_DIRECTORY = os.getenv("EXPORT_DIRECTORY")
if EXPORT_DIRECTORY is None:
    raise ValueError(
        "Please set the EXPORT_DIRECTORY environment variable to the path of the directory to import."
    )

# Whether to import Logs.
IMPORT_LOGS = os.getenv("IMPORT_LOGS", "").lower() == "true"

MAX_TRIES_PER_FILE = 3
FILE_ATTEMPTS = Counter()

organization_id: str | None = None


def main(
    dir: Path,
):
    """Main function to import files from a directory to Humanloop."""
    imports_dir = Path(__file__).parent / "imports"
    import_dir = imports_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = import_dir / "import.log"
    logger.add(log_path)
    logger.info(f"Logging to {log_path}")

    logger.info(f"Importing files from {dir} to Humanloop...")

    if not dir.is_dir():
        raise ValueError(f"{dir} is not a valid directory.")

    global organization_id
    organization_id = get_organization_id()

    files_dir = dir / "files"
    if not files_dir.is_dir():
        raise ValueError(f"{files_dir} is not a valid directory.")
    file_dirs = deque(files_dir.iterdir())
    while file_dirs:
        file_dir = file_dirs.popleft()
        if not file_dir.is_dir():
            continue
        logger.info(f"Importing File from {file_dir}...")

        FILE_ATTEMPTS[file_dir.name] += 1
        try:
            try:
                file = import_file(file_dir)
            except FileHasDependenciesError as e:
                # If the file has dependencies, we need to requeue it for later processing
                logger.warning(
                    f"File {file_dir.name} has dependencies that need to be imported first. Requeuing."
                )
                if FILE_ATTEMPTS[file_dir.name] >= MAX_TRIES_PER_FILE:
                    logger.error(
                        f"File {file_dir.name} has been attempted {MAX_TRIES_PER_FILE} times. Skipping."
                    )
                else:
                    file_dirs.append(file_dir)

                continue

            file_type = file["type"]
            if file_type in [
                "prompt",
                "tool",
                # "dataset",
                "flow",
                "agent",
                "evaluator",
            ]:
                if IMPORT_LOGS:
                    import_logs(file_dir=file_dir, file_type=file_type)
        except Exception as e:
            logger.exception(f"Failed to import file {file_dir}: {e}")
            continue

    logger.info("Import completed.")
    logger.info(id_mappings)
    with (import_dir / "id_mappings.json").open("w") as f:
        # We do not expect the IDs of Files, Versions, or Logs to change.
        # However, we keep this mapping as an additional record of the import process.
        json.dump(id_mappings.mappings, f, indent=2)


def get_organization_id() -> str:
    """Get the organization ID from the environment variable or prompt the user."""
    response = request(
        url="/organizations/from-auth",
        version_override="v4",
    )
    return response["id"]


@lru_cache
def get_environments() -> list[dict[str, Any]]:
    response = request(
        url=f"/organizations/{organization_id}/environments",
        version_override="v4",
    )
    return response


def create_environment_if_needed(
    name: str, default: bool, old_id: str
) -> dict[str, Any]:
    """Creates the specified environment if one with the given name does not exist.

    Also ensures ID mappings are updated accordingly.
    """
    environments = get_environments()
    if default:
        # Handle a default environment by renaming the existing default environment
        for env in environments:
            if env["tag"] == "default":
                if env["name"] != name:
                    # Rename the existing default environment
                    logger.info(
                        f"Renaming existing default environment '{env['name']}' to '{name}'"
                    )
                    request(
                        url=f"/organizations/{organization_id}/environments/{env['id']}",
                        method="PATCH",
                        version_override="v4",
                        json={"name": name},
                    )
                    get_environments.cache_clear()  # Clear the cache to refresh environments
                id_mappings.add_mapping(
                    old_id=old_id,
                    new_id=env["id"],
                )
                return env
        raise ValueError(
            f"No default environment found to rename to '{name}'. Please create a new environment."
        )
    else:
        # Check if the environment already exists
        for env in environments:
            if env["name"] == name:
                id_mappings.add_mapping(
                    old_id=old_id,
                    new_id=env["id"],
                )
                return env
        logger.info(f"Creating new environment '{name}'")
        response = request(
            url=f"/organizations/{organization_id}/environments",
            method="POST",
            version_override="v4",
            json={"name": name},
        )
        get_environments.cache_clear()  # Clear the cache to refresh environments
        id_mappings.add_mapping(
            old_id=old_id,
            new_id=response["id"],
        )
        return response


def deploy_version_to_environment(
    file_id: str,
    version_id: str,
    file_type: str,
    environment_name: str,
    default_environment: bool,
    old_environment_id: str,
):
    """Deploy a file version to a specific environment."""
    environment = create_environment_if_needed(
        name=environment_name,
        default=default_environment,
        old_id=old_environment_id,
    )

    request(
        url=f"/{file_type}s/{file_id}/environments/{environment['id']}",
        method="POST",
        params={
            "version_id": version_id,
        },
    )


class UnhandledFileTypeError(Exception):
    """Exception raised for unhandled file types during import."""

    def __init__(self, file_type: str):
        super().__init__(f"Unhandled file type: {file_type}")


class FileHasDependenciesError(Exception):
    def __init__(self, file_type: str, file_path: str):
        super().__init__(
            f"File type '{file_type}' at path '{file_path}' has other linked Files that need to be imported first."
        )


def import_file(file_dir: Path) -> dict[str, Any]:
    file_json = file_dir / "file.json"
    if not file_json.is_file():
        raise ValueError(f"{file_json} is not a valid file.")
    with file_json.open() as f:
        file_data = json.load(f)

    file_type = file_data["type"]

    versions_dir = file_dir / "versions"
    version_files = list(versions_dir.iterdir())

    versions: list[dict[str, Any]] = []
    for version_file in version_files:
        if not version_file.is_file():
            continue
        with version_file.open() as f:
            version = json.load(f)
            versions.append(version)

    skipped_versions = []
    for version in versions:
        skip_version = False
        tools = version.get("tools", [])
        if tools:
            for tool in tools:
                if tool.get("type") == "file":
                    link = tool.get("link")
                    link_file_id = link.get("file_id")
                    link_version_id = link.get("version_id")
                    if link_file_id not in id_mappings.mappings:
                        # If the linked file is not yet imported, skip this version
                        skipped_versions.append(version)
                        logger.warning(
                            f"Skipping version {version['version_id']} of {file_type} {file_dir.name} "
                            f"because it has a linked file that has not been imported yet: {link_file_id}"
                        )
                        skip_version = True
                        break
                    if (
                        link_version_id is not None
                        and link_version_id not in id_mappings.mappings
                    ):
                        # If the linked version is not yet imported, skip this version
                        skipped_versions.append(version)
                        logger.warning(
                            f"Skipping version {version['version_id']} of {file_type} {file_dir.name} "
                            f"because it has a linked version that has not been imported yet: {link_version_id}"
                        )
                        skip_version = True
                        break
        if skip_version:
            continue

        # Properties that can be directly passed in from the File
        file_properties: list[str]
        # Additional properties
        additional_properties: dict[str, Any] = {}

        match file_type:
            case "prompt":
                file_properties = [
                    "model",
                    "endpoint",
                    "template",
                    "template_language",
                    "provider",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "stop",
                    "presence_penalty",
                    "frequency_penalty",
                    "other",
                    "seed",
                    "response_format",
                    "reasoning_effort",
                    "tools",
                    "attributes",
                    "version_name",
                    "version_description",
                    "description",
                    "tags",
                    "readme",
                ]
            case "agent":
                file_properties = [
                    "model",
                    "endpoint",
                    "template",
                    "template_language",
                    "provider",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "stop",
                    "presence_penalty",
                    "frequency_penalty",
                    "other",
                    "seed",
                    "response_format",
                    "reasoning_effort",
                    "tools",
                    "attributes",
                    "version_name",
                    "version_description",
                    "description",
                    "tags",
                    "readme",
                ]
            case "tool":
                file_properties = [
                    "function",
                    "source_code",
                    "setup_values",
                    "attributes",
                    "tool_type",
                    "version_name",
                    "version_description",
                ]
            case "evaluator":
                file_properties = [
                    "spec",
                    "version_name",
                    "version_description",
                ]
            case "dataset":
                file_properties = [
                    "attributes",
                    "version_name",
                    "version_description",
                ]

                datapoints = []
                datapoints_dir = file_dir / "datapoints"
                version_datapoints_dir = datapoints_dir / version["version_id"]
                for datapoints_jsonl in version_datapoints_dir.iterdir():
                    with datapoints_jsonl.open() as f:
                        for line in f.readlines():
                            try:
                                datapoint = json.loads(line)
                                datapoints.append(datapoint)
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Failed to decode JSON from {datapoints_jsonl}: {e}"
                                )

                additional_properties = {
                    "action": "set",
                    "datapoints": datapoints,
                }

            case "flow":
                file_properties = [
                    "attributes",
                    "version_name",
                    "version_description",
                ]
            case _:
                logger.warning(
                    f"Skipping file import as file type {file_type} is not supported: {file_dir.name}"
                )
                raise UnhandledFileTypeError(file_type=file_type)

        version_request_dict = {
            "path": version["path"],
            "id_if_creating": version["id"],
            "version_id_if_creating": version["version_id"],
        }
        for attr in file_properties:
            if attr in version:
                version_request_dict[attr] = version[attr]
        version_request_dict.update(additional_properties)

        response = request(
            url=f"/{file_type}s",
            method="POST",
            json=version_request_dict,
        )
        id_mappings.add_mapping(
            old_id=version["id"],
            new_id=response["id"],
        )
        id_mappings.add_mapping(
            old_id=version["version_id"],
            new_id=response["version_id"],
        )

    # Deploy created versions
    file_id = file_dir.name
    deployments_file = file_dir / "deployments.json"
    with deployments_file.open() as f:
        deployments = json.load(f)
    for deployment in deployments:
        deploy_version_to_environment(
            file_id=id_mappings.get(file_id),
            version_id=id_mappings.get(deployment["version_id"]),
            file_type=file_type,
            environment_name=deployment["environment_name"],
            default_environment=deployment["environment_tag"] == "default",
            old_environment_id=deployment["environment_id"],
        )

    logger.info(
        f"Imported {file_type.capitalize()} at path '{file_data['path']}' with {len(versions)} versions"
    )
    if skipped_versions:
        logger.info(
            f"Skipped {len(skipped_versions)} versions of {file_type} {file_dir.name} due to dependencies."
        )
        raise FileHasDependenciesError(
            file_type=file_type,
            file_path=file_dir.name,
        )
    return file_data


def import_logs(file_dir: Path, file_type: str):
    logs_dir = file_dir / "logs"
    if not logs_dir.is_dir():
        logger.warning(f"No logs found for File {file_dir.name} (at {logs_dir})")
        return

    logs = []
    for log_jsonl in logs_dir.iterdir():
        with log_jsonl.open() as f:
            for line in f.readlines():
                log_entry = json.loads(line)
                logs.append(log_entry)
    for log in logs:
        log_properties: list[str]

        additional_properties = {}
        match file_type:
            case "prompt":
                log_properties = [
                    "output_message",
                    "prompt_tokens",
                    "reasoning_tokens",
                    "output_tokens",
                    "prompt_cost",
                    "output_cost",
                    "finish_reason",
                    "messages",
                    "tool_choice",
                    "start_time",
                    "end_time",
                    "output",
                    "created_at",
                    "error",
                    "provider_latency",
                    "stdout",
                    "provider_request",
                    "provider_response",
                    "inputs",
                    "source",
                    "metadata",
                    # "source_datapoint_id",
                    # "trace_parent_id",
                    "user",
                    "environment",
                ]
            case "tool":
                log_properties = [
                    "start_time",
                    "end_time",
                    "output",
                    "created_at",
                    "error",
                    "provider_latency",
                    "stdout",
                    "provider_request",
                    "provider_response",
                    "inputs",
                    "source",
                    "metadata",
                    # "source_datapoint_id",
                    # "trace_parent_id",
                    "user",
                    "environment",
                ]
            case "evaluator":
                log_properties = [
                    "start_time",
                    "end_time",
                    "output",
                    "created_at",
                    "error",
                    "provider_latency",
                    "stdout",
                    "provider_request",
                    "provider_response",
                    "inputs",
                    "source",
                    "metadata",
                    # "source_datapoint_id",
                    # "trace_parent_id",
                    "user",
                    "environment",
                    "output_message",
                    "judgment",
                    "marked_completed",
                ]
                # additional_properties = {
                #     "parent_id": id_mappings.get(log["parent_id"]),
                # }
            case "dataset":
                raise ValueError("Datasets do not have Logs.")
            case "flow":
                log_properties = [
                    "messages",
                    "output_message",
                    "start_time",
                    "end_time",
                    "output",
                    "created_at",
                    "error",
                    "provider_latency",
                    "stdout",
                    "provider_request",
                    "provider_response",
                    "inputs",
                    "source",
                    "metadata",
                    "log_status",
                    # "source_datapoint_id",
                    # "trace_parent_id",
                    "user",
                    "environment",
                ]
            case "agent":
                log_properties = [
                    "output_message",
                    "prompt_tokens",
                    "reasoning_tokens",
                    "output_tokens",
                    "prompt_cost",
                    "output_cost",
                    "finish_reason",
                    "messages",
                    "tool_choice",
                    "start_time",
                    "end_time",
                    "output",
                    "created_at",
                    "error",
                    "provider_latency",
                    "stdout",
                    "provider_request",
                    "provider_response",
                    "inputs",
                    "source",
                    "metadata",
                    # "source_datapoint_id",
                    # "trace_parent_id",
                    "user",
                    "environment",
                ]
            case _:
                raise UnhandledFileTypeError(file_type=file_type)

        log_request = {
            "log_id": log["id"],
            "id": log[file_type]["id"],
            **additional_properties,
        }
        for prop in log_properties:
            if prop in log:
                log_request[prop] = log[prop]
            else:
                logger.warning(
                    f"Log property '{prop}' not found in log {log['id']} for {file_type} {file_dir.name}"
                )
        # Handle parent_id and session_id
        if log["parent_id"]:
            parent_id = id_mappings.get(log["parent_id"])
            if parent_id:
                log_request["parent_id"] = parent_id
        if log["session_id"]:
            session_id = id_mappings.get(log["session_id"])
            if session_id:
                log_request["session_id"] = session_id

        try:
            request(
                url=f"/{file_type}s/log",
                method="POST",
                json=log_request,
                params={
                    "version_id": log[file_type]["version_id"],
                },
            )
        except Exception as e:
            logger.exception(
                f"Failed to import log for {file_type} {file_dir.name} with ID {log['id']}: {e}"
            )
            continue


def request(
    url: str,
    method: str = "GET",
    version_override: str | None = None,
    **kwargs,
):
    """Performs an HTTP request to the Humanloop API with auth headers.

    Set `version_override` to "v4" to use the v4 API instead of v5.
    """
    timeout = kwargs.pop("timeout", 60)
    base_url = HUMANLOOP_API_URL
    if version_override:
        base_url = base_url.replace("v5", version_override)
    request_url = f"{base_url}{url}"
    response = httpx.request(
        method=method,
        url=request_url,
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


class IDMappings:
    def __init__(self):
        """Initialize the IDMappings class."""
        self.mappings = {}

    def add_mapping(self, old_id: str, new_id: str):
        """Add a mapping from old_id to new_id."""
        self.mappings[old_id] = new_id
        logger.info(f"ID '{old_id}' mapped to '{new_id}'")

    def __str__(self):
        return json.dumps(self.mappings)

    def get(self, old_id: str) -> str:
        """Get the new ID for an old ID."""
        return self.mappings.get(old_id)


id_mappings = IDMappings()


if __name__ == "__main__":
    main(Path(EXPORT_DIRECTORY))
