# Humanloop Export Tool

A Python script to export data from Humanloop, including Files, Versions, Logs, and Evaluations.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Humanloop API key

The script can also be run without `uv` by installing the dependencies manually.
The dependencies required can be found in the comments at the top of the script.
If doing so, ensure you're using Python 3.10 or higher.

If you don't have a Humanloop API key, you can create one in the [Humanloop UI](https://app.humanloop.com/account/api-keys).

## Setup

Update your `.env` file with the following:

### `HUMANLOOP_API_KEY`

Set `HUMANLOOP_API_KEY` to your Humanloop API key (more details in previous section).

### `HUMANLOOP_DIRECTORY_ID` (optional)

The `HUMANLOOP_DIRECTORY_ID` variable is optional. Specify this (e.g. `dir_...`) to only
export Files within that directory (and all subdirectories). If not specified, all Files
will be exported.

### `EXPORT_LOGS` (optional)

If you have fewer than 1 million Logs, you can set the `EXPORT_LOGS` environment variable to `true` to export Logs.
By default, the script will not export Logs.

If you have more than 1 million Logs and wish to export them, contact us and we can arrange a custom export of your Logs via S3.

E.g.

```env
HUMANLOOP_API_KEY=hl_sk_4ea7b2d77f532c13af94420c0329c2c012b0cfad6dd9b8ac
HUMANLOOP_DIRECTORY_ID=dir_xvNi1CKM872aETtBU6SBF
EXPORT_LOGS=true
```

## Usage

Run the export script:

```bash
uv run export.py
```

By default, the script will create a new export, exporting all Files.
You should see a progress bar indicating the number of Files processed.
If you have Files with more than 100k Logs, the script may take a while to complete.

## Output Structure

The script creates an `exports` directory with timestamped subdirectories.
Every time the script is run, it will create a new timestamped directory in the `exports` directory.

Each export contains:

```
exports/
└── <timestamp>/  # YYYYMMDD-HHMMSS format
    └── export.log  # Logging for the export script
    └── files/
        └── <file_id>/  # E.g. pr_...
            ├── file.json  # File metadata
            ├── deployments.json  # List of Versions deployed to Environments
            ├── versions/
            │   └── <version_id>  # E.g. prv_... Version definition (JSON)
            ├── logs/  # (Only if EXPORT_LOGS is set to true)
            │   └── 1.jsonl, 2.jsonl, ...  # Each file contains up to 1000 Logs; each line contains a single Log
            ├── evaluations/
            │   └── <evaluation_id>/  # E.g. evr_...
            │       ├── evaluation.json  # Evaluation metadata - E.g. Name, Evaluators
            │       ├── stats.json  # Evaluation statistics
            │       ├── runs/
            │       │   └── <run_id>  # E.g. run_... Run definition, e.g. the dataset and version (JSON)
            │       └── logs/  # (Only if EXPORT_LOGS is set to true)
            │           └── 1.jsonl, 2.jsonl, ...  # Each line corresponds to a Log, associated with a Run ID and optional Datapoint, along with the Evaluator Logs containing judgments.
            └── datapoints/  # (Only for Datasets)
                └── <version_id>/  # E.g. dsv_...
                    └── 1.jsonl, 2.jsonl, ...  # Each file contains up to 1000 Datapoints; each line corresponds to a Datapoint
```

## Notes

### Request Errors and Retries

The script retries requests to the Humanloop API if they fail, up to 5 times.
These retries, along with errors, are logged to the `export.log` file.

### Resuming an Export

To resume an existing export that had been previously interrupted, you can specify the export directory
to resume from within `export.py`. E.g.

```python
... existing code ...

if __name__ == "__main__":
    ... existing code ...
    main(
        directory_id=directory_id,
        export_dir=Path("exports/20250606-120000"),  # Pass in a directory here to continue a previous export.
    )
```

This will skip any Files that have already been exported.
