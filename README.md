# ⛟ Export → Parquet

CLI tool to export a Qdrant collection to a Parquet file.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qdrant-labs/to-parquet/main/docs/export-dark.gif">
  <img src="https://raw.githubusercontent.com/qdrant-labs/to-parquet/main/docs/export-light.gif" width="792" alt="The collection is split into ranges, workers export them in parallel and save chunks, an interrupted export resumes from the saved chunks, and the chunks are joined into out.parquet">
</picture>

- All vector types: unnamed, named, multi-vectors and sparse.
- Parallel: ranges of points are exported by worker processes.
- Resumable: an interrupted export continues where it stopped.

## Usage

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uvx qdrant-to-parquet my_collection out.parquet            # localhost

export QDRANT_URL=https://xyz.cloud.qdrant.io QDRANT_API_KEY=...
uvx qdrant-to-parquet my_collection out.parquet            # Qdrant Cloud
```

| Option          | Default                                  |                                                      |
| --------------- | ---------------------------------------- | ---------------------------------------------------- |
| `--url`         | `$QDRANT_URL` or `http://localhost:6333` | Qdrant URL                                           |
| `--api-key`     | `$QDRANT_API_KEY` or none                | API key                                              |
| `--grpc-port`   | `6334`                                   | gRPC port, on the host of `--url`                    |
| `--rest`        | off (gRPC)                               | Use REST instead of gRPC                             |
| `--workers`     | up to `4`, by CPUs and collection size   | Worker processes                                     |
| `--batch-size`  | `256`                                    | Points per request                                   |
| `--timeout`     | `60`                                     | Request timeout in seconds                           |
| `--retries`     | `3`                                      | Retries per failed request                           |
| `--restart`     | off (resume)                             | Discard an interrupted export instead of resuming it |
| `-f`, `--force` | off                                      | Replace the output file if it already exists.        |
| `-q`, `--quiet` | off                                      | No progress bar, warnings or summary                 |

Exit codes: `0` done, `1` failed, `2` invalid arguments, `130` interrupted, `143` terminated.

## Output

| Column | Type |
| --- | --- |
| `id` | `string`: an integer (`"42"`) or a UUID |
| `vector` (the unnamed vector) or `vector_<name>` | `list<float32>` |
| `vector_<name>` (multi-vector) | `list<list<float32>>` |
| `vector_<name>` (sparse) | `struct<indices: list<uint32>, values: list<float32>>` |
| `payload` | `string`: JSON, keys sorted |

Rows are in point id order. A missing vector is `null`. The file metadata holds the collection's configuration.

```sql
-- DuckDB
SELECT id, payload->>'title' AS title, len(vector) AS dim FROM 'out.parquet';
```

## Library

```python
from qdrant_to_parquet.exporter import ClientConfig, export_collection

if __name__ == "__main__":
    config = ClientConfig(url="http://localhost:6333")
    export_collection(
        config.make(), "my_collection", "out.parquet", workers=4, client_config=config
    )
```

## Development

```sh
uv run pytest                                      # without a server
QDRANT_URL=http://localhost:6333 uv run pytest     # also against a server, over gRPC and REST
```
