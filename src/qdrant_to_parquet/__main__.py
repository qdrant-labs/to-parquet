from qdrant_to_parquet.cli import main

if __name__ == "__main__":  # worker processes (spawned) import this module too
    raise SystemExit(main())
