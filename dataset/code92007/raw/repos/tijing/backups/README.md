# Database backups

The scheduled `Backup catalog` workflow writes two equivalent snapshots here:

- `catalog.json`: canonical machine-readable restore file.
- `catalog.md`: human-readable catalog plus the same recoverable JSON payload.

Git history provides retention. The `Restore catalog` workflow accepts either format and requires an explicit confirmation string.
