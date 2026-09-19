# Scripts

Helper scripts for local development and deployment.

## Files

- `setup.ps1` - One-time local setup (venv + deps + .env).
- `dev.ps1`   - Start the dev server with hot-reload.
- `ingest.ps1` - Re-ingest all documents from `data/uploaded_docs/`.

## Usage

```powershell
# From the project root
.\scripts\setup.ps1
.\scripts\dev.ps1
.\scripts\ingest.ps1
