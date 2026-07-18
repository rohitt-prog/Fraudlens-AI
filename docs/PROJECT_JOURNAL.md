# Project Journal

## Refactoring to Monorepo Architecture (2026-07-18)
To scale development and maintain a clean separation of concerns, the repository was migrated to a monorepo setup.

### Decisions
1. **Isolated Packages**: Separated machine learning code (`ml/`), FastAPI backend (`backend/`), and Next.js frontend (`frontend/`).
2. **Self-contained Environments**: Each subproject contains its own dependency files (`requirements.txt`, `package.json`), linter/formatter rules (`pyproject.toml`), and Docker build specifications (`Dockerfile`).
3. **Paths Resolving**: Updated settings system to support looking up `.env` files dynamically across both subproject level and project root directories.
