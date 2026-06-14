# Releasing `phileas-memory`

Phileas publishes to [PyPI](https://pypi.org/project/phileas-memory/) so that
`pip install phileas-memory` works for anyone. Publishing runs automatically from
GitHub Actions (`.github/workflows/release.yml`) when a GitHub Release is
published, using [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — there is no API token stored in the repository.

## One-time setup (maintainer)

Do this once, before the first release.

1. Reserve the name on PyPI by creating the project's first release (the steps
   below) — or, if the name is unclaimed, configure a **pending publisher** so
   the project is created by the first Trusted-Publishing upload.
2. On PyPI, go to the project (or **Your projects → Publishing** for a pending
   publisher) and add a **GitHub** trusted publisher with:
   - **Owner:** `alexajuno`
   - **Repository:** `phileas`
   - **Workflow name:** `release.yml`
   - **Environment:** `pypi`
3. In the GitHub repo settings, create an **Environment** named `pypi`
   (Settings → Environments → New environment). No secrets are needed; the
   environment just scopes the OIDC trust and lets you add reviewers later.

## Cutting a release

1. Make sure `main` is green and pick the new version (semver).
2. Bump `version` in `pyproject.toml` and commit it on `main`.
3. Tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. Create a GitHub Release for that tag (`gh release create v0.1.0 --generate-notes`).
   Publishing the release triggers `release.yml`, which builds the sdist + wheel,
   runs `twine check`, and publishes to PyPI.
5. Confirm it landed: `pip install phileas-memory==0.1.0` in a clean environment.

## Building locally (optional sanity check)

```bash
pip install build twine
python -m build          # writes dist/*.tar.gz and dist/*.whl
twine check dist/*
```
