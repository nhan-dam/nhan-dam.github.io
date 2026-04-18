# Nhan Dam's Machine Learning Notes

Technical notes on Machine Learning.

**Published site:** https://nhan-dam.github.io

## Local preview

```bash
pip install -r requirements.txt
mkdocs serve
```

Then open http://localhost:8000.

## Structure

```
docs/           # all note content (Markdown)
  rlhf/         # RLHF curriculum notes
  projects/     # project write-ups
  reference/    # setup guides and cheat sheets
src/            # source code for projects
mkdocs.yml      # site configuration
```

## Deploying

Push to `main` — the GitHub Actions workflow in `.github/workflows/deploy.yml` builds and deploys automatically to GitHub Pages.
