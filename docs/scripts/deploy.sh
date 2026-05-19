uv run mkdocs build -f pages/mkdocs.yml
uv run mkdocs gh-deploy --config-file ./pages/mkdocs.yml --remote-branch docs