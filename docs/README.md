# Docs site

A static, multi-page site that explains both bundles in this repo, as an
education and developer-advocacy resource. It covers the business value, the
datasets and their domain, what each model and reducer does, how
PCA/ICA/dimensionality reduction reads outcomes, and where the pattern reuses.

Plain HTML and one CSS file. No build step, no dependencies.

## Host it on GitHub Pages

1. Push the repo to GitHub.
2. **Settings → Pages → Build and deployment.**
3. Source: **Deploy from a branch**. Branch: your default branch, folder: **`/docs`**.
4. Save. The site publishes at `https://<user>.github.io/<repo>/`.

`.nojekyll` is included so Pages serves the files as-is.

## Pages

| File | What it covers |
|---|---|
| `index.html` | Business value, framed for education and developer advocacy |
| `bundles.html` | Both bundles in depth: datasets, domain, medallion flow, infra, and the obj_recg vision model zoo |
| `dr-methods.html` | PCA, ICA, random projection, clustering, the classic classifiers, and how they surface which actions move an outcome |
| `roadmap.html` | Built vs. next (from the source TODOs) and the vision for reuse across education |

Models are documented per bundle: the vision model zoo lives in `bundles.html`
under obj_recg, and the classic classifiers live in `dr-methods.html` as the
pipeline's model step.

## Edit

Content lives in each HTML file; shared styling is `assets/site.css`, and the mobile
nav toggle is `assets/site.js`. Colors and type are bound to the Databricks FDE
palette (oat paper, navy ink, one lava accent) and DM Sans / DM Mono.
