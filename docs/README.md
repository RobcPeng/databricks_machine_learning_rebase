# Docs site

A static, multi-page site that documents both bundles in this repo, as an
education and developer-advocacy resource. It grew out of rebuilding
machine-learning coursework as two real Databricks Asset Bundles.

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
| `index.html` | Overview: repository and purpose, plus the personal intro |
| `architecture.html` | Architecture landing, linking to the two breakdowns |
| `arch-cv.html` | Computer vision architecture: topology, data flow, ERD, schema reference |
| `arch-dr.html` | Dimensionality reduction architecture: topology, data flow, ERD, per-cell pipeline |
| `repositories.html` | Repositories landing, linking to the two walkthroughs |
| `repo-cv.html` | Computer vision repo (obj_recg): zero-to-hero walkthrough |
| `repo-dr.html` | Dimensionality reduction repo (pca_ica): zero-to-hero walkthrough |
| `kanban.html` | Shipped, next up, and backlog, from the source TODOs |

Navigation: Overview · Architecture (▾ Computer vision, Dimensionality reduction) ·
Repositories (▾ Computer vision, Dimensionality reduction) · Kanban. There are no
footers by design. Architecture pages are diagram- and table-first; Repositories
pages are the step-by-step walkthroughs.

## Edit

Content lives in each HTML file; shared styling is `assets/site.css`, and the mobile
nav toggle is `assets/site.js`. Colors and type are bound to the Databricks FDE
palette (oat paper, navy ink, one lava accent) and DM Sans / DM Mono.
