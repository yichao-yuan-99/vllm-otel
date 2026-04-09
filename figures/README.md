This directory holds publication-style, high-level figures.

Each figure lives in its own subdirectory and should follow the same two-step layout:

1. Data materialization
   - Read raw experiment outputs from `results/` or other source directories.
   - Reorganize them into a figure-specific intermediate dataset.
   - Write the intermediate files into the figure's `data/` subdirectory.
2. Visualization
   - Read only the processed data from `data/`.
   - Render the final figure into a local output path such as `output/`.

Expected structure for each figure directory:

- `description.txt`
  - Start with a paper-ready figure title on its own line.
  - Then include a concise single paragraph in 2-3 sentences.
  - Explain what the figure shows and what comparison or trend it is meant to highlight.
- `README.md`
  - Explain the goal of the figure.
  - Document the source data location.
  - Define every metric and formula used in the plot.
  - Show the exact commands for step 1 and step 2.
- `data/`
  - Stores generated intermediate datasets only.
  - Must be gitignored.
- plotting and materialization scripts
  - Keep the raw-data collection step separate from the rendering step.

When possible, prefer figure pipelines that are:

- Re-runnable from the command line.
- Deterministic for the same input data and parameters.
- Explicit about the selected time window and filtering rules.
