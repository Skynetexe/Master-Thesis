# Transparent Hybrid Music Recommender at Full-Catalogue Scale

Code for the MSc thesis of the same name.

The pipeline maps the 21 billion Last.fm listens of MLHD+ to 6.6 million MusicBrainz recordings and links them to Spotify audio features and genres. From these it computes nine evidence concepts, and a generalized additive model (GAM) over six of them ranks the full catalogue. The baselines are popularity, implicit ALS and ItemKNN.

## Contents

| Files | What they do |
|---|---|
| `FullDataDownload.ipynb` | Downloads the data sources. |
| `MlhdToParquet.py`, `verify_mlhd_parquet.py` | Convert MLHD+ to Parquet and check the result against the raw files. |
| `MusicbrainzToParquet.py`, `CanonicalToParquet.py` | Convert the MusicBrainz dumps and the canonical data to Parquet. |
| `BuildFeatures.py` | Builds the feature tables and the interaction matrix. |
| `SpotifyBridge.py`, `FuzzyBridge.py`, `FeaturesExtend.py` | Link the tracks to Spotify (the ISRC and URL tiers, then the name tiers) and build the audio features. |
| `FuzzyCensus.py` | Measures the precision of the name matcher for the thesis. |
| `normalizer_check.py` | Shows the "feat." defect of the title normalizer. |
| `als_status.py` | Shows the state of a running ALS search. |
| `pipeline/` | The runner, the stages P1 to P13b2, the script stages after them, and the scripts that run outside the runner, such as `tree_audit.py`, `lb_demo.py` and `figures.py`. |
| `curated/` | The hand-made tag lists and the chosen ALS parameters, which the stages read. |
| `Thesis Figures.ipynb` | Draws the figures of the thesis. |
| `test_*.py` (26 files), `p13_world.py`, `p13_chain.py` | The tests, with the synthetic world and the stage chain that several of them share. |
| `build_fixture.py`, `extend_fixture_spotify.py`, `extend_fixture_fuzzy.py` | Build a small synthetic data folder for the preparation scripts. The text block at the end of each file gives the expected output. |
| `demo/`, `figures/`, `model/` | Copies of outputs (see below). |
| `requirements.txt` | The packages and their versions. |

## Requirements

- Python 3.12 or later, and the packages in `requirements.txt`.
- `zstdcat` (part of zstd), for `verify_mlhd_parquet.py`.
- Playwright for Python and its Chromium browser, for the browser checks in `test_demo_page.py`. Without Playwright, the test skips these checks.
- For the full pipeline: a computer with 32 GB of memory, about 500 GB of disk for the Parquet tables, and more for the downloads.

To install the packages, then Playwright and its browser:

```
python3 -m pip install -r requirements.txt
python3 -m pip install playwright
python3 -m playwright install chromium
```

## Tests

The tests need no data. Each test makes its own small synthetic data and prints its checks. Run each one from the repository root, for example:

```
python3 test_gam.py
```

## Data

The data is not in the repository. The Parquet tables take about 500 GB, and the Spotify data may be used for research only. The sources are:

- MLHD+: Last.fm listening logs with MusicBrainz identifiers, from MetaBrainz, for non-commercial use.
- The MusicBrainz full export and canonical data: the core data under CC0, the tags under CC BY-NC-SA 3.0.
- A third-party copy of the Spotify catalogue (the Kaggle dataset `lordpatil/spotify-metadata-by-annas-archive`), distributed for research use only.
- The ListenBrainz API, for the demo only.

The code reads the data folder from the environment variable `THESIS_DATA_ROOT`. Without it, the code uses `data/` in the repository. A new copy has no such folder, so the runner stops at "no interaction files".

A full run, from the repository root, with `THESIS_DATA_ROOT` set:

1. `FullDataDownload.ipynb`: switch on the sources (`enabled`), run the dry run to see the sizes, and set `CONFIRM_DOWNLOAD = True` to download. The Spotify copy needs a Kaggle API token.
2. `MlhdToParquet.py`, then `verify_mlhd_parquet.py`.
3. `MusicbrainzToParquet.py` and `CanonicalToParquet.py`.
4. `BuildFeatures.py`.
5. `SpotifyBridge.py`, `FuzzyBridge.py`, then `FeaturesExtend.py`.
6. The runner. The script stage `als_phase_d` needs the ALS search of phases A to C, so the first full run stops there. Run the search, then run the runner again. Stages with a current marker are skipped.
   ```
   python3 -m pipeline.runner --only p1_audit
   python3 -m pipeline.runner        # stops at als_phase_d
   python3 -m pipeline.als_tune
   python3 -m pipeline.runner
   ```
7. The tree audit, the demo and the figure script:
   ```
   python3 -m pipeline.tree_audit
   python3 -m pipeline.lb_demo --user <ListenBrainz name>
   python3 -m pipeline.figures --demo-user <name>
   ```
8. The figures notebook, `Thesis Figures.ipynb`.

The docstring of each script gives its options. Each finished stage of the runner writes a marker with the hashes of its parameters, its code and its inputs. A stage runs again when one of them changes, or with `--force`. A script stage that finds its outputs without a current marker stops instead, with exit code 2, and names two commands:

```
python3 -m pipeline.runner --only <stage> --adopt
python3 -m pipeline.runner --only <stage> --force
```

The first gives the outputs a marker without a run, after a freshness check and a read-back of the outputs. The second computes them again.

## Demo, figures, models

Three folders hold small copies of outputs that the code writes to the data folder, so that the demo page and the figures can be seen without the data. They were added after the runs, with the agreement of the supervisor. The long module docstrings keep the rules as they were declared before the runs, so some of them still say that these outputs stay in the data folder.

- `demo/`: the demo page. Open `demo/index.html` in a browser. `demo/users.js` names the one listener, and `demo/listener/demo_data.js` holds every number on the page. The listener is the thesis author, with the account name replaced and the dates cut to years. The file holds the listening history itself: 21,188 ListenBrainz listens from 2022 to 2026, and the top artists with their listen counts. It also holds values from Spotify for the tracks of the demo: track ids, ISRCs, popularity, genres, and the audio reasons, which come from the Spotify audio features. `python3 -m pipeline.lb_demo --user <name>` builds the page for another ListenBrainz account, in the data folder.
- `figures/`: the 15 figures of the thesis as PNG (300 dpi), PDF and SVG, each with a CSV file of the plotted values, and `captions.md` with the captions as the notebook writes them, the sources, and the values that the text quotes. The thesis has the final wording of the captions. `fig_demo_match.*` and `fig_explanations.*` show the listening history of the demo. The figures notebook writes the figures to `parquet/model/figures/notebook/` in the data folder. The files that the Sources lines name are outputs in `parquet/model/` and `parquet/scratch/` there.
- `model/`: the final GAM with its bootstrap and its near-optimal set (`p13c_bounded_p_pop-B_lambda_model.json`, which `p13c_final.json` names as final), the refit (`gam_refit.json`), the bounds of P13d (`p13d.json`), the cold-track GAM of P14 (`p14_cold_gam.json`) and the tree-GAM audit (`tree_audit.json`). The audio terms are in standard deviations over the catalogue.

## Known defects

Two defects in the Spotify bridge were found too late to fix. The bridge keeps the code as it ran, so the stored tables and the results of the thesis include their effects.

- The title normalizer in `FuzzyBridge.py` removes "feat" with no word boundaries and keeps the name after "feat." in the key, so "Defeat" becomes "de". `normalizer_check.py` shows the defect, and the thesis discusses it.
- The URL steps in `SpotifyBridge.py` read only the web form of a Spotify link (open.spotify.com or play.spotify.com). They are meant to fall back to the `spotify:track:` and `spotify:artist:` forms, but `regexp_extract` returns an empty string, not NULL, when its pattern does not match, so `coalesce()` never reaches the second pattern. Item 3 of the fixture in `extend_fixture_spotify.py` shows the defect. The MusicBrainz export used in the thesis holds no Spotify links in these forms, so this defect does not change the results.
