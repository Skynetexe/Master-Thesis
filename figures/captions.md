# Figure captions

The figures of the thesis in thesis order, as drawn by the figures notebook. Each figure is in this folder as PNG (300 dpi), PDF and SVG, with its plotted values in a CSV file of the same name. The thesis links the PNG files.

## Figure 3.1: fig_pipeline.png

Figure 3.1: The pipeline from the raw sources to explained recommendations. Each box is a phase, and each row in it a stage of the runner, which reads only the outputs of earlier stages. Grey stages (P7, P8, P9, P10) are not used by the final model. The last box holds the plain scripts and the notebook that draws the figures from the stored outputs.

Sources: the markers of the 31 runner stages (markers/).

## Figure 3.2: fig_bridge_tiers.png

Figure 3.2: Tiers of the Spotify bridge, from the most certain rule (top) to the least certain, with each tier's share of the 6,637,382 items and of all listens. The identifier tiers carry 68.6% of the listens but only 15.4% of the items, because popular tracks are the ones that carry identifiers.

Sources: Table B.1 of the thesis, checked against id_spaces.json and pipeline/config.py.

Values for the text: identifier tiers 15.37% of items and 68.65% of listens; matched tiers 61.04% of items and 94.52% of listens.

## Figure 3.3: fig_coverage.png

Figure 3.3: Share of rows with data for each concept of the final model, for random tracks of the catalogue (uniform negatives), popularity-weighted tracks (p_pop negatives, Section 4.2) and played tracks (training positives). The more popular the tracks, the more often each kind of evidence exists.

Sources: p13a_model.json (audit.missing.concepts).

Values for the text: Same-artist co-listening 72.2 / 99.1 / 99.3%; Artist affinity 99.7 / 100.0 / 99.9%; Artist-tag affinity 56.3 / 84.3 / 90.8%; Tag affinity 16.9 / 59.7 / 71.8%; Genre affinity 42.4 / 68.0 / 71.7%; Audio fit 51.3 / 81.9 / 87.2%.

## Figure 4.1: fig_loss_forms.png

Figure 4.1: How the loss form changes the popularity shape. (a, b) The popularity term at percentiles of the positives' popularity under Form A and Form B, for three samplers with 16 negatives each. (c) The largest gap between the explanation values of a fit and those of the reference fit with 8,192 negatives, for the mixture sampler, against the number of negatives (both axes logarithmic).

Sources: p13b_model.json (arms, shape_quantiles), p13b2_model.json (ref_fits).

Values for the text: popularity p95 minus p5: uniform-A +11.9, p_pop-A -11.5, mixture-A -10.7, uniform-B +2.7, p_pop-B +2.0, mixture-B +1.9; largest gap at m = 32: Form A 11.78, Form B 1.24.

## Figure 4.2: fig_shapes.png

Figure 4.2: Shape functions of the final model, in log-odds points, centred on the popularity-weighted negatives of its training data. The blue line is the fitted shape with its pointwise 95% band over 100 listener bootstrap refits, and the grey line is the same model fitted without monotone bounds. The points at the left are the no-match and no-data states. The marks under each curve show the median, the middle half and the 5th to 95th percentiles of the concept's values among the training positives with a match. Each panel has its own vertical scale, and its importance is the mean absolute contribution of the term.

Sources: p13c_bounded_p_pop-B_lambda_model.json, p13c_bounded_p_pop-B_lambda_arrays.npz, p13a_model.json, p13d_model_set.npz, scratch/p13c_<tag>/fits.json (background means).

Values for the text: Tag affinity: p5/p50/p95 +0.247 / +0.247 / +0.247, no match -0.596, no data -0.334, importance 0.264; Artist-tag affinity: p5/p50/p95 +0.118 / +0.367 / +0.367, no match -1.929, no data -0.333, importance 0.373; Genre affinity: p5/p50/p95 +0.203 / +0.240 / +0.290, no match -1.131, no data +0.289, importance 0.335; Artist affinity: p5/p50/p95 +1.216 / -1.247 / -2.690, no match -0.044, no data -0.000, importance 0.167; Same-artist co-listening: p5/p50/p95 +2.888 / +7.884 / +12.122, no match -0.392, no data -0.748, importance 0.673; Audio fit: p5/p50/p95 -0.130 / +0.154 / +0.261, no data -0.279, importance 0.162.

## Figure 4.3: fig_artist_suppression.png

Figure 4.3: Suppression between the two artist terms. Artist affinity in the final model (solid blue) and in the model without co-listening (pink), and same-artist co-listening in the final model (dashed blue), at percentiles of each concept's values among the training positives.

Sources: p14_cold_gam.json (shapes).

Values for the text: Same-artist co-listening, final model: p5 +2.888, p50 +7.884, p95 +12.122; Artist affinity, final model: p5 +1.216, p50 -1.247, p95 -2.690; Artist affinity, model without co-listening: p5 +3.267, p50 +5.734, p95 +8.110.

## Figure 6.1: fig_accuracy.png

Figure 6.1: Accuracy on the evaluation listeners. (a) NDCG@100 of every arm. (b, c) The GAM minus each arm, with 95% intervals over listeners. The grey band marks the SESOI of 0.00083 on either side of zero. C1 to C4 are the confirmatory tests, E1 to E3 the exploratory tests on NDCG@100.

Sources: p14.json (arms, tests), checked against p14.npz.

Values for the text: Artist terms only 0.04499; ItemKNN, same-artist list 0.04152; GAM 0.04150; ItemKNN, both lists 0.03925; ALS, 256 factors 0.02647; ALS, 64 factors 0.01673; Cold-track GAM 0.00933; Popularity 0.00290.

## Figure 6.2: fig_slices.png

Figure 6.2: NDCG@100 by slice. (a) By popularity band of the pick, with the number of picks per band. (b) For picks with and without a same-artist co-listening neighbour in the listener's history.

Sources: p14.json (slices), checked against p14.npz.

Values for the text: without a neighbour: 7,396 of 39,998 picks; GAM 0.00000, Artist terms only 0.00000, ItemKNN, same-artist list 0.00000, ALS, 256 factors 0.00190, Popularity 0.00237; GAM with a neighbour 0.05092.

## Figure 6.3: fig_beyond.png

Figure 6.3: The top-10 lists of each arm beyond accuracy (19,999 listeners). (a) The cumulative share of listeners whose first track by a new artist stands at or above each rank (logarithmic axis), the curve behind the median of Table 6.4. (b) The change in NDCG@10 against no cap, with 95% intervals over listeners, against the mean number of distinct artists in the top 10. From left to right, each line passes no cap and caps of 5, 3, 2 and 1 tracks per artist (Table 6.5).

Sources: beyond.npz (first_new_<arm>, <arm>_cap_ndcg10, <arm>_cap_distinct), beyond.json (results.arms).

Values for the text: first new artist, median rank: GAM 46,181, Artist terms only 106,321, Cold-track GAM 112,635, ItemKNN, same-artist list 45,648, ItemKNN, both lists 175, ALS, 256 factors 37, Popularity 1; GAM cap 2: +0.00223 (+0.00178 to +0.00269), 6.48 artists.

## Figure 6.4: fig_cold_start.png

Figure 6.4: (a) NDCG@100 and (b) R@100 of 19,995 evaluation listeners when each history keeps only its first N training pairs, with 95% intervals over listeners. The points at the right use the full history. Popularity ignores the history. Both axes are logarithmic.

Sources: cold_start_curve.json (results), checked against cold_start_curve.npz.

Values for the text: GAM NDCG@100 at N = 5 0.00322, 10 0.00398, 25 0.00493, 100 0.00768, full 0.04151; N = 100 reaches 18.5% of the full history.

## Figure 6.5: fig_reasons.png

Figure 6.5: Reasons on the GAM's top 10 for the evaluation listeners (199,990 tracks). For each reason, the share of tracks where its concept matched, where it passes the bootstrap rule D1, and where it is shown after the model-choice rule S4, with its mean contribution and odds factor when shown.

Sources: p14.json (reasons.rows), checked against p14.npz.

Values for the text: Artist shown 100.0%; Artist tags shown 99.8%; Track tags shown 54.5%; Audio fit shown 4.9%; Genres shown 4.6%.

## Figure 6.6: fig_evidence_removal.png

Figure 6.6: Removing the evidence of the shown reasons, for the GAM's top 10 of 19,999 listeners. (a) The share of tested tracks that leave the top 10, with 95% intervals over listeners. Row 3 keeps only the artist evidence, and its label adds the share that stay. (b) The new rank of the tested tracks: median (point), 25th to 75th percentile (thick line) and 5th to 95th percentile (thin line), on a logarithmic axis. The vertical line marks rank 10. The rows follow Table 6.8.

Sources: evidence_removal.json (results), evidence_removal.npz (rank_*).

Values for the text: 1 Listed co-listened tracks removed: 37.6% leave, median 7; 2 All history tracks by the artist removed: 100.0% leave, median 151,878; 3 Only the history tracks by the artist kept: 40.4% leave, median 8; 4 Cited tags and genres removed: 25.5% leave, median 6; 4 Artist tags: 18.8% leave, median 6; 4 Track tags: 33.1% leave, median 7; 4 Genres: 81.7% leave, median 85; 5 No audio data: 62.8% leave, median 15; 6 Evidence of every shown reason removed: 100.0% leave, median 302,792.

## Figure 6.7: fig_tree_audit.png

Figure 6.7: The tree-based audit. Each panel shows one term at percentiles of the training positives' values and at its no-match (NM) and no-data (ND) states, for the boosted tree model (with the range over its five bags), the free spline fit and the bounded final model (each with its 95% listener band).

Sources: tree_audit.json (points.rows).

Values for the text: tree model, change from p5 to p95: Artist affinity -3.65, Artist-tag affinity +0.31, Audio fit +0.42, Same-artist co-listening +9.04, Genre affinity +0.19, Tag affinity -0.34.

## Figure 6.8: fig_demo_match.png

Figure 6.8: What a catalogue that ends in 2013 keeps of a recent history. (a) The ten most played artists of the demonstration listener's 21,188 ListenBrainz listens, marked by whether they are among the 15 largest artists of the matched history. (b) The 15 largest artists of the matched history.

Sources: demo/<user>/lb_demo.json (match_artists, history.top_artists).

Values for the text: seven of the ten most played artists are not among the 15 largest matched artists.

## Figure 6.9: fig_explanations.png

Figure 6.9: The reasons shown for three of the GAM's top 25 tracks for the demonstration history (ranks 1, 10 and 11). Each bar is the contribution of a reason to the log-odds of the track against the mean candidate, with its odds factor. Only the reasons that pass both stability rules appear.

Sources: demo/<user>/lb_demo.json (gam, model).

Values for the text: rank 1: Artist +10.436 (×34,058), Artist tags +0.557 (×1.75), Track tags +0.489 (×1.63); rank 10: Artist +10.002 (×22,067), Artist tags +0.557 (×1.75), Track tags +0.489 (×1.63); rank 11: Artist +9.822 (×18,432), Artist tags +0.557 (×1.75).
