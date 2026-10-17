# Walkthrough (for Mariam)

## Why it is built this way

Fingerprinting has no physics in the model: it never turns RSSI into distance. It memorises
what the radio environment looks like at known spots and later finds the most similar spot.
So the repo is one loader, one feature transform, two interchangeable models and one
evaluation path. Both models have the same `fit` and `predict`, so the CLI, the
cross-validation and the tests do not care which one they hold.

A random K-fold on the training file gives a kNN median
error of 0.0 m, which looks wonderful and means nothing: each reference point was
sampled about 22 times, so the held-out fingerprint's twin is in the training fold. The dataset
ships a validation file taken four months later by other people on mostly other phones. That
is the only number quoted as the result, and the README says why.

Every number in the README comes from `results/results.json`, which `make figures` rewrites.

## File map

- `src/rssi_loc/data.py`: SHA-256 table, download with fallbacks, `Fingerprints` (RSSI
  matrix, UTM metres, floor, building, metadata). Code 100 ("not seen") becomes -105 dBm here.
- `features.py`: dBm to [0, 1]; `positive` is linear, `powed` raises it to the power e.
- `models.py`: `KnnLocalizer` finds k neighbours once and uses them for both the weighted
  mean position and the floor/building vote. `MlpLocalizer` is two scikit-learn networks: a
  regressor on standardised x, y and a classifier on a packed building×floor label.
- `evaluate.py`: errors in metres, `Metrics`, `cross_validate`, the k sweep and
  `rssi_repeatability`, which measures the noise directly from repeated samples.
- `plots.py`: four figures, no computation. `cli.py`: `download` and `run`.
- `synthetic.py`: a toy building with log-distance path loss, so `make check` and
  `make smoke` work without the 45 MB download. Its errors are not evidence of anything.
- `tests/`: 14 tests; the download test serves a corrupt zip and a good mirror from memory.

## Three questions a professor may ask

**Why is the median 4.8 m but the mean 8.1 m?**
The error distribution has a long tail (95th percentile 23.4 m). Most test fingerprints
land in the right corridor; a few match a similar-looking spot in another wing or building,
and one 50 m miss moves the mean a lot. So the CDF is the figure, not one number, and
"accurate to X m" needs a percentile.

**Why does kNN beat a neural network here?**
With 19 937 fingerprints at only 905 spots, the training set is a noisy lookup table, and
kNN is exactly a noisy lookup. The MLP must learn a smooth map from 520 inputs to two outputs
from the same 905 spots; it is close (9.7 m mean) but not better in this run, and it
was not tuned on the test set. Hierarchical models (building, then floor, then position) and ensembles do better
in the literature.

**Would this work on a farm or in a factory?**
The method, yes: an ESP32 sees the same RSSI vector with `WiFi.scanNetworks()`. The accuracy,
only if a few metres is enough. One access point at one spot varies by 3.0 dB (median),
a factor 1.3 in distance at path-loss exponent 3, and metal, vehicles and animals
make it worse. For "which pen" or "which room", fine; for "where is the robot", use UWB.

## One thing to extend

Add the hierarchical variant: a building classifier, then a floor classifier per building,
then a per-floor kNN on position. About 40 lines on top of `KnnLocalizer`, reusing every evaluation
function; then you can say from your own run whether it closes the gap to published numbers.
