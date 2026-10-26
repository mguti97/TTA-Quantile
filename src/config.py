"""Central configuration: physics bounds, normalization constants, paths.

All bounds are *hard physics-defined* values (Section 4/6 of the spec), never
data-dependent statistics. The same bounds are used both for normalization and
as hard clipping thresholds during preprocessing.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Each dataset is self-contained under data/<name>/{raw,processed}. The registry
# records the few properties that genuinely differ between providers; everything
# downstream (cleaning/sampling/features/SUVAT) is pure physics and identical.
#   fps    -- native capture rate; drives DT and the horizon frame bounds.
#   loader -- which reader in src/ builds a Match (see data_loader.dispatch).
#   state_stride -- keep every Nth valid state frame (per segment) when sampling,
#                   to bound dataset size; 1 = every frame. Optional, default 1.
#   pitch  -- (length, width) of the playing surface in metres, for the
#             inference grid / reachable-area eval (see PITCH_L/PITCH_W below).
DATASETS = {
    "ffpc_match": {"fps": 29.97, "loader": "pff", "state_stride": 6},           # PFF FC, 64 matches
    # NFL Big Data Bowl 2019 (2017 season, first 6 weeks): one tracking CSV per
    # game, 10 fps, coords in yards (the reader converts to metres). One Match per
    # game, one segment per play. Field is 120x53.3 yd == 109.7x48.8 m. Ships as a
    # single game (2017090700) in this checkout -> split by player, not by match.
    # savgol_window=5: the Savitzky-Golay smoother is defined in FRAMES, so the
    # canonical 9 frames spans 0.9 s at 10 fps (vs 0.3 s at 30 fps) and over-smooths
    # the explosive-burst accelerations that characterise NFL play. 5 frames = 0.5 s
    # restores an fps-comparable smoothing timescale (odd, > SAVGOL_POLY).
    "bdb_match":  {"fps": 10,    "loader": "bdb", "pitch": (109.73, 48.77),
                   "pitch_origin": (0.0, 0.0), "split_mode": "player",
                   "savgol_window": 5},                                         # NFL BDB 2019
    # NBA SportVU 2015-16: 25 fps, coords in feet (reader converts to metres).
    # One Match per game, one segment per quarter; up to 636 games available.
    # Multi-game like ffpc_match -> same state_stride to stay tractable.
    "nba_match":  {"fps": 25,    "loader": "nba", "pitch": (28.65, 15.24),
                   "pitch_origin": (0.0, 0.0), "state_stride": 6},              # NBA SportVU, 636 games
}

# Switch the active dataset by changing DATASET (env override KOGNIA_DATASET wins,
# so a run can target another provider without editing this file). raw/ holds
# source files; processed/ holds derived artifacts.
DATASET = os.environ.get("KOGNIA_DATASET", "ffpc_match")  # active dataset: one of DATASETS
assert DATASET in DATASETS, f"unknown DATASET {DATASET!r}; choose from {list(DATASETS)}"


def dataset_dir(name: str = DATASET) -> Path:
    return DATA_DIR / name


RAW_DIR = dataset_dir() / "raw"
PROCESSED_DIR = dataset_dir() / "processed"


def outputs_dir(name: str = None) -> Path:
    """Per-dataset outputs (models/heatmaps/logs), tagged by the SUVAT caps.

    The caps change the label-noise filter, hence the training set, hence every
    checkpoint / fitted-model json / log derived from it -- so a non-canonical
    caps pair gets its own directory instead of overwriting the paper's run.
    Legacy caps -> no tag -> exactly the historical outputs/<dataset>/ path.
    Cross-dataset scripts must use this helper, not ROOT/"outputs"/<ds>.
    """
    return ROOT / "outputs" / ((name or DATASET) + CAPS_TAG)


# ---- scope of the per-player v_max/a_max profile (see PLAYER_VMAX_PCT below) --
#   "match"       -- pool each player's speeds over the match they appear in.
#                    The canonical build: reproduces the paper's samples.npz.
#   "train_split" -- ONE global table per provider player id, pooled over the
#                    TRAIN-split matches only and applied to train/val/test
#                    alike. Removes the look-ahead leakage of "match" (a test
#                    match's v_max was estimated from that same match's frames).
VMAX_SCOPE = os.environ.get("KOGNIA_VMAX_SCOPE", "match")  # "match" | "train_split"
assert VMAX_SCOPE in ("match", "train_split"), f"unknown VMAX_SCOPE {VMAX_SCOPE!r}"

# ---- SUVAT label filter (see SUVAT_A_MAX/SUVAT_V_MAX below) ------------------
#   "on"  -- canonical: drop training samples whose observed arrival time is
#            faster than the closed-form kinematic minimum (Section 6.3).
#   "off" -- reviewer-requested sensitivity variant (5MaP-M1 / JAvJ-M3): keep
#            every sample regardless of the physics model. The degenerate
#            distance guard (d > 1e-6) stays on either way -- that is numerical
#            hygiene, not physics.
SUVAT_FILTER = os.environ.get("KOGNIA_SUVAT_FILTER", "on")  # "on" | "off"
assert SUVAT_FILTER in ("on", "off"), f"unknown SUVAT_FILTER {SUVAT_FILTER!r}"

# ---- SUVAT model caps (Section 8) -------------------------------------------
# Caps of the two-phase kinematic model (src/pipeline/suvat.py). CANONICAL:
# (8.0, 9.0) -- strictly tighter than the model-free clipping bounds A_CLIP /
# V_CLIP defined below, so the filter removes only displacements the per-frame
# cleaner would already flag. Every published result uses this pair.
#
# These were briefly raised to (10.0, 12.0) on 2026-08-29 -- equal to A_CLIP /
# V_CLIP -- and the pipeline was fully re-run; on 2026-08-30 that widening was
# demoted to a SENSITIVITY ARM and the canonical pair restored. Widening leaves
# the fitted floor put (base MAE 2.602 -> 2.603) but roughens the surface:
# monotonicity violations rise 1.4% -> 23.4% of rays under an identical audit,
# and 99%-gate matched-coverage areas loosen by 12-17%. That two-sided result
# (this arm plus SUVAT_FILTER="off") is what the rebuttal reports as evidence
# that the filter is a label-noise screen and not a prior on the surface; see
# outputs/CAPS_RERUN_RESULTS.md and rebuttal/rebuttal.tex R1-M1.
#
# To reproduce the widened arm without editing this file:
#   KOGNIA_SUVAT_A_MAX=10 KOGNIA_SUVAT_V_MAX=12  -> outputs/*_caps10_12/
SUVAT_A_MAX = float(os.environ.get("KOGNIA_SUVAT_A_MAX", 8.0))   # m/s^2 (< A_CLIP)
SUVAT_V_MAX = float(os.environ.get("KOGNIA_SUVAT_V_MAX", 9.0))   # m/s   (< V_CLIP)

# Canonical caps for the artifacts already on disk (the current paper's results).
# Any other pair tags the artifact names so a re-run lands SIDE BY SIDE and can
# never clobber them; this exact pair yields the empty tag, i.e. today's paths.
_CAPS_LEGACY = (8.0, 9.0)
CAPS_TAG = ("" if (SUVAT_A_MAX, SUVAT_V_MAX) == _CAPS_LEGACY
            else "_caps%g_%g" % (SUVAT_A_MAX, SUVAT_V_MAX))

OUTPUTS_DIR = outputs_dir()   # per-dataset (caps-tagged): models/heatmaps/logs

# Anything built with a non-canonical v_max scope or with the SUVAT filter
# disabled is written NEXT TO the canonical artifacts under a suffix, so
# samples.npz is never overwritten. Suffix parts compose in a fixed order so a
# given (scope, filter) pair maps to exactly one filename. An explicit
# KOGNIA_SAMPLES_SUFFIX wins (e.g. to point train/eval at a variant build while
# leaving both switches at their defaults).
_SUFFIX_PARTS = (("_vmaxtrain" if VMAX_SCOPE == "train_split" else "")
                 + ("_nofilter" if SUVAT_FILTER == "off" else "")
                 + CAPS_TAG)
SAMPLES_SUFFIX = os.environ.get("KOGNIA_SAMPLES_SUFFIX", _SUFFIX_PARTS)
DATASET_NPZ = PROCESSED_DIR / f"samples{SAMPLES_SUFFIX}.npz"
# player_id -> vmax/amax table written by build_dataset in "train_split" mode
# and read back by the eval scripts. Unused in "match" mode.
VMAX_TABLE_NPZ = PROCESSED_DIR / f"vmax_table{SAMPLES_SUFFIX}.npz"

# ---- FFPC (PFF FC) raw layout ----------------------------------------------
# data/ffpc_match/raw/<Competition>/{Tracking Data,Metadata,Rosters}/<gameId>.*
# A competition is any subfolder of raw/ that contains a "Tracking Data" dir.
def ffpc_competition_dirs() -> list[Path]:
    raw = dataset_dir("ffpc_match") / "raw"
    return sorted(p for p in raw.glob("*") if (p / "Tracking Data").is_dir())


# Restrict which PFF games are built (list of str game ids). None -> all found.
# (The 8-game baseline used ids 3812..3819 with SPLIT_MODE="match".)
FFPC_GAME_IDS: list[str] | None = None

# ---- BDB (Big Data Bowl 2019) raw layout -----------------------------------
# Ships as one tracking CSV per game plus shared games/plays/players metadata
# under data/Big-Data-Bowl/Data/ (not the data/<name>/raw convention), so point
# at it directly; processed/ + outputs/ still live under data/bdb_match/ and
# outputs/bdb_match/ per the active-dataset name.
#   tracking_gameId_<gameId>.csv  -- one row per (player, frame) at 10 fps
BDB_TRACKING_DIR = DATA_DIR / "Big-Data-Bowl" / "Data"
# Restrict the build to a subset of game ids (list of int/str); None -> every
# tracking_gameId_*.csv found. This checkout carries a single game (2017090700).
BDB_GAME_IDS: list[str] | None = None
BDB_YARD_M = 0.9144  # 1 yard in metres; tracking coords are yards, the pipeline is SI

# ---- NBA (SportVU 2015-16) raw layout --------------------------------------
# Ships as one 7z archive per game under data/NBA-Player-Movements/data/
# 2016.NBA.Raw.SportVU.Game.Logs/<date>.<away>.at.<home>.7z, each holding a
# single big JSON of overlapping event windows (see src/nba.py for the dedup).
# Restrict the build to a subset of game ids (list of str); None -> all 636.
NBA_GAME_IDS: list[str] | None = None
NBA_FOOT_M = 0.3048  # 1 foot in metres; SportVU coords are feet, the pipeline is SI

MODEL_PT = OUTPUTS_DIR / "mlp.pt"
MODEL_PROFILE_PT = OUTPUTS_DIR / "mlp_profile.pt"  # MLP + per-player max-speed feature

# ---------------------------------------------------------------- capture
# Frame rate is per-dataset (e.g. NBA 25 fps, FFPC 29.97). Everything time-derived
# (DT, derivative spacing, horizon frame bounds) follows from this single value.
FPS = DATASETS[DATASET]["fps"]
DT = 1.0 / FPS  # seconds per frame

# ---------------------------------------------------------------- cleaning (Section 6.1)
# Break a track when a single-frame position jump implies a supra-physical speed
# (ID swaps / teleports). The bound is a SPEED (m/s); the per-frame distance
# threshold is that speed over one frame (JUMP_BREAK_V * DT), so it means the same
# physical thing at any capture rate -- a fixed 1.5 m/frame would be 37.5 m/s at
# 25 fps but only 15 m/s at 10 fps, wrongly breaking real fast tracks. Only catches
# gross discontinuities; the per-frame clipping below removes ordinary kinematic noise.
JUMP_BREAK_V = 37.5             # m/s (teleport speed; 1.5 m / 40 ms at the 25 fps baseline)
JUMP_BREAK_M = JUMP_BREAK_V * DT  # per-frame distance bound at this dataset's fps
# Per-dataset (registry "savgol_window"), default 9. Defined in frames, so a
# dataset's native fps sets the physical smoothing span; low-fps datasets (NFL
# BDB at 10 fps) override to a smaller window to keep that span comparable.
SAVGOL_WINDOW = DATASETS[DATASET].get("savgol_window", 9)  # frames (odd, 7-11 typical)
SAVGOL_POLY = 2
MIN_RUN_LEN = SAVGOL_WINDOW  # a continuous run must be at least this long to use

# Hard clipping -- discard any *state* frame violating these (Section 6.1.4).
# All three are PHYSICAL bounds. V_CLIP/A_CLIP are already fps-invariant (a speed
# and an acceleration); DV_CLIP is a bound on |Δv| between *consecutive frames*,
# which is an acceleration in disguise (|Δv| == a * DT), so it is defined as a
# max inter-frame acceleration (m/s^2) and converted to the per-frame Δv bound via
# DT. A fixed 4 m/s/frame would mean 40 m/s^2 at 10 fps but 120 m/s^2 at 30 fps --
# the same number, three different physical limits; DV_CLIP_A fixes that.
V_CLIP = 12.0    # m/s  (fastest recorded human sprint ~11.5)
A_CLIP = 10.0    # m/s^2 (biomechanical upper bound, on the SMOOTHED accel)
DV_CLIP_A = 100.0             # m/s^2 (max raw inter-frame accel; 4 m/s / 40 ms at 25 fps)
DV_CLIP = DV_CLIP_A * DT      # per-frame |Δv| bound at this dataset's fps

# ---------------------------------------------------------------- features (Section 4)
D_NORM = 120.0  # m
V_NORM = 12.0   # m/s
A_NORM = 10.0   # m/s^2
N_FEATURES = 6

# ---- player-profile features (Section 12): personalised locomotion capacity ----
# Each player's max speed / max acceleration, estimated as the 99th percentile of
# their observed (cleaned, state-valid) speed / accel magnitude -- robust to the
# rare noisy peak yet capturing true top-end capacity. Computed per (match, player)
# at build time and stored as the `vmax`/`amax` meta columns on every sample (does
# NOT alter the base 6-feature X). scripts/train_profile.py appends vmax/V_NORM as
# a 7th input so the MLP can scale its reachable envelope per individual. (amax is
# still built as a meta column for provenance, but the acceleration-profile model
# variants were removed 2026-07-03 -- no between-player spread.)
# VMAX_SCOPE (defined in the paths section, because it renames the output npz)
# selects WHICH frames the percentile pools: the player's own match ("match",
# canonical) or every train-split match they appear in ("train_split", leakage-
# free -- the reviewer-requested variant).
PLAYER_VMAX_PCT = 99.0
PLAYER_AMAX_PCT = 99.0

# ---------------------------------------------------------------- sampling (Section 6.2)
# Training time window (model scope): 0.25 .. 8.0 s. Widened from 0.25..5.0 s on
# 2026-06-25 for the new experiments -- the model is now fit (and meant to be
# trusted) out to an 8 s horizon, where the SUVAT-only extrapolation regime
# previously began. Result tables in the docs predate this and need re-running.
#
# "fixed"  -> spec's integer-second horizons (1..8 s). Clean, but the discrete
#             labels imprint as plateaus between integers in the learned surface.
# "random" -> per state frame, draw N_RANDOM_HORIZONS future offsets uniformly in
#             [HORIZON_MIN_F, HORIZON_MAX_F] frames; label = offset/FPS. Same
#             sample count, but continuous labels -> removes the quantization.
HORIZON_MODE = "random"
HORIZONS_S = [1, 2, 3, 4, 5, 6, 7, 8]             # fixed-mode horizons (seconds)
HORIZONS_F = [int(round(h * FPS)) for h in HORIZONS_S]  # ...in frames
# Random-mode bounds are defined in *seconds* and converted to frames so they
# stay physically constant across datasets with different fps.
HORIZON_MIN_S = 0.10
HORIZON_MAX_S = 8.0
HORIZON_MIN_F = int(round(HORIZON_MIN_S * FPS))    # 6 @25fps, 7 @29.97fps
HORIZON_MAX_F = int(round(HORIZON_MAX_S * FPS))    # 200 @25fps, 240 @29.97fps
N_RANDOM_HORIZONS = 5                              # per state frame (random mode)

# Keep every STATE_STRIDE-th valid state frame (per segment) -> ~1/STATE_STRIDE
# the samples. At ~30 fps adjacent states are ~33 ms apart and highly redundant,
# so multi-match datasets stride down to stay tractable. 1 = every frame.
STATE_STRIDE = DATASETS[DATASET].get("state_stride", 1)

# ---------------------------------------------------------------- SUVAT baseline (Section 8)
# SUVAT_A_MAX / SUVAT_V_MAX are defined in the paths section above, because
# they rename the output artifacts (CAPS_TAG).

# ---------------------------------------------------------------- model / training (Section 7)
PINBALL_TAU = 0.01
HIDDEN = 64
LR = 1e-3
BATCH_SIZE = 4096
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 12
SEED = 42

# ---- area-aware auxiliary loss (canonical for PLAYER models since 2026-07-03) ----
# Pointwise pinball never sees the integrated pitch area its surface produces, so
# per-direction slack in sparse geometries inflates the reachable sublevel set at
# long horizons. The aux term soft-counts the cells of a heading-frame grid inside
# the horizon-h reachable set, sigmoid((h - tta)/AREA_BETA), and penalises the mean
# fraction -- its gradient can only push predictions UP (pure area-shrinking
# pressure), opposed by the pinball asymmetry wherever data contains fast arrivals.
# KNOWN TRADE-OFF at AREA_LAM=0.01: feature-conditioned coverage drops below target
# in the sparsest far bands (30m+ distance, 4-8s predictions); accepted deliberately
# for the ~30% matched-coverage area gain at long horizons (see paper Section on
# calibration). AREA_LAM=0 recovers the pure-pinball recipe. The ball-pass pipeline
# does NOT use this term (no reachable-area concept validated for ball flight).
AREA_LAM = 0.01          # aux weight (canonical); 0 disables
AREA_BETA = 0.25         # sigmoid temperature (s)
AREA_HORIZONS = (3.0, 5.0)  # horizons the soft-area term penalises
AREA_GRID_STEP = 4.0     # heading-frame grid resolution (m)
AREA_GRID_HALF = 48.0    # grid half-extent (m); covers the 5 s reachable set
AREA_AUX_STATES = 48     # states per training step in the soft-area term
AREA_VAL_STATES = 256    # fixed val states for the composite early-stop criterion

# ---------------------------------------------------------------- split (Section 6.5)
#   "match"  -> hold out whole matches (the canonical split). Only meaningful for
#               multi-match datasets like ffpc_match.
#   "player" -> hold out whole players (no trajectory leakage; tests cross-
#               individual generalization). Default for single-match datasets.
#   "block"  -> hold out contiguous time blocks within each half (more samples
#               per split, but a trajectory can straddle a block boundary).
# Per-dataset default (registry "split_mode"), falling back to "match" for the
# multi-match providers; env KOGNIA_SPLIT_MODE overrides for a one-off run. A
# single-match dataset (e.g. the single-game bdb_match) must use "player"
# or "block" -- "match" would put its only match wholly in train.
SPLIT_MODE = os.environ.get(
    "KOGNIA_SPLIT_MODE", DATASETS[DATASET].get("split_mode", "match")
)  # "match" | "player" | "block"
SPLIT_BLOCK_S = 60          # seconds per block (used when SPLIT_MODE == "block")
SPLIT_FRACS = (0.70, 0.15, 0.15)  # train / val / test

# ---------------------------------------------------------------- inference grid (Section 9)
# Playing-surface dimensions (metres) for the reachable-area / heatmap grid.
# Per-dataset when the registry gives a "pitch" (e.g. the NFL field); otherwise
# the standard football pitch. The feature transform is translation/rotation
# invariant, so this only sizes the evaluation grid, not the model.
PITCH_L, PITCH_W = DATASETS[DATASET].get("pitch", (105.0, 68.0))  # along x, along y
# Lower-left corner of the grid in the dataset's own coordinates. Football data
# is centred on the pitch (origin at centre) -> corner at (-L/2, -W/2); NFL NGS
# coords run 0..L / 0..W from a field corner -> origin (0, 0). Registry may
# override with "pitch_origin"; otherwise assume a centred pitch.
PITCH_X0, PITCH_Y0 = DATASETS[DATASET].get("pitch_origin", (-PITCH_L / 2, -PITCH_W / 2))
GRID_RES = 1.0   # metres per cell

# ================================================================ ball-pass pipeline
# A parallel pipeline that predicts the *ball* minimum arrival time through a
# successful pass. It mirrors the player pipeline exactly -- same pinball loss,
# same MLP, same quantile (τ=0.05 lower envelope = the fastest passes) -- but the
# "state" is the ball at pass release and the "target" is the ball at reception.
# Features are driven by the PASSER's velocity (no ball velocity, by design):
#       [ d/D_NORM, cosθ, sinθ, v_passer/V_NORM ]
# where d is release->reception distance and θ is the angle between the passer's
# velocity and the pass direction.
#
# It runs on ffpc_match only: it needs ball tracking + event data, which the other
# datasets lack. So it ignores the active DATASET and pins the FFPC fps.
PASS_PROCESSED_DIR = DATA_DIR / "ffpc_match" / "processed"
PASS_NPZ = PASS_PROCESSED_DIR / "pass_samples.npz"
PASS_OUTPUTS_DIR = ROOT / "outputs" / "ffpc_pass"
PASS_MODEL_PT = PASS_OUTPUTS_DIR / "mlp.pt"

PASS_FPS = DATASETS["ffpc_match"]["fps"]   # ball pipeline is ffpc-only
PASS_DT = 1.0 / PASS_FPS

# Successful pass = PFF passOutcomeType "C" (complete). Others: D/B/O/S (incomplete).
PASS_SUCCESS_OUTCOMES = ("C",)

# Features: distance, two angle components, passer speed. No ball velocity (v1).
PASS_N_FEATURES = 4

# Straight-line ball baseline AND noise floor: a pass cannot travel faster than
# this ball speed, so any observed travel time below d/BALL_V_MAX is a mislabeled
# release/reception pairing and is discarded (mirrors SUVAT's dual role).
BALL_V_MAX = 35.0   # m/s (~126 km/h; hardest strikes ~this, ordinary passes far below)

# Label sanity window (s) -- drop mis-paired release/reception frames -- and a
# minimum pass distance (m) so micro-touches/taps don't dominate the fit.
PASS_T_MIN = 0.2
PASS_T_MAX = 6.0
PASS_D_MIN = 3.0

# Hold out whole matches (the canonical split for a 64-match dataset).
PASS_SPLIT_FRACS = (0.70, 0.15, 0.15)  # train / val / test
