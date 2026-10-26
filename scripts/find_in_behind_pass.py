"""Find long passes played in behind the defensive back line (FFPC data).

A "ball in behind" is defined at the moment of RELEASE: the reception point
lies beyond the defensive line as it stood when the pass was played, and the
receiver starts in front of that line and runs beyond it to collect.
"""
import numpy as np

from src.ffpc import available_game_ids
from src.ffpc_passes import load_ffpc_pass_match

LONG_PASS_MIN = 25.0   # m, straight-line ball travel release -> reception
RUN_MIN = 4.0          # m the receiver must advance (attack direction) release -> reception
LINE_MARGIN = 1.0      # m the reception point must be beyond the release-time line
ONSIDE_TOL = 0.5       # m of tolerance for "receiver level with the line" at release
RECV_BALL_MAX = 3.0    # m, receiver must actually be at the ball at reception


def seg_bounds(match, period):
    for seg_id, lo, hi in match.segment_slices():
        if seg_id == period:
            return lo, hi
    return None


def team_gk_idx(match, team, lo, hi):
    """Proxy goalkeeper: player with the largest mean |x| over the half (hugs own goal line)."""
    best_idx, best_val = None, -1.0
    for p in np.where(match.side_id == team)[0]:
        xs = np.abs(match.positions[lo:hi, p, 0])
        if not np.isfinite(xs).any():
            continue
        m = np.nanmean(xs)
        if m > best_val:
            best_val, best_idx = m, p
    return best_idx


def attack_dir_for(match, gk_idx, lo, hi):
    """Attack away from the own goalkeeper's side of the pitch."""
    gk_x = np.nanmean(match.positions[lo:hi, gk_idx, 0])
    return -1.0 if gk_x > 0 else 1.0


def defensive_line_x(match, def_team, frame, attack_dir, gk_idx):
    """x of the last outfield defender (the offside line) at `frame`."""
    idxs = [p for p in np.where(match.side_id == def_team)[0]
            if p != gk_idx and match.in_field[frame, p] == 1
            and not np.isnan(match.positions[frame, p, 0])]
    if not idxs:
        return np.nan
    xs = match.positions[frame, idxs, 0]
    return xs.max() if attack_dir > 0 else xs.min()


def receiver_idx(match, pm, pe, poss_team):
    """Teammate closest to the ball at reception; -1 if nobody is actually on the ball."""
    ball_recv = pm.ball[pe.reception_g]
    idxs = [p for p in np.where(match.side_id == poss_team)[0]
            if p != pe.passer_idx and match.in_field[pe.reception_g, p] == 1
            and not np.isnan(match.positions[pe.reception_g, p, 0])]
    if not idxs:
        return -1
    d = [np.hypot(*(match.positions[pe.reception_g, p] - ball_recv)) for p in idxs]
    k = int(np.argmin(d))
    if d[k] > RECV_BALL_MAX:
        return -1  # nearest teammate is nowhere near the ball -> not a clean reception
    return idxs[k]


def find_candidates(gid):
    pm = load_ffpc_pass_match(gid)
    match = pm.match
    out = []
    for pe in pm.passes:
        if pe.passer_idx < 0:
            continue
        f0, f1 = pe.release_g, pe.reception_g
        if not (0 <= f0 < pm.ball.shape[0] and 0 <= f1 < pm.ball.shape[0]):
            continue
        # all 22 players tracked in-frame for the whole flight of the pass
        if not np.all((match.in_field[f0:f1 + 1] == 1).sum(axis=1) >= 22):
            continue

        ball0, ball1 = pm.ball[f0], pm.ball[f1]
        if np.isnan(ball0).any() or np.isnan(ball1).any():
            continue  # NaN would silently pass every `<` gate below
        dist = float(np.hypot(*(ball1 - ball0)))
        if dist < LONG_PASS_MIN:
            continue

        poss_team = int(match.side_id[pe.passer_idx])
        def_team = 1 - poss_team
        bounds = seg_bounds(match, pe.period)
        if bounds is None:
            continue
        lo, hi = bounds
        def_gk = team_gk_idx(match, def_team, lo, hi)
        own_gk = team_gk_idx(match, poss_team, lo, hi)
        if def_gk is None or own_gk is None:
            continue
        a_dir = attack_dir_for(match, own_gk, lo, hi)

        # the pass itself must go forward
        if a_dir * (ball1[0] - ball0[0]) <= 0:
            continue

        # defensive line at RELEASE -- this is what "in behind" means
        dline_x = defensive_line_x(match, def_team, f0, a_dir, def_gk)
        if np.isnan(dline_x):
            continue

        rec = receiver_idx(match, pm, pe, poss_team)
        if rec < 0:
            continue
        rec_release = match.positions[f0, rec]
        rec_reception = match.positions[f1, rec]
        if np.isnan(rec_release[0]) or np.isnan(rec_reception[0]):
            continue

        # receiver starts in front of (or level with) the line and runs beyond it
        if a_dir * (rec_release[0] - dline_x) > ONSIDE_TOL:
            continue  # already behind the line at release -> not a line-breaking run
        if a_dir * (ball1[0] - dline_x) < LINE_MARGIN:
            continue  # reception point not beyond the release-time line

        # receiver must advance forward, not just drift 4 m sideways
        run_fwd = float(a_dir * (rec_reception[0] - rec_release[0]))
        if run_fwd < RUN_MIN:
            continue

        run_dist = float(np.hypot(*(rec_reception - rec_release)))
        out.append((gid, pm, pe, dist, run_dist, rec))
    return out


if __name__ == "__main__":
    all_candidates = []
    for gid in available_game_ids()[:15]:  # widen this slice if nothing turns up
        all_candidates.extend(find_candidates(gid))
        if len(all_candidates) >= 5:
            break

    if not all_candidates:
        raise SystemExit("no long ball-in-behind found -- widen the game_id slice "
                         "or lower LONG_PASS_MIN/RUN_MIN")

    all_candidates.sort(key=lambda c: -c[3])  # longest qualifying pass first
    GAME_ID, pm, pe, dist, run_dist, receiver_pidx = all_candidates[0]
    match = pm.match
    frame = pe.release_g
    poss_team = int(match.side_id[pe.passer_idx])
    print(f"{len(all_candidates)} candidates")
    for gid, _pm, _pe, d, rd, rc in all_candidates:
        print(f"  game {gid}: pass {d:5.1f} m, receiver ran {rd:4.1f} m, "
              f"release frame {_pe.release_g}, receiver {_pm.match.player_name[rc]}")
    print(f"chosen: game {GAME_ID}, pass {dist:.1f} m, receiver ran {run_dist:.1f} m onto it, "
          f"frame {frame}, possession team {poss_team}, receiver idx {receiver_pidx}")
