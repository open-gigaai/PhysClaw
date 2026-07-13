#!/usr/bin/env python3
# -- coding: UTF-8
"""Judge recorded episodes as good / marginal / bad from their stats JSONs.

episode_stats.py measures; this script judges. Two kinds of rules:

  absolute sanity (bad):
    - no readable frames / unreadable parts
    - shorter than --min-frames / --min-duration
    - static arm: no arm joint's std exceeds --min-arm-std (recorder ran but
      the robot never moved -- classic dead-topic failure)
    - dead camera: mean brightness below --dark-brightness
    - >= --bad-discontinuities joint jumps (>0.10 rad/frame each)

  dataset-relative outliers (marginal):
    - rms jerk > --jerk-factor x dataset median
    - camera sharpness < --sharp-factor x that camera's dataset median
      (catches a uniformly blurred lens that per-episode blur_frac misses)
    - blur_frac / over- / under-exposure fractions above thresholds
    - 1..N discontinuities, gripper never actuated

Task outcome (script/vision success) is reported alongside but does NOT
affect the recording-quality verdict: a failed grasp can still be a perfectly
recorded episode.

Run AFTER episode_stats.py --all (this reads only the stats JSONs).
Stdlib only -- runs on plain python3, no conda needed.

Usage:
  python data_quality_check.py --dataset-dir D [thresholds...]

Output: human table + greppable lines
  episode_verdict: <task> <session> <good|marginal|bad> [reason; reason...]
  quality_summary: good=N marginal=M bad=K
and <dataset>/stats/quality_report.json.
"""
import argparse
import glob
import json
import os
import sys
import time

LEFT_ARM = list(range(0, 6))
RIGHT_ARM = list(range(7, 13))


def median(vals):
    s = sorted(v for v in vals if v is not None)
    if not s:
        return None
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def load_all_stats(dataset_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(dataset_dir, "*", "*", "stats.json"))):
        task_entry = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if task_entry in ("stats", "training") or task_entry.startswith("."):
            continue
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            print("Warning: unreadable %s" % p, file=sys.stderr)
    return out


def baselines(all_stats):
    """Dataset-relative reference points (medians are robust to the very
    outliers we are hunting)."""
    jerks, sharp = [], {}
    for s in all_stats:
        sm = s.get("smoothness") or {}
        jerks.append((sm.get("rms_jerk") or {}).get("overall"))
        cams = (s.get("image_quality") or {}).get("cameras") or {}
        for cam, c in cams.items():
            sharp.setdefault(cam, []).append((c.get("sharpness") or {}).get("mean"))
    return {"median_jerk": median(jerks),
            "median_sharpness": {cam: median(v) for cam, v in sharp.items()}}


def judge(s, base, args):
    """Returns (verdict, [reasons], outcome_str)."""
    bad, marginal = [], []
    res = s.get("result") or {}
    outcome = "%s/%s" % (res.get("script") or "?", res.get("vision") or "-")

    frames = s.get("frames") or 0
    if frames <= 0:
        return "bad", ["no readable frames (recorder captured nothing)"], outcome
    if frames < args.min_frames:
        bad.append("only %d frames (< %d)" % (frames, args.min_frames))
    if (s.get("duration_s") or 0) < args.min_duration:
        bad.append("duration %.2fs < %.1fs" % (s.get("duration_s") or 0,
                                               args.min_duration))
    for w in s.get("warnings") or []:
        if "unreadable part" in w:
            bad.append(w)

    # --- action signal: did the arm actually do anything? -------------------
    act = s.get("action") or {}
    pd = act.get("per_dim") or {}
    stds = pd.get("std")
    if stds:
        arm_std = max(stds[d] for d in LEFT_ARM + RIGHT_ARM if stds[d] is not None)
        if arm_std < args.min_arm_std:
            bad.append("static arm: max joint std %.4f rad (< %.3f) -- "
                       "variance ~0, nothing to learn" % (arm_std, args.min_arm_std))
    grip = act.get("gripper") or {}
    if args.gripper_check and grip:
        actuated = any((grip.get(side) or {}).get("actuated") for side in ("left", "right"))
        if not actuated:
            marginal.append("gripper never actuated (no open/close in a grasp episode)")

    # --- smoothness ----------------------------------------------------------
    sm = s.get("smoothness") or {}
    disc = sm.get("discontinuity_count")
    if disc is not None:
        if disc >= args.bad_discontinuities:
            bad.append("%d discontinuities >0.10 rad/frame" % disc)
        elif disc >= args.marginal_discontinuities:
            marginal.append("%d discontinuit%s" % (disc, "y" if disc == 1 else "ies"))
    jerk = (sm.get("rms_jerk") or {}).get("overall")
    mj = base["median_jerk"]
    if jerk is not None and mj not in (None, 0) and jerk > args.jerk_factor * mj:
        marginal.append("rms jerk %.1f = %.1fx dataset median (rough motion)"
                        % (jerk, jerk / mj))

    # --- cameras -------------------------------------------------------------
    cams = (s.get("image_quality") or {}).get("cameras") or {}
    for cam in sorted(cams):
        c = cams[cam]
        bright = (c.get("brightness") or {}).get("mean")
        if bright is not None and bright < args.dark_brightness:
            bad.append("%s dead/dark (brightness %.0f)" % (cam, bright))
            continue
        sh = (c.get("sharpness") or {}).get("mean")
        ms = base["median_sharpness"].get(cam)
        if sh is not None and ms not in (None, 0) and sh < args.sharp_factor * ms:
            marginal.append("%s blurry vs dataset (sharpness %.2fx median)"
                            % (cam, sh / ms))
        if (c.get("blur_frac") or 0) > args.blur_bad:
            bad.append("%s blur in %d%% of frames" % (cam, 100 * c["blur_frac"]))
        elif (c.get("blur_frac") or 0) > args.blur_marginal:
            marginal.append("%s blur in %d%% of frames" % (cam, 100 * c["blur_frac"]))
        if (c.get("overexposed_frac") or 0) > args.exposure_marginal:
            marginal.append("%s overexposed (%.0f%% blown pixels)"
                            % (cam, 100 * c["overexposed_frac"]))
        if (c.get("underexposed_frac") or 0) > args.exposure_marginal:
            marginal.append("%s underexposed (%.0f%% black pixels)"
                            % (cam, 100 * c["underexposed_frac"]))

    if bad:
        return "bad", bad + marginal, outcome
    if marginal:
        return "marginal", marginal, outcome
    return "good", [], outcome


def main(argv=None):
    ap = argparse.ArgumentParser(description="Judge episode recording quality "
                                             "from stats JSONs.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--min-arm-std", type=float, default=0.01,
                    help="rad; below this on every arm joint = static arm")
    ap.add_argument("--bad-discontinuities", type=int, default=3)
    ap.add_argument("--marginal-discontinuities", type=int, default=1)
    ap.add_argument("--jerk-factor", type=float, default=3.0)
    ap.add_argument("--dark-brightness", type=float, default=10.0,
                    help="mean gray below this = dead/dark camera")
    ap.add_argument("--sharp-factor", type=float, default=0.3)
    ap.add_argument("--blur-bad", type=float, default=0.6)
    ap.add_argument("--blur-marginal", type=float, default=0.3)
    ap.add_argument("--exposure-marginal", type=float, default=0.2)
    ap.add_argument("--no-gripper-check", dest="gripper_check",
                    action="store_false")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    dataset_dir = os.path.abspath(args.dataset_dir)
    all_stats = load_all_stats(dataset_dir)
    if not all_stats:
        print("Error: no stats JSONs under %s -- run episode_stats.py --all first"
              % dataset_dir, file=sys.stderr)
        return 1
    base = baselines(all_stats)

    rows, counts = [], {"good": 0, "marginal": 0, "bad": 0}
    for s in sorted(all_stats, key=lambda x: (x.get("task") or "",
                                              x.get("session") or "")):
        verdict, reasons, outcome = judge(s, base, args)
        counts[verdict] += 1
        rows.append({"task": s.get("task"), "session": s.get("session"),
                     "verdict": verdict, "reasons": reasons, "outcome": outcome,
                     "frames": s.get("frames"),
                     "smoothness_score": (s.get("smoothness") or {}).get("score"),
                     "aesthetics": (s.get("aesthetics") or {}).get("score")})
        print("episode_verdict: %s %s %s%s"
              % (s.get("task"), s.get("session"), verdict,
                 (" [" + "; ".join(reasons) + "]") if reasons else ""))

    report = {"schema_version": 1,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "dataset_dir": dataset_dir,
              "baselines": base,
              "thresholds": {k: v for k, v in vars(args).items()
                             if k not in ("dataset_dir", "quiet")},
              "counts": counts, "episodes": rows}
    out = os.path.join(dataset_dir, "stats", "quality_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, separators=(",", ":"))
    os.replace(tmp, out)

    print("quality_summary: good=%d marginal=%d bad=%d"
          % (counts["good"], counts["marginal"], counts["bad"]))
    if not args.quiet:
        print("report: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
