<!-- Moved out of README.md. The README keeps the current result, the
     understanding needed to read it, and how to reproduce it; everything
     else lives here. -->

# Roadmap


* [x] Cartesian impedance controller, characterised on the full scene
* [x] DualSense teleoperation and demonstration collection
* [x] Evaluation harness, validated against recorded demonstrations
* [x] ACT and Diffusion Policy on 71 episodes, indistinguishable
* [x] Data scaling to 151 episodes. ACT **77.2% ± 5.0%**, Diffusion
      **71.3% ± 5.9%**, separated at 600 trials per policy
* [x] Selection bias quantified across four runs
* [x] Targeted collection at x < 0.53 m, where both policies were 14 to 21
      points weaker. Confirmed a coverage gap, not a control one, for both:
      near-region success went 67.9% → 87.8% for ACT and 57.9% → 82.4% for
      diffusion (221 episodes, six seeds each)
* [x] Stage 0. Measured the environment ceiling and classified the
      failures. The oracle scores 600/600, so the whole 16.5 point gap is
      learnable. ACT's failures are 62% missed grasps, 18% knocked away,
      20% dropped, and zero near misses
* [x] Stage 1. Temporal ensembling and action chunk length sweep. No
      configuration beat the baseline. Success collapses monotonically as
      the chunk shortens: 88, 85, 61, 26, 0 percent at 32, 16, 8, 4, 1
      action steps. The policy depends on open-loop chunk commitment
* [x] Stage 2. Teacher, at 100% over 600 trials and 0.991 trajectory
      similarity to the demonstrations. Eleven runs of RL produced nothing
      better than the scripted oracle from Stage 0, so the oracle is the
      teacher. Five reward exploits along the way, four of them reporting
      over 99% while shoving, hovering or dropping the block. See
      docs/RL_PROCESS.md
* [x] Stage 3. Distil the teacher into a vision only student with DAgger.
      200 oracle-labelled episodes on the student's own state distribution,
      merged with the 221 recorded ones. **76% against the 83.5% baseline,
      so it made the policy worse.** Plateaued from 15k to 30k steps
* [x] Stage 3b. Oracle demonstrations at scale. 800 machine-generated
      demonstrations mixed with the 221 recorded ones, 1,006 episodes and
      298k frames. **86.8% +- 3.7% over 600 trials against the 83.5%
      baseline**, the first approach of four that did not land below it.
      Predicted 88 to 93% beforehand, so it came in just under
* [x] Stage 3c. **The success criterion was replaced.** `check_success` is
      satisfied by a shove, a drop and a hover, which this project had
      already proven five times against RL teachers and never applied to
      the vision student. `src/eval/strict.py` scores the whole specified
      sequence against nine gates calibrated so all 221 recorded
      demonstrations pass. Validated both ways: the oracle scores 100%, an
      oracle at 3x speed scores 100% loose and **0% strict**. Every earlier
      number in this file is a loose number. See docs/STRICT_EVAL.md
* [x] Stage 3d. **The training labels were wrong.** Oracle collection had
      been using the reactive DAgger labeller, whose fixed 12 mm target
      lead produces one constant speed, so the demonstrations'
      slow-fast-slow profile was absent and the retreat ran 1.7x too slow.
      The dataset scored 0.832 against its own trajectory profile where the
      demonstrations score 0.994. Re-collected with the rate-limited phase
      machine: **0.974**. Also fixed a second bug where the phase machine
      ignored every jitter field, making it a deterministic demonstrator
* [x] Stage 3e. ACT on 997 corrected oracle-only episodes.
      **92.3% +- 2.6% strict and 96.7% loose over 600 trials** on seeds
      never used for selection, against the previous best policy's 72.5%
      strict. Mean lift 80 mm against the demonstrations' 80, block moved
      0.22 mm after release, far-region deficit gone
* [x] Stage 3f. Two controlled negative results. Replanning more often is
      monotonically worse (92.5, 85.0, 32.5 percent at 32, 16, 8 action
      steps). DAgger costs 27 points: 285 on-policy episodes added to the
      997 took the best checkpoint from 92.5% to 65.0% under an otherwise
      identical run, after two labelling bugs were fixed and the dataset
      passed every composition check
* [x] Stage 4. Closed the gap to 97%. Three architecture and horizon
      variants all plateaued at 90 to 93 percent, so capacity and chunk
      length are not the constraint. What worked was that two of them fail
      in *different places* — ResNet18 is 6.8 points better near the base,
      ResNet34 5.4 points better far from it, on identical data — so
      averaging their target poses, and voting rather than averaging the
      binary gripper, removes the regional weakness entirely.
      **97.0% strict over 1,200 trials, 95% CI 96.0 to 98.0**
* [x] Stage 5. Reported over twelve seeds of 100 trials, six of them
      contaminated only by having suggested the experiment and six never
      used for anything. The two sets differ by 1.0 point, which is noise;
      for contrast the chunk-64 experiment shrank by 6.0 points between its
      screening and reporting seeds and reversed sign
* [x] Stage 6. **Diagnosed the residual 3%, and ruled out the runtime fix
      for it.** The teacher was remeasured at **99.7% strict over 1,000
      trials**, not the 97.5% a 40-trial sample had reported, so the
      student is 2.7 points behind it and the gap is the student's. The
      residual failure is a wrist-yaw error that is *proportional* to the
      rotation the block requires — `error = 0.115 × offset`, holding
      across every bin from 0 to 45 degrees — which jams the descent on the
      block's top face. A runtime layer that refuses the mistimed close and
      forces a replan was built and measured in six configurations: **all
      six score 97.00% and fail the same six trials.** See
      docs/PATH_TO_99.md
* [ ] Beat the teacher. The 2.7 points to the demonstrator are the
      student's to close, and the levers that have been ruled out by
      measurement now are: a runtime supervisor (six configurations, no
      change), a third ensemble member (97.0% → 92.0%, a regression), more
      frequent or less frequent replanning, DAgger, a bigger backbone, and
      higher image resolution for block *position*. What has not been tried
      is a policy whose orientation target is not a discontinuous function
      of a symmetric object's pose
* [x] A multimodal task variant — **it turns out the task was already
      multimodal and nobody had noticed.** A cube is symmetric every 90
      degrees, so four wrist orientations grasp it equally well and the
      absolute yaw label is a sawtooth in the block's yaw. This is the axis
      ACT and diffusion were designed to differ on, and it has been present
      in the single-mode task from the beginning
* [ ] π0 fine-tuning
* [ ] Sim-to-real transfer

---

