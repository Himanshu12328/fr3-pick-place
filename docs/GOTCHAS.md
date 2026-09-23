<!-- Moved out of README.md. The README keeps the current result, the
     understanding needed to read it, and how to reproduce it; everything
     else lives here. -->

# Gotchas


Each of these cost real debugging time. All of them produced wrong results
that looked plausible, rather than raising an error.

**Evaluation resolution must match training resolution.** The vision
backbone is fully convolutional, so it accepts any input size without
complaint. But `crop_shape` is applied in pixels.

Rendering at 640 × 480 for a policy trained at 128 × 160 turns a
whole-scene 90% crop into a 115 × 144 patch of dead centre. The policy goes
effectively blind and runs on proprioception alone. This dropped measured
success from 60% to 0.4%. The tell was identical rollout distances across
different checkpoints and different architectures.

**MuJoCo's API moved.** `data.qM` was renamed to `data.M`. It still holds
the sparse packed lower triangle, not the dense matrix. And `mj_fullM`
changed signature from `(model, dst, src)` to `(model, data, dst)`. The
`mass_matrix()` function in `impedance.py` handles all three variants.

**`qfrc_bias` does not include passive joint forces.** It holds only
`C(q,q̇)q̇ + g(q)`. MuJoCo's joint damping and springs live in
`qfrc_passive` and get added separately. If you leave them uncompensated,
the model's damping stacks on top of the controller's, and the achieved
damping ratio is not the ζ you asked for.

**Dry friction cannot be cancelled feedforward.** Menagerie's FR3 sets about
6 Nm of total `frictionloss`. That produces a constant 6 N opposing force
and 22 mm of steady-state error at Kp=300.

It is resolved by the constraint solver, not readable as a force, so it gets
zeroed for controller characterisation. On real hardware it is real, and it
sets a minimum usable stiffness: `Kp × acceptable_error > 6 N`.

**DualSense mappings differ from the SDL documentation.** On this pad L1 and
R1 are buttons 9 and 10, not 4 and 5. Buttons 4 and 5 are Share and the PS
button.

Yaw also needs its sign flipped for front-of-arm operation. That is despite
the geometric argument that rotation should read the same from either
viewpoint. Verify both empirically with `test_pad.py` and `teleop_test.py`.

**Windows console encoding.** LeRobot prints policy names with non-cp1252
characters, which crashes the console. Set `PYTHONIOENCODING=utf-8`.

---

