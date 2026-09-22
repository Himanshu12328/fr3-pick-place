<!-- Moved out of README.md. The README keeps the current result, the
     understanding needed to read it, and how to reproduce it; everything
     else lives here. -->

# Requirements and setup, in full

The README carries the short path. This is every step, including the
Windows-specific ones that are easy to miss and expensive to diagnose.


* Windows 10 or 11
* An NVIDIA GPU. Built and tested on an RTX 5080 (Blackwell, sm_120)
* Python 3.11
* A DualSense controller, if you want to collect new demonstrations
* About 80 GB free disk for a 150-episode dataset at 640 × 480

---


## Setup


### 1. Python

Install Python 3.11 from python.org. Do not use the Microsoft Store build.
It sandboxes paths and breaks venv activation. Check "Add python.exe to
PATH" during install.

### 2. Clone and create the environment

```powershell
git clone https://github.com/<your-username>/fr3-pick-place.git
cd fr3-pick-place
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. Install PyTorch first, on its own

Install PyTorch before anything else. That way no other package can pull in
a wrong CUDA build as a dependency.

```powershell
pip install --upgrade pip
pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu128
```

**Blackwell GPUs need the cu128 index.** Standard PyPI wheels have no sm_120
kernels. Verify both of these before you continue:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
python -c "import torch; a=torch.randn(4096,4096,device='cuda'); print((a@a).sum().item())"
```

The first must print `(12, 0)`. The second must print a number. If
`cuda.is_available()` returns True that does not prove the kernels exist.
Only the matmul proves it.

### 4. Install the project

```powershell
pip install -e ".[train]"
```

Confirm the whole stack imports together:

```powershell
python -c "import torch, numpy, mujoco, lerobot; print(torch.__version__, numpy.__version__, mujoco.__version__, lerobot.__version__)"
```

### 5. Get the robot model

```powershell
git clone https://github.com/google-deepmind/mujoco_menagerie
```

Or set `MENAGERIE_DIR` if you keep it somewhere else.

### 6. Turn on Developer Mode

Go to Settings, System, For developers, and turn **Developer Mode** on.
LeRobot writes a symlink when it saves a checkpoint, and Windows refuses
symlinks without this. Training will otherwise crash at the first
checkpoint, hours into the run.

### 7. Point Python at the discrete GPU

Go to Settings, System, Display, Graphics. Add a desktop app. Browse to
`.venv\Scripts\python.exe`. Set it to **High performance**. Also turn off
Windows Energy Saver, which quietly caps the GPU.

This step is not optional and it is not obvious. Without it MuJoCo's
offscreen renderer falls back to Microsoft's software rasterizer. That is
**48 ms per frame instead of 0.8 ms**, a 60× penalty.

The interactive viewer works fine either way, because windowed OpenGL takes
a different path. So a smooth viewer is not evidence that headless rendering
is on the GPU. Measure it.

### 8. Build the scene

```powershell
fr3-build-scene
python -m src.scripts.preview_cameras
```

This takes Menagerie's bare FR3 and adds a parallel-jaw gripper, a table, a
graspable block, and three cameras. It writes
`models/fr3_pick_place.xml`. The verify output should say
`nq=16 nv=15 nu=7`.

---

