# fc_dec_transp


---

# Installation Guide

Follow these steps to set up and use this package in a local virtual environment.

---

### 1. Clone the repository

```bash
git clone git@github.com:nicdeca/fc_dec_transp.git
cd fc_dec_transp
```

---

### 2. Create a virtual environment

Create a local Python virtual environment:

```bash
python3 -m venv .venv
```

---

### 3. Activate the virtual environment

* **On Linux/macOS:**

  ```bash
  source .venv/bin/activate
  ```

* **On Windows (PowerShell):**

  ```bash
  .venv\Scripts\Activate.ps1
  ```

You should now see `(.venv)` in your terminal prompt.

---

### 4. Install the package in editable mode

From the repository root (where `setup.py` is located):

```bash
pip install -e .
```

This installs the package in *editable mode*, meaning that any change to the source files in the `fc/` folder is immediately reflected when importing the package.

---

### 5. Verify the installation

Open a Python shell or run your example script to verify the installation:

```bash
python
>>> from fc.dec_controller import config_controller
>>> from fc.flycrane_utils import flycrane_utils
```

Or test the provided example:

```bash
python examples/sim_fc_dec_dynalloc.py
```

---

## Dependencies

Dependencies are listed in `setup.py` under `install_requires`.
To install everything automatically, simply run:

```bash
pip install -e .
```

If you also include a `requirements.txt`, you can install dependencies separately:

```bash
pip install -r requirements.txt
```

---

### 🧹 Deactivating the virtual environment

When you are done, deactivate the environment with:

```bash
deactivate
```
