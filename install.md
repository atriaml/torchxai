---
title: Installation
show_datetime: true
---

## Quick Install

The PyPI distribution is named **`torchxai-tools`**; the Python import name is `torchxai`.

=== "pip"

        :::bash
        pip install torchxai-tools

=== "uv"

        :::bash
        uv add torchxai-tools

=== "poetry"

        :::bash
        poetry add torchxai-tools

After installation:

```python
from torchxai.explainers import SaliencyExplainer   # import name is torchxai
```

---

## Optional extras

GPU-accelerated image examples require torchvision:

=== "pip"

        :::bash
        pip install "torchxai-tools[vision]"

=== "uv"

        :::bash
        uv add "torchxai-tools[vision]"

---

## Install from source

To install the latest development version directly from GitHub:

=== "pip"

        :::bash
        pip install git+https://github.com/saifullah3396/torchxai.git

=== "uv"

        :::bash
        uv add git+https://github.com/saifullah3396/torchxai.git

=== "Clone & install"

        :::bash
        git clone https://github.com/saifullah3396/torchxai.git
        cd torchxai
        pip install -e .
