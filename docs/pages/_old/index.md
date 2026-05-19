---
title: TorchXAI
sidebar_title: Home
show_title: true
order: 0
external_links:
  "github": https://github.com/saifullah3396/torchxai
---

TorchXAI is a lightweight PyTorch toolkit for evaluating machine learning models using explainability techniques. It offers efficient implementations of explainability metrics that integrate seamlessly with the Captum ecosystem, with a focus on batch computation and task/data-agnostic evaluation to make scalable XAI evaluation easy.
<!-- 
!!! info "Migrating from mkdocs-material? "
    In the general case, I would advise **not migrating** from [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) since this theme is well established, very mature, with a lot of features we love.

    You can easily migrate to this theme if your documentaion does not rely on too many extensions/plugins. Of course, this is also a good choice for your next brand new project!
   -->
## Why TorchXAI?

**Designed for XAI evaluation** — gives you ready-to-use metrics to quantify explanation quality (for example completeness and other axiomatic metrics).

**Captum-compatible** — works alongside Captum explainers so you can compute metrics on attributions you already produce.

**Batch & scalable** — implementations aim to be efficient for dataset-scale evaluation so you can compare explainers across many inputs.

## Who is this for?

- ML researchers and engineers who want quantitative ways to evaluate explanation methods.
- People already using Captum who want plug-and-play metrics for large datasets.
- Anyone comparing multiple explainers and needing standardized evaluation metrics.
