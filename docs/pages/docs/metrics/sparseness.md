---
title: Sparseness
---

# Sparseness

Computes the Gini index of attribution magnitudes (Chalasani et al.). Higher score means attributions are more concentrated on a small number of features, making the explanation simpler. Does not require a model forward pass or a baseline. ↑ better.

Includes `sparseness` (per-feature) and `sparseness_feature_grouped` (pooled over feature groups defined by a mask).

::: torchxai.metrics.complexity.sparseness
    options:
        docstring_options:
        ignore_init_summary: true
        docstring_section_style: list
        relative_crossrefs: true
        scoped_crossrefs: true
        signature_crossrefs: true
        unwrap_annotated: true
        filters: public
        inherited_members: true
        summary: true
        heading_level: 1
        parameter_headings: true
        type_parameter_headings: true
        show_root_heading: true
        show_root_full_path: false
        show_symbol_type_heading: true
        show_symbol_type_toc: true
        line_length: 88
        merge_init_into_class: true
        separate_signature: true
        show_signature_annotations: true
        show_signature_type_parameters: true
        backlinks: tree
        show_bases: false
        show_inheritance_diagram: true
