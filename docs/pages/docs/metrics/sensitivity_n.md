# Sensitivity-N

Wraps infidelity with a random n-feature perturbation function: it randomly zeroes out n features at a time and measures the Pearson correlation between attributions and the resulting output changes (Ancona et al.). Lower score indicates the attributions better predict which features matter. Supports feature masks and multi-target mode.

::: torchxai.metrics.faithfulness.sensitivity_n
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
