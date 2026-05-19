1. Make an abstract port of attention-explainers from atria-insights. For the time being it uses TransformersEncoderModel and depends on it to get attention outputs, but essentially the two can be easily decoupled into separate modules and there we can simply check for dict or 'attention' in the output to apply the forward wrappers.
References:
https://github.com/atriaml/atria/tree/main/atria_insights/src/atria_insights/explainers
https://github.com/atriaml/atria/blob/main/atria_insights/src/atria_insights/model_pipelines/_forward_wrappers/_sequence_forward_wrappers.py
