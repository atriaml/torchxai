import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple

from torchxai.data_types import TensorOrTupleOfTensorsGeneric


def _attribution_localization_impl(
    attributions: TensorOrTupleOfTensorsGeneric,
    feature_mask: TensorOrTupleOfTensorsGeneric,
    positive_attributions: bool = True,
    weighted: bool = False,
) -> TensorOrTupleOfTensorsGeneric:
    with torch.no_grad():
        is_attributions_tuple = _is_tuple(attributions)
        attributions = _format_tensor_into_tuples(attributions)
        feature_mask = _format_tensor_into_tuples(feature_mask)
        assert feature_mask[0].dtype == torch.bool, (
            "Segmentation mask must be of type bool."
        )
        assert (
            len(feature_mask) == len(attributions)
            and feature_mask[0].shape == attributions[0].shape
        ), "Segmentation mask must have the same shape as the attributions."

        if positive_attributions:
            attributions = tuple(
                torch.clamp(attribution, min=0) for attribution in attributions
            )
        print("attributions", attributions)

        bsz = attributions[0].shape[0]
        localization_scores = tuple(
            (attribution * mask).view(bsz, -1).sum(dim=1)
            / attribution.view(bsz, -1).sum(dim=1)
            for attribution, mask in zip(attributions, feature_mask, strict=False)
        )

        mask_size_ratios = tuple(
            mask.numel() / mask.contiguous().view(bsz, -1).sum(dim=1)
            for mask in feature_mask
        )

        if weighted:
            localization_scores = tuple(
                score * mask_size_ratio
                for score, mask_size_ratio in zip(
                    localization_scores, mask_size_ratios, strict=False
                )
            )

        localization_scores = _format_output(is_attributions_tuple, localization_scores)
        return localization_scores


def attribution_localization(
    attributions: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    feature_mask: tuple[torch.Tensor, ...] | None,
    multi_target: bool = False,
    positive_attributions: bool = True,
    weighted: bool = False,
    return_dict: bool = False,
) -> (
    TensorOrTupleOfTensorsGeneric
    | list[TensorOrTupleOfTensorsGeneric]
    | dict[str, TensorOrTupleOfTensorsGeneric]
    | dict[str, list[TensorOrTupleOfTensorsGeneric]]
):
    """
    Implementation of the Attribution Localization by Kohlbrenner et al., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Attribution Localization implements the ratio of positive attributions within the target to the overall
    attribution. High scores are desired, as it means, that the positively attributed pixels belong to the
    targeted object class.

    References:
        1) Max Kohlbrenner et al., "Towards Best Practice in Explaining Neural Network Decisions with LRP."
        IJCNN (2020): 1-7.

    Args:
        attributions (Tuple[Tensor,...] | list[tuple[torch.Tensor, ...]]): A tuple of tensors or a list of tuples
            of tensors representing attributions of separate inputs. Each tensor in the tuple has shape
            (batch_size, num_features). If the list is passed, it is assumed that each element in the list corresponds
            to the attributions of each batch with respect to a single target.
            For example:
                - Single target: [ (Tensor(batch_size, num_features), ) ]
                - Multi target: [ (Tensor(batch_size, num_features), ) for t1, (Tensor(batch_size, num_features), ),  for t1 ... ]
        multi_target (bool, optional): If True, it indicates that the attributions correspond to multiple targets per input.
            Default is False. This flag must be explicitely passed when passing a list of attributions batches.
        feature_mask (Tuple[Tensor,...]): A tuple of boolean mask tensors
            representing the desired segmented region mask for each input attribution where its localization
            is to be measured. Each tensor in the tuple has shape (batch_size, num_features). Whether a list is passed
            or a single tuple, the feature mask remains the same for all targets. And corresponds to the input samples.
        positive_attributions (bool, optional): If True, only positive attributions are considered
            for computing the localization score. Default is True.
        weighted (bool, optional): If True, the metric is weighted by the ratio of the total mask size and the
            size of segmented region.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
            with keys as the metric names and values as the corresponding metric outputs.
            Default is False.

    Returns:
        Tensor: A tensor of scalar complexity scores per
                input example. The first dimension is equal to the
                number of examples in the input batch and the second
                dimension is one.
    Examples::
        # ImageClassifier takes a single input tensor of images Nx3x32x32,
        # and returns an Nx10 tensor of class probabilities.
        net = ImageClassifier()
        saliency = Saliency(net)
        input = torch.randn(2, 3, 32, 32, requires_grad=True)
        baselines = torch.zeros(2, 3, 32, 32)
        # Computes saliency maps for class 3.
        attribution = saliency.attribute(input, target=3)
        # define a perturbation function for the input

        # Computes the monotonicity correlation and non-sensitivity scores for saliency maps
        attribution_localization_score = attribution_localization(
            attribution, feature_mask
        )
    """
    if feature_mask is None:
        feature_mask = tuple(torch.ones_like(attributions[0][0], dtype=torch.bool))

    if multi_target:
        assert isinstance(attributions, list), (
            "For multi-target attributions, a list of attributions per target must be provided."
        )
        scores_per_target = [
            _attribution_localization_impl(
                attributions=attribution_per_sample,
                feature_mask=feature_mask,
                positive_attributions=positive_attributions,
                weighted=weighted,
            )
            for attribution_per_sample in attributions
        ]
        if return_dict:
            return {"score": scores_per_target}
        else:
            return scores_per_target
    else:
        assert isinstance(attributions, tuple | torch.Tensor), (
            "For single-target attributions, a tuple or list of attributions must be provided."
        )
        assert all(isinstance(t, torch.Tensor) for t in attributions), (
            "Attributions must be a tensor or a tuple of tensors."
        )
        score = _attribution_localization_impl(
            attributions=attributions,
            feature_mask=feature_mask,
            positive_attributions=positive_attributions,
            weighted=weighted,
        )
        return {"score": score} if return_dict else score
