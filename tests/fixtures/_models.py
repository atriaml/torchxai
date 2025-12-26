import logging

import pytest
import torch

from tests.helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel7_ReluMultiTensor,
    BasicModel7_SumMultiTensor,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    ParkFunction,
)
from tests.helpers.classification_models import (
    SigmoidModel,
    SoftmaxModel,
    SoftmaxModelTupleInput,
)
from tests.utils.common import _set_all_random_seeds
from tests.utils.configs import BaseTestConfig
from tests.utils.types import ExplanationInputs, MetricInputs
from torchxai.data_types import ExplanationTarget

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_runtest_setup():
    _set_all_random_seeds(1234)


@pytest.fixture()
def park_function_configuration():
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)],
            inputs=(torch.tensor([[0.24, 0.48, 0.56, 0.99, 0.68, 0.86]]),),
        ),
        model=ParkFunction(),
        n_features=6,
    )


@pytest.fixture()
def basic_model_single_input_config():
    yield BaseTestConfig(
        model=BasicModel2(),
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)],
            inputs=(torch.tensor([3.0]), torch.tensor([1.0])),
        ),
        n_features=2,
    )


@pytest.fixture()
def basic_model_single_batched_input_config():
    yield BaseTestConfig(
        model=BasicModel2(),
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)],
            inputs=(torch.tensor([[3.0]]), torch.tensor([[1.0]])),
        ),
        n_features=2,
    )


@pytest.fixture()
def basic_model_batch_input_config():
    yield BaseTestConfig(
        model=BasicModel2(),
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(3)],
            inputs=(torch.tensor([3.0] * 3), torch.tensor([1.0] * 3)),
        ),
        n_features=2,
    )


@pytest.fixture()
def basic_model_batch_input_with_additional_forward_args_config():
    config = BaseTestConfig(
        model=BasicModel4_MultiArgs(),
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)],
            inputs=(torch.tensor([[1.5, 2.0, 3.3]]), torch.tensor([[3.0, 3.5, 2.2]])),
            additional_forward_args=(torch.tensor([[1.0, 3.0, 4.0]]),),
        ),
        n_features=6,
    )
    yield config


@pytest.fixture()
def classification_convnet_model_with_multiple_targets_config():
    yield BaseTestConfig(
        model=BasicModel_ConvNet_One_Conv(),
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(20)],
            inputs=(
                torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(
                    20, 1, 4, 4
                ),
            ),
            target=ExplanationTarget.from_raw_input(torch.tensor([1] * 20)),
        ),
        n_features=(1 * 4 * 4),
    )


@pytest.fixture()
def classification_multilayer_model_with_tuple_targets_config():
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(4)],
            inputs=(torch.arange(1.0, 13.0).view(4, 3).float(),),
            additional_forward_args=(torch.arange(1, 13).view(4, 3).float(), True),
            target=ExplanationTarget.from_raw_input(
                [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
            ),
        ),
        model=BasicModel_MultiLayer(),
        n_features=3,
    )


@pytest.fixture()
def classification_multilayer_model_with_baseline_and_tuple_targets_config():
    yield BaseTestConfig(
        model=BasicModel_MultiLayer(),
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(4)],
            inputs=(torch.arange(1.0, 13.0).view(4, 3).float(),),
            additional_forward_args=(torch.arange(1, 13).view(4, 3).float(), True),
            target=ExplanationTarget.from_raw_input(
                [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
            ),
            baselines=(torch.ones(4, 3),),
        ),
        metric_inputs=MetricInputs(baselines=torch.ones(4, 3)),
        n_features=3,
    )


@pytest.fixture()
def classification_sigmoid_model_single_input_single_target_config():
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)],
            inputs=(torch.tensor([[1.0] * 10]),),
            target=ExplanationTarget.from_raw_input(torch.tensor([1])),
        ),
        model=SigmoidModel(10, 20, 10),
        n_features=10,
    )


@pytest.fixture()
def classification_softmax_model_single_input_single_target_config():
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)], inputs=(torch.tensor([[1.0] * 10]),)
        ),
        model=SoftmaxModel(10, 20, 10),
        n_features=10,
    )


@pytest.fixture()
def classification_softmax_model_multi_input_single_target_config():
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(3)],
            inputs=(torch.tensor([[1.0] * 10] * 3),),
        ),
        model=SoftmaxModel(10, 20, 10),
        n_features=10,
    )


@pytest.fixture()
def classification_softmax_model_multi_tuple_input_single_target_config():
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(3)],
            inputs=(torch.tensor([[1.0] * 10] * 3), torch.tensor([[-1.0] * 10] * 3)),
            target=ExplanationTarget.from_raw_input(torch.tensor([1])),
        ),
        model=SoftmaxModelTupleInput(10, 20, 10),
        n_features=20,
    )


@pytest.fixture()
def classification_alexnet_model_single_sample_config():
    from torchvision.models import alexnet

    model = alexnet(pretrained=True)
    model.eval()
    model.zero_grad()
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)],
            inputs=(torch.randn(1, 3, 224, 224),),
            target=ExplanationTarget.from_raw_input(torch.tensor([1])),
        ),
        model=model,
        n_features=(3 * 224 * 224),
    )


@pytest.fixture()
def classification_alexnet_model_config():
    from torchvision.models import alexnet

    model = alexnet(pretrained=True)
    model.eval()
    model.zero_grad()
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(10)],
            inputs=(torch.randn(10, 3, 224, 224),),
            target=ExplanationTarget.from_raw_input(torch.tensor([1])),
        ),
        model=model,
        n_features=(3 * 224 * 224),
    )


@pytest.fixture()
def classification_alexnet_model_real_images_single_sample_config():
    from io import BytesIO

    import requests
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from torchvision.models import alexnet

    image_urls = [
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01440764_tench.JPEG?raw=true"
    ]
    labels = [0]

    images = []
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the image if needed
                transforms.ToTensor(),  # Convert to a tensor (normalizes pixel values to [0, 1])
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize pixel values to ImageNet values
            ]
        )
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(  # type: ignore
            0
        )  # Shape: [1, 3, 256, 256] for a batch of 1 image
        if images == []:
            images = image_tensor
        else:
            images = torch.cat((images, image_tensor), dim=0)  # type: ignore
    labels = torch.tensor(labels)
    model = alexnet(pretrained=True)
    model.eval()
    model.zero_grad()
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)], inputs=images, target=labels
        ),
        model=model,
        n_features=(3 * 224 * 224),
    )


@pytest.fixture()
def classification_alexnet_model_real_images_config():
    from io import BytesIO

    import requests
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from torchvision.models import alexnet

    image_urls = [
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01440764_tench.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01537544_indigo_bunting.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01641577_bullfrog.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01693334_green_lizard.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01819313_sulphur-crested_cockatoo.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01883070_wombat.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01990800_isopod.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02091467_Norwegian_elkhound.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02099429_curly-coated_retriever.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02113624_toy_poodle.JPEG?raw=true",
    ]
    labels = [0, 14, 30, 46, 89, 106, 126, 174, 206, 265]

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize the image if needed
            transforms.ToTensor(),  # Convert to a tensor (normalizes pixel values to [0, 1])
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize pixel values to ImageNet values
        ]
    )

    images = []
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image_tensor: torch.Tensor = tf(image)  # type: ignore
        image_tensor = image_tensor.unsqueeze(
            0
        )  # Shape: [1, 3, 256, 256] for a batch of 1 image
        images.append(image_tensor)

    images = torch.cat(images, dim=0)
    labels = torch.tensor(labels)
    model = alexnet(pretrained=True)
    model.eval()
    model.zero_grad()
    yield BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(10)],
            inputs=(images,),
            target=ExplanationTarget.from_raw_input(labels),
        ),
        model=model,
        n_features=(3 * 224 * 224),
    )


@pytest.fixture()
def multi_modal_sequence_sum():
    def test_sequence_tensor(size=12, embedding_size=3):
        return (
            torch.tensor([0] + list(range(3, size + 3)) + [1, 2])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, embedding_size, size + 3)
            .repeat(1, 1, 1)
            .permute(0, 2, 1)
        ).float()

    def test_image(size=9):
        return (
            torch.arange(size)
            .view(1, 1, 3, 3)
            .repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
            .float()
        )

    size = 6
    sequence1 = test_sequence_tensor(size)
    sequence2 = test_sequence_tensor(size) + size + 3
    sequence3 = test_sequence_tensor(size) + (size + 3) * 2
    image1 = test_image() + (size + 3) * 3
    feature_mask = (
        sequence1.clone().long(),
        sequence2.clone().long(),
        sequence3.clone().long(),
        image1.clone().long(),
    )
    frozen_features = torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])
    n_features = (
        torch.cat([x.flatten() for x in feature_mask]).unique().numel()
        - frozen_features.numel()
    )
    inputs = (sequence1, sequence2, sequence3, image1)
    total_sum = sum(x.sum() for x in inputs)
    inputs = tuple(x / total_sum for x in inputs)
    target = None

    yield BaseTestConfig(
        model=BasicModel7_SumMultiTensor(),
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)],
            inputs=inputs,
            target=ExplanationTarget.from_raw_input(target),
            feature_mask=feature_mask,
            baselines=tuple(torch.zeros_like(x) for x in inputs),
            frozen_features=[torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])],
        ),
        metric_inputs=MetricInputs(
            baselines=tuple(torch.zeros_like(x) for x in inputs),
            feature_mask=feature_mask,
        ),
        n_features=n_features,
    )


@pytest.fixture()
def multi_modal_sequence_relu():
    def test_sequence_tensor(size=12, embedding_size=4):
        return (
            torch.tensor([0] + list(range(3, size + 3)) + [1, 2])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, embedding_size, size + 3)
            .repeat(1, 1, 1)
            .permute(0, 2, 1)
        ).float()

    def test_image(size=9):
        return (
            torch.arange(size)
            .view(1, 1, 3, 3)
            .repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
            .float()
        )

    size = 6
    sequence1 = test_sequence_tensor(size)
    sequence2 = test_sequence_tensor(size) + size + 3
    sequence3 = test_sequence_tensor(size) + (size + 3) * 2
    image1 = test_image() + (size + 3) * 3
    feature_mask = (
        sequence1.clone().long(),
        sequence2.clone().long(),
        sequence3.clone().long(),
        image1.clone().long(),
    )
    frozen_features = torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])
    n_features = (
        torch.cat([x.flatten() for x in feature_mask]).unique().numel()
        - frozen_features.numel()
    )
    inputs = (sequence1, sequence2, sequence3, image1)
    mean = torch.cat(tuple(x.flatten() for x in inputs)).mean()
    std = torch.cat(tuple(x.flatten() for x in inputs)).std()
    inputs = tuple((x - mean) / std for x in inputs)
    target = None

    yield BaseTestConfig(
        model=BasicModel7_ReluMultiTensor(),
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(1)],
            inputs=inputs,
            target=ExplanationTarget.from_raw_input(target),
            feature_mask=feature_mask,
            baselines=tuple(torch.zeros_like(x) for x in inputs),
            frozen_features=[torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])],
        ),
        metric_inputs=MetricInputs(
            baselines=tuple(torch.zeros_like(x) for x in inputs),
            feature_mask=feature_mask,
        ),
        n_features=n_features,
    )
