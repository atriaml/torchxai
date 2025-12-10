import pytest

from tests.utils.mnist_train import mnist_trainer


@pytest.fixture()
def mnist_train_configuration():
    def _mnist_train_configuration(model_type: str, train_and_eval_model: bool):
        return mnist_trainer(model_type, train_and_eval_model)

    yield _mnist_train_configuration
