import logging

from tests.utils.common import _set_all_random_seeds

from .fixtures._explainer import *  # noqa: F403, F401
from .fixtures._metric import *  # noqa: F403, F401
from .fixtures._models import *  # noqa: F403, F401
from .fixtures._trainers import *  # noqa: F403, F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_runtest_setup():
    _set_all_random_seeds(1234)
