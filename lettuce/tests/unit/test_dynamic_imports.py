import sys
import pytest

from options.base_options import InferenceType, BaseOptions
settings = BaseOptions()

not_using_local_weights = settings.inference_type != InferenceType.LLAMA_CPP

@pytest.mark.skipif(not_using_local_weights, reason="Not using local weights")
def test_llama_loaded():
    from components.models import local_models

    llama_cpp_modules = [x for x in sys.modules.keys() if "llama_cpp" in x]

    assert(len(llama_cpp_modules) > 0)

def test_llama_not_loaded():
    from components.models import get_model

    llama_cpp_modules = [x for x in sys.modules.keys() if "llama_cpp" in x]
    print(llama_cpp_modules)

    assert(len(llama_cpp_modules) == 0)
