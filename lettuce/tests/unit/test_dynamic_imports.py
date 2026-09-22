import sys
import subprocess
import pytest

from options.base_options import InferenceType, BaseOptions

settings = BaseOptions()

not_using_local_weights = settings.inference_type != InferenceType.LLAMA_CPP


def test_llama_not_loaded_subprocess():
    # Because pytest pulls in the local_models stuff, this has to run in its own subprocess
    # or local_models is in sys.modules
    # Unfortunately, haystack pulls in llama_cpp stuff if it's installed,
    # whether you like it or not. Obviously this is a good design for them,
    # but it's inconvenient, as this test doesn't work
    # llama_cpp_modules = [x for x in sys.modules.keys() if "llama_cpp" in x]
    # print(llama_cpp_modules)
    #
    # assert(len(llama_cpp_modules) == 0)
    # and we just have to test this instead:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from components.models import get_model; "
            "import sys; "
            "assert 'components.models.local_models' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not_using_local_weights, reason="Not using local weights")
def test_llama_loaded():
    from components.models import local_models

    llama_cpp_modules = [x for x in sys.modules.keys() if "llama_cpp" in x]

    assert len(llama_cpp_modules) > 0
