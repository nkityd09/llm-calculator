import torch
from accelerate.commands.estimate import check_has_model, create_empty_model
from urllib.parse import urlparse
from accelerate.utils import calculate_maximum_sizes
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
import streamlit as st

DTYPE_MODIFIER = {"float32": 1, "float16/bfloat16": 2, "int8": 4, "int4": 8}

def translate_llama2(text):
    "Translates llama-2 to its hf counterpart"
    if not text.endswith("-hf"):
        return text + "-hf"
    return text

def get_model(model_name: str, library: str, access_token: str):
    "Finds and grabs model from the Hub, and initializes on `meta`"
    if "meta-llama" in model_name:
        model_name = translate_llama2(model_name)
    if library == "auto":
        library = None
    model_name = extract_from_url(model_name)
    try:
        model = create_empty_model(model_name, library_name=library, trust_remote_code=True, access_token=access_token)
    except GatedRepoError:
        st.error(
            f"Model `{model_name}` is a gated model, please ensure to pass in your access token and try again if you have access. You can find your access token here : https://huggingface.co/settings/tokens. "
        )
        st.stop()
    except RepositoryNotFoundError:
        st.error(f"Model `{model_name}` was not found on the Hub, please try another model name.")
        st.stop()
    except ValueError:
        st.error(
            f"Model `{model_name}` does not have any library metadata on the Hub, please manually select a library_name to use (such as `transformers`)"
        )
        st.stop()
    except (RuntimeError, OSError) as e:
        library = check_has_model(e)
        if library != "unknown":
            st.error(
                f"Tried to load `{model_name}` with `{library}` but a possible model to load was not found inside the repo."
            )
            st.stop()
        st.error(
            f"Model `{model_name}` had an error, please open a discussion on the model's page with the error message and name: `{e}`"
        )
        st.stop()
    except ImportError:
        # hacky way to check if it works with `trust_remote_code=False`
        model = create_empty_model(
            model_name, library_name=library, trust_remote_code=False, access_token=access_token
        )
    except Exception as e:
        st.error(
            f"Model `{model_name}` had an error, please open a discussion on the model's page with the error message and name: `{e}`"
        )
        st.stop()
    return model

def extract_from_url(name: str):
    "Checks if `name` is a URL, and if so converts it to a model name"
    is_url = False
    try:
        result = urlparse(name)
        is_url = all([result.scheme, result.netloc])
    except Exception:
        is_url = False
    # Pass through if not a URL
    if not is_url:
        return name
    else:
        path = result.path
        return path[1:]

def calculate_memory(model: torch.nn.Module, options: list):
    "Calculates the memory usage for a model init on `meta` device"
    total_size, largest_layer = calculate_maximum_sizes(model)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data = []
    for dtype in options:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]

        modifier = DTYPE_MODIFIER[dtype]
        dtype_total_size /= modifier
        dtype_largest_layer /= modifier

        dtype_training_size = dtype_total_size * 4 / (1024**3)
        dtype_inference = dtype_total_size * 1.2  / (1024**3)
        dtype_total_size = dtype_total_size  / (1024**3)
        data.append(
            {
                "dtype": dtype,
                "Total Size (GB)": dtype_total_size,
                "Inference (GB)" : dtype_inference,
                "Training using Adam (GB)": dtype_training_size,
                "Parameters (Billion)" : num_parameters / 1e9
            }
        )
    return data