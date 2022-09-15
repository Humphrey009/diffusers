# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pickle import UnpicklingError
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from requests import HTTPError

from .modeling_utils import WEIGHTS_NAME
from .utils import CONFIG_NAME, DIFFUSERS_CACHE, HUGGINGFACE_CO_RESOLVE_ENDPOINT, logging


FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"

logger = logging.get_logger(__name__)


class FlaxModelMixin:
    r"""
    Base class for all flax models.

    [`FlaxModelMixin`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models.
    """
    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _flax_internal_args = ["name", "parent"]

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

    def _cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.dtype, mask: Any = None) -> Any:
        """
        Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.
        """

        # taken from https://github.com/deepmind/jmp/blob/3a8318abc3292be38582794dbf7b094e6583b192/jmp/_src/policy.py#L27
        def conditional_cast(param):
            if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.floating):
                param = param.astype(dtype)
            return param

        if mask is None:
            return jax.tree_map(conditional_cast, params)

        flat_params = flatten_dict(params)
        flat_mask, _ = jax.tree_flatten(mask)

        for masked, key in zip(flat_mask, flat_params.keys()):
            if masked:
                param = flat_params[key]
                flat_params[key] = conditional_cast(param)

        return unflatten_dict(flat_params)

    def to_bf16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast
        the `params` in place.

        This method can be used on TPU to explicitly convert the model parameters to bfloat16 precision to do full
        half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip.

        Examples:

        ```python
        >>> from diffusers import FlaxUNet2DConditionModel

        >>> # load model
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4")
        >>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision
        >>> params = model.to_bf16(params)
        >>> # If you don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4")
        >>> flat_params = traverse_util.flatten_dict(params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> params = model.to_bf16(params, mask)
        ```"""
        return self._cast_floating_to(params, jnp.bfloat16, mask)

    def to_fp32(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `params` to `jax.numpy.float32`. This method can be used to explicitly convert the
        model parameters to fp32 precision. This returns a new `params` tree and does not cast the `params` in place.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip

        Examples:

        ```python
        >>> from diffusers import FlaxUNet2DConditionModel

        >>> # Download model and configuration from huggingface.co
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4")
        >>> # By default, the model params will be in fp32, to illustrate the use of this method,
        >>> # we'll first cast to fp16 and back to fp32
        >>> params = model.to_f16(params)
        >>> # now cast back to fp32
        >>> params = model.to_fp32(params)
        ```"""
        return self._cast_floating_to(params, jnp.float32, mask)

    def to_fp16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `params` to `jax.numpy.float16`. This returns a new `params` tree and does not cast the
        `params` in place.

        This method can be used on GPU to explicitly convert the model parameters to float16 precision to do full
        half-precision training or to save weights in float16 for inference in order to save memory and improve speed.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip

        Examples:

        ```python
        >>> from diffusers import FlaxUNet2DConditionModel

        >>> # load model
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4")
        >>> # By default, the model params will be in fp32, to cast these to float16
        >>> params = model.to_fp16(params)
        >>> # If you want don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4")
        >>> flat_params = traverse_util.flatten_dict(params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> params = model.to_fp16(params, mask)
        ```"""
        return self._cast_floating_to(params, jnp.float16, mask)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        dtype: jnp.dtype = jnp.float32,
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained flax model from a pre-trained model configuration.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids are namespaced under a user or organization name, like
                      `CompVis/stable-diffusion-v1-4`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_pretrained`],
                      e.g., `./my_model_directory/`.
            dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
                The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
                `jax.numpy.bfloat16` (on TPUs).

                This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
                specified all the computation will be performed with the given `dtype`.

                **Note that this only specifies the dtype of the computation and does not influence the dtype of model
                parameters.**

                If you wish to change the dtype of the model parameters, see [`~ModelMixin.to_fp16`] and
                [`~ModelMixin.to_bf16`].
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~ConfigMixin.from_config`]). Each key of `kwargs` that corresponds to
                      a configuration attribute will be used to override said attribute with the supplied `kwargs`
                      value. Remaining keys that do not correspond to any configuration attribute will be passed to the
                      underlying model's `__init__` function.

        Examples:

        ```python
        >>> from diffusers import FlaxUNet2DConditionModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("./test/saved_model/")
        ```"""
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        subfolder = kwargs.pop("subfolder", None)

        user_agent = {"file_type": "model", "framework": "flax", "from_auto_class": from_auto_class}

        # Load config if we don't provide a configuration
        config_path = config if config is not None else pretrained_model_name_or_path
        model, model_kwargs = cls.from_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            # model args
            dtype=dtype,
            **kwargs,
        )

        # Load model
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                # Load from a Flax checkpoint
                model_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
            # At this stage we don't have a weight file so we will raise an error.
            elif os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME):
                raise EnvironmentError(
                    f"Error no file named {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} "
                    "but there is a file for PyTorch weights."
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
                    f"{pretrained_model_name_or_path}."
                )
        else:
            try:
                model_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=FLAX_WEIGHTS_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {FLAX_WEIGHTS_NAME}."
                )
            except HTTPError as err:
                raise EnvironmentError(
                    f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                    f"{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}.\nCheckout your"
                    " internet connection or see how to run the library in offline mode at"
                    " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                )

        try:
            with open(model_file, "rb") as state_f:
                state = from_bytes(cls, state_f.read())
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            try:
                with open(model_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please"
                            " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                            " folder you cloned."
                        )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise EnvironmentError(f"Unable to convert {model_file} to Flax deserializable object. ")
        # make sure all arrays are stored as jnp.ndarray
        # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
        # https://github.com/google/flax/issues/1261
        state = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), state)

        # flatten dicts
        state = flatten_dict(state)

        # dictionary of key: dtypes for the model params
        param_dtypes = jax.tree_map(lambda x: x.dtype, state)
        # extract keys of parameters not in jnp.float32
        fp16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.float16]
        bf16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.bfloat16]

        # raise a warning if any of the parameters are not in jnp.float32
        if len(fp16_params) > 0:
            logger.warning(
                f"Some of the weights of {model.__class__.__name__} were initialized in float16 precision from "
                f"the model checkpoint at {pretrained_model_name_or_path}:\n{fp16_params}\n"
                "You should probably UPCAST the model weights to float32 if this was not intended. "
                "See [`~ModelMixin.to_fp32`] for further information on how to do this."
            )

        if len(bf16_params) > 0:
            logger.warning(
                f"Some of the weights of {model.__class__.__name__} were initialized in bfloat16 precision from "
                f"the model checkpoint at {pretrained_model_name_or_path}:\n{bf16_params}\n"
                "You should probably UPCAST the model weights to float32 if this was not intended. "
                "See [`~ModelMixin.to_fp32`] for further information on how to do this."
            )

        return model, unflatten_dict(state)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        params: Union[Dict, FrozenDict],
        is_main_process: bool = True,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~FlaxModelMixin.from_pretrained`]` class method

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # save model
        output_model_file = os.path.join(save_directory, FLAX_WEIGHTS_NAME)
        with open(output_model_file, "wb") as f:
            model_bytes = to_bytes(params)
            f.write(model_bytes)

        logger.info(f"Model weights saved in {output_model_file}")