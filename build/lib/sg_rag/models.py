"""Local language-model and embedding adapters used by SG-RAG."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

class LocalCausalLLM:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        max_input_tokens: int = 7900,
        dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.dtype = dtype
        self.device_map = device_map
        self._tokenizer = None
        self._model = None

    def _torch_dtype(self):
        if self.dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]

    def load(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.truncation_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_dtype(),
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        self._model.generation_config.do_sample = False
        self._model.generation_config.temperature = None
        self._model.generation_config.top_p = None

    @property
    def tokenizer(self):
        self.load()
        return self._tokenizer

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = ""
            for message in messages:
                text += f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n"
            text += "<assistant>\n"

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        eos_token_ids = [self._tokenizer.eos_token_id]
        eot_token_id = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_token_id, int) and eot_token_id >= 0 and eot_token_id not in eos_token_ids:
            eos_token_ids.append(eot_token_id)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens or self.max_new_tokens),
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


class LocalEmbeddingModel:
    def __init__(
        self,
        model_path: str,
        batch_size: int = 8,
        max_length: int = 8192,
        dtype: str = "auto",
        backend: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = dtype
        self.backend = backend
        self.backend_used = "unloaded"
        self._tokenizer = None
        self._model = None

    def _torch_dtype(self):
        if self.dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]

    def load(self) -> None:
        if self._model is not None:
            return
        if self.backend in {"auto", "flag"}:
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore
                import transformers as _transformers

                use_fp16 = torch.cuda.is_available() and self.dtype != "float32"
                original_from_pretrained = _transformers.AutoModel.from_pretrained

                def _from_pretrained_compat(*args, **kwargs):
                    if "dtype" in kwargs and "torch_dtype" not in kwargs:
                        kwargs["torch_dtype"] = kwargs.pop("dtype")
                    else:
                        kwargs.pop("dtype", None)
                    return original_from_pretrained(*args, **kwargs)

                _transformers.AutoModel.from_pretrained = _from_pretrained_compat
                try:
                    self._model = BGEM3FlagModel(self.model_path, use_fp16=use_fp16)
                finally:
                    _transformers.AutoModel.from_pretrained = original_from_pretrained
                self.backend_used = "FlagEmbedding.BGEM3FlagModel"
                return
            except Exception as exc:
                if self.backend == "flag":
                    raise RuntimeError(
                        "FlagEmbedding backend was requested but could not be loaded. "
                        "Install it with `pip install FlagEmbedding` or use --embedding-backend hf."
                    ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_dtype(),
            trust_remote_code=True,
        ).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self.backend_used = "transformers_cls_pooling"

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        self.load()
        assert self._model is not None
        if self.backend_used == "FlagEmbedding.BGEM3FlagModel":
            if not texts:
                return np.empty((0, 0), dtype=np.float32)
            outputs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense = outputs["dense_vecs"] if isinstance(outputs, dict) else outputs
            vectors = np.asarray(dense, dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / np.clip(norms, 1e-12, None)

        assert self._tokenizer is not None
        device = self._model.device
        vectors = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="embedding", leave=False)
        for start in iterator:
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = self._model(**encoded)
                pooled = outputs.last_hidden_state[:, 0]
                pooled = F.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().float().cpu().numpy())
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(vectors, axis=0)

