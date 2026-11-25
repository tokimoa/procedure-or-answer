"""MLX inference backend: sampled generation and mean-pooled token embeddings for retrieval."""

import numpy as np


def mx_seed(s: int):
    """Seed MLX's global PRNG so sampled decoding is reproducible per data seed."""
    import mlx.core as mx

    mx.random.seed(20260707 + 10007 * s)


class MLXBackend:
    def __init__(self, repo="mlx-community/Llama-3.2-3B-Instruct-4bit", max_tokens=110, temp=0.7):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.repo = repo
        self.max_tokens = max_tokens
        self.model, self.tok = load(repo)
        self._embt = self._find_embed_layer()
        try:
            from mlx_lm.sample_utils import make_sampler

            self._sampler = make_sampler(temp=temp)
        except Exception:
            self._sampler = None  # fall back to greedy decoding

    def _find_embed_layer(self):
        """Return the token-embedding module, whose location varies by architecture."""
        for path in (
            ("model", "embed_tokens"),
            ("embed_tokens",),
            ("model", "model", "embed_tokens"),
            ("language_model", "model", "embed_tokens"),
        ):
            obj, ok = self.model, True
            for p in path:
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    ok = False
                    break
            if ok and callable(obj):
                return obj
        # Otherwise scan the module tree for *.embed_tokens, then any embedding module.
        try:
            mods = list(self.model.named_modules())
            for name, mod in mods:
                if name.endswith("embed_tokens") and callable(mod):
                    return mod
            for name, mod in mods:
                if "embed" in name.lower() and callable(mod):
                    return mod
        except Exception:
            pass  # no usable module tree: report the failure below
        raise AttributeError(f"embed_tokens not found for {self.repo}")

    def _template(self, user: str):
        msgs = [{"role": "user", "content": user}]
        try:  # turn off reasoning output where the chat template supports the flag
            return self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tok.apply_chat_template(msgs, add_generation_prompt=True)

    def _gen(self, user: str) -> str:
        from mlx_lm import generate

        prompt = self._template(user)
        kw = {"max_tokens": self.max_tokens}
        if self._sampler is not None:
            kw["sampler"] = self._sampler
        return generate(self.model, self.tok, prompt=prompt, verbose=False, **kw)

    def gen(self, user: str) -> str:
        """Generate a reply to a single user message."""
        return self._gen(user)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled input token embeddings, one row per text."""
        out = []
        for t in texts:
            ids = self.mx.array(self.tok.encode(t if t.strip() else " "))
            v = self._embt(ids).mean(axis=0).astype(self.mx.float32)  # bf16 -> f32
            out.append(np.array(v, dtype=np.float32))
        return np.stack(out)
