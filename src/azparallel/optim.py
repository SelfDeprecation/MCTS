"""AdamW, ASAM over AdamW, and hybrid Muon+AdamW; clipping in every pass."""

import torch


class OptimizerBundle:
    def __init__(
        self,
        model,
        name="adamw",
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        rho=0.1,
        eta=0.01,
        muon_lr=0.02,
    ):
        if name not in ("adamw", "asam", "muon"):
            raise ValueError(f"Unknown optimizer {name}")
        if lr <= 0 or grad_clip <= 0 or rho <= 0 or eta < 0 or muon_lr <= 0:
            raise ValueError("Invalid optimizer parameters")
        self.model, self.name = model, name
        self.grad_clip, self.rho, self.eta = grad_clip, rho, eta
        self.parameters = list(model.parameters())
        self.optimizers = []
        if name == "muon":
            if not hasattr(torch.optim, "Muon"):
                raise RuntimeError("Muon requires PyTorch >= 2.9; select AdamW/ASAM or upgrade")
            # Muon is for hidden 2D matrices. Input/output layers and 1D biases/norms
            # use AdamW as required by the PyTorch Muon implementation.
            muon = [
                p for n, p in model.named_parameters() if n.startswith("trunk.") and p.ndim == 2
            ]
            ids = {id(p) for p in muon}
            other = [p for p in self.parameters if id(p) not in ids]
            if not muon:
                raise ValueError("Muon requires at least one residual block")
            self.optimizers.append(torch.optim.Muon(muon, lr=muon_lr, weight_decay=weight_decay))
            self.optimizers.append(torch.optim.AdamW(other, lr=lr, weight_decay=weight_decay))
        else:
            self.optimizers.append(
                torch.optim.AdamW(self.parameters, lr=lr, weight_decay=weight_decay)
            )

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def _backward(self, closure):
        self.zero_grad()
        loss, parts = closure()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.grad_clip, error_if_nonfinite=True
        )
        return {
            **parts,
            "objective": float(loss.detach()),
            "gradient_norm_before_clip": float(norm),
        }

    def step(self, closure) -> dict:
        stats = self._backward(closure)
        if self.name == "asam":
            # e = rho * T(w)^2 g / ||T(w)g||_2, T=|w|+eta for weights, 1 for biases.
            scales = {
                id(p): (p.detach().abs() + self.eta if "weight" in n else torch.ones_like(p))
                for n, p in self.model.named_parameters()
            }
            norm = torch.linalg.vector_norm(
                torch.stack(
                    [(scales[id(p)] * p.grad).norm() for p in self.parameters if p.grad is not None]
                )
            )
            originals = {}
            with torch.no_grad():
                for p in self.parameters:
                    if p.grad is not None:
                        originals[id(p)] = p.clone()
                        p.add_(self.rho * scales[id(p)].square() * p.grad / (norm + 1e-12))
            try:
                perturbed = self._backward(closure)
                stats["perturbed_objective"] = perturbed["objective"]
                stats["perturbed_gradient_norm_before_clip"] = perturbed[
                    "gradient_norm_before_clip"
                ]
            finally:
                with torch.no_grad():
                    for p in self.parameters:
                        if id(p) in originals:
                            p.copy_(originals[id(p)])
        for optimizer in self.optimizers:
            optimizer.step()
        return stats

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        for optimizer, state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(state)
