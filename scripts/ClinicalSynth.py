from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

Array = np.ndarray

@dataclass
class Config:
    """
    Configuration for the clinical-style synthetic generator.

    H, W: image height/width (pixels)
    seed: RNG seed (reproducible)
    a0, a_*: disease (z) logistic model coefficients
    eta_fn/fp: label noise (false negative / false positive)
    size_*: lesion size model coefficients (mm)
    con_*: lesion contrast model coefficients (higher -> brighter)
    bg_base: background texture strength
    device_blur: blur steps per device (A/B/C) to simulate scanner differences
    """
    H: int = 128
    W: int = 128
    seed: int = 0
    a0: float = -2.2
    a_age: float = 0.02
    a_den: float = 0.50
    a_risk: float = 0.80
    eta_fn: float = 0.08
    eta_fp: float = 0.05
    size_base: float = 12.0
    size_age: float = 0.05
    size_den: float = 1.50
    con_base: float = 1.30
    con_den: float = 0.25
    bg_base: float = 0.10
    device_blur: Dict[str, int] = None  # set in __post_init__

    def __post_init__(self):
        if self.device_blur is None:
            self.device_blur = {"A": 0, "B": 1, "C": 2}


def sigmoid(x: Array) -> Array:
    """Numerically stable logistic function."""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class ClinicalSynth:
    """
    Synthetic multimodal generator with a simple clinical hidden rule.

    Hidden rule:
      1) z ~ Bernoulli( sigmoid(a0 + a_age*age + a_den*density + a_risk*risk) )
      2) If z=1 -> lesion params (size_mm, contrast); else near-zero lesion
      3) image = background(density, device) + z * lesion(size, contrast)
      4) radiomics = simple functions of (image, lesion) + small noise
      5) y (observed label) = noisy version of z (eta_fn/eta_fp)

    Public API:
      - sample_one()  -> (image, radiomics, y, z, meta, names)
      - sample_batch(n)
    """

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

    # 1) Latents -----------------------------------------------------
    def _sample_patient(self) -> Tuple[int, int, int, str]:
        """Sample (age, density[0..3], risk[0/1], device['A'|'B'|'C'])."""
        age = int(self.rng.randint(40, 86))
        density = int(self.rng.choice([0, 1, 2, 3], p=[0.15, 0.35, 0.35, 0.15]))
        risk = int(self.rng.binomial(1, 0.20))
        device = str(self.rng.choice(["A", "B", "C"], p=[0.50, 0.35, 0.15]))
        return age, density, risk, device

    def _sample_z(self, age: int, den: int, risk: int) -> int:
        """Sample true disease state z in {0,1} from a logistic model."""
        logit = (self.cfg.a0 +
                 self.cfg.a_age * age +
                 self.cfg.a_den * den +
                 self.cfg.a_risk * risk)
        return int(self.rng.rand() < sigmoid(logit))

    def _lesion_params(self, z: int, age: int, den: int) -> Dict[str, float]:
        """Return lesion parameters (mm, contrast). Zeroed if z=0."""
        if z == 0:
            return {"size_mm": 0.0, "contrast": 0.0}
        size = (self.cfg.size_base
                + self.cfg.size_age * (age - 55)
                + self.cfg.size_den * den
                + self.rng.normal(0.0, 2.0))
        size = float(max(0.0, size))
        contrast = (self.cfg.con_base
                    - self.cfg.con_den * den
                    + self.rng.normal(0.0, 0.10))
        contrast = float(max(0.0, contrast))
        return {"size_mm": size, "contrast": contrast}

    # ---------- 2) Image -------------------------------------------------------
    def _background(self, den: int, device: str) -> Array:
        """Create simple breast-like texture. Density ↑ => stronger texture; device => blur."""
        H, W = self.cfg.H, self.cfg.W
        tex = np.zeros((H, W), np.float32)
        # multi-scale noise (fast 1/f-ish)
        for scale, weight in [(8, 1.0), (16, 0.6), (32, 0.3)]:
            h, w = max(1, H // scale), max(1, W // scale)
            n = self.rng.normal(0, 1, (h, w)).astype(np.float32)
            n = np.repeat(np.repeat(n, scale, axis=0), scale, axis=1)[:H, :W]
            tex += weight * n
        tex /= (tex.std() + 1e-6)

        den_scale = 1.0 + 0.25 * den
        img = 0.5 + self.cfg.bg_base * den_scale * tex

        # simple blur: average of neighbors, repeated device_blur[device] times
        k = int(self.cfg.device_blur[device])
        for _ in range(k):
            img = (img +
                   np.roll(img, 1, 0) + np.roll(img, -1, 0) +
                   np.roll(img, 1, 1) + np.roll(img, -1, 1)) / 5.0

        return np.clip(img, 0.0, 1.0)

    def _draw_lesion(self, img: Array, size_mm: float, contrast: float) -> Array:
        """Add a soft circular lesion. Return boolean mask of lesion area."""
        if size_mm <= 0.0:
            return np.zeros_like(img, dtype=bool)

        H, W = img.shape
        px_per_mm = 1 / 0.3  # ~0.3 mm per pixel
        radius_px = max(2, int(0.5 * size_mm * px_per_mm))
        cy = int(self.rng.randint(H // 4, 3 * H // 4))
        cx = int(self.rng.randint(W // 4, 3 * W // 4))

        yy, xx = np.ogrid[:H, :W]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        mask = dist2 <= radius_px ** 2

        # soft Gaussian-like bump
        sigma = max(1.0, 0.35 * radius_px)
        blob = np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)
        img += float(contrast) * blob

        return mask

    # ---------- 3) Radiomics ---------------------------------------------------
    def _radiomics(self, img: Array, mask: Array, den: int,
                   size_mm: float, contrast: float) -> Tuple[Array, List[str]]:
        """
        Minimal radiomics:
          - lesion_area_px      : area in pixels
          - mean_intensity      : global image mean
          - lesion_intensity    : mean inside lesion (0 if none)
          - edge_sharpness      : avg gradient magnitude at lesion boundary
          - parenchyma_density  : density index (0..3)
          - size_mm, contrast   : returned with small measurement noise
        """
        names = ["lesion_area_px", "mean_intensity", "lesion_intensity",
                 "edge_sharpness", "parenchyma_density", "size_mm", "contrast"]

        area = float(mask.sum())

        mean_int = float(img.mean())
        lesion_mean = float(img[mask].mean()) if area > 0 else 0.0

        # gradient magnitude
        gy, gx = np.gradient(img.astype(np.float32))
        grad = np.sqrt(gx ** 2 + gy ** 2)

        # boundary: 4-neighbor difference on mask
        if area > 0:
            m = mask.astype(np.uint8)
            edge = (np.abs(np.roll(m, 1, 0) - m) +
                    np.abs(np.roll(m, -1, 0) - m) +
                    np.abs(np.roll(m, 1, 1) - m) +
                    np.abs(np.roll(m, -1, 1) - m)) > 0
            edge_sharp = float(grad[edge].mean()) if edge.any() else 0.0
        else:
            edge_sharp = 0.0

        # add small measurement noise
        noise = self.rng.normal
        feats = np.array([
            area + noise(0, 3.0),
            mean_int + noise(0, 0.01),
            lesion_mean + noise(0, 0.02),
            edge_sharp + noise(0, 0.005),
            float(den) + noise(0, 0.1),
            float(size_mm) + noise(0, 0.5),
            float(contrast) + noise(0, 0.05),
        ], dtype=np.float32)

        return feats, names

    # ---------- 4) Observed label ---------------------------------------------
    def _clinical_label(self, z: int) -> int:
        """Noisy observed label y from true z using FN/FP rates."""
        if z == 1 and self.rng.rand() < self.cfg.eta_fn:
            return 0
        if z == 0 and self.rng.rand() < self.cfg.eta_fp:
            return 1
        return z

    # ---------- Public API -----------------------------------------------------
    def sample_one(self):
        """
        Returns:
          image      : (H, W) float32 in [0,1]
          radiomics  : (7,) float32 vector
          y          : observed clinical label (0/1)
          z          : true disease (0/1)
          meta       : dict with age, density, risk, device
          names      : list of radiomic feature names
        """
        age, den, risk, device = self._sample_patient()
        z = self._sample_z(age, den, risk)
        les = self._lesion_params(z, age, den)

        img = self._background(den, device)
        mask = self._draw_lesion(img, les["size_mm"], les["contrast"])

        # normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = img.astype(np.float32)

        radiomics, names = self._radiomics(img, mask, den, les["size_mm"], les["contrast"])
        y = self._clinical_label(z)
        meta = {"age": age, "density": den, "risk": risk, "device": device, "z": z}

        return img, radiomics, int(y), int(z), meta, names

    def sample_batch(self, n: int):
        """Vectorized wrapper over sample_one()."""
        imgs, tabs, ys, zs, metas = [], [], [], [], []
        names = None
        for _ in range(n):
            img, r, y, z, meta, names = self.sample_one()
            imgs.append(img); tabs.append(r); ys.append(y); zs.append(z); metas.append(meta)
        return (np.stack(imgs, axis=0),
                np.stack(tabs, axis=0),
                np.asarray(ys, dtype=np.int64),
                np.asarray(zs, dtype=np.int64),
                metas, names)
    
    
# ---------------- Example use ----------------
# gen = ClinicalSynth(Config(seed=42))
# imgs, tabs, y, z, meta, names = gen.sample_batch(8)
# print(imgs.shape)  # (8, 128, 128)
# print(tabs.shape)  # (8, 7)
# print(names)       # radiomic feature names
