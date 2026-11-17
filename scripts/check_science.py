"""Compare shared JavaScript numerical & deep-learning routines with Python / PyTorch references."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).absolute().parents[1]
rng = np.random.default_rng(402)
cases = [rng.normal(size=(37, 5)), rng.normal(size=(8, 19)), np.ones((4, 3))]
x = [1, 2, 4, 4, 9, 12]
y = [2, -1, 3, 3, 6, 10]
z = [-2.5, -0.7, 0.0, 0.45, 1.8, 3.1]
va = rng.normal(size=11).tolist()
vb = rng.normal(size=11).tolist()
Phi = rng.normal(size=(28, 4))
yt = rng.normal(size=28)

script = """const fs=require('fs'),vm=require('vm');
const sandbox={window:{},document:{},console,setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
const M=sandbox.window.DL4HL, d=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify({
  fit:M.linfit(d.x,d.y),
  rank:M.rank(d.x),
  rho:M.spearman(d.x,d.y),
  pca:d.cases.map(x=>M.pca(x,2)),
  relu:d.z.map(M.relu),
  gelu:d.z.map(M.gelu),
  softmax:M.softmax(d.z, 0.7),
  cosine:M.cosine(d.va, d.vb),
  ridge:M.lstsq(d.Phi, d.yt, 1.5)
}));"""

payload = dict(
    x=x, y=y, z=z, va=va, vb=vb,
    Phi=Phi.tolist(), yt=yt.tolist(),
    cases=[a.tolist() for a in cases],
)
raw = subprocess.check_output(["node", "-e", script, str(ROOT / "dl4hl/assets/core.js")], input=json.dumps(payload).encode())
actual = json.loads(raw)

fit = LinearRegression().fit(np.array(x)[:, None], y)
np.testing.assert_allclose([actual["fit"]["m"], actual["fit"]["b"]], [fit.coef_[0], fit.intercept_])
np.testing.assert_allclose(actual["rank"], rankdata(x))
np.testing.assert_allclose(actual["rho"], spearmanr(x, y).statistic)

zt = torch.tensor(z, dtype=torch.float64)
np.testing.assert_allclose(actual["relu"], F.relu(zt).numpy(), atol=1e-12)
np.testing.assert_allclose(actual["gelu"], F.gelu(zt, approximate="tanh").numpy(), atol=1e-6)
np.testing.assert_allclose(actual["softmax"], F.softmax(zt / 0.7, dim=0).numpy(), atol=1e-10)
cos_ref = float(F.cosine_similarity(torch.tensor(va)[None], torch.tensor(vb)[None]).item())
np.testing.assert_allclose(actual["cosine"], cos_ref, atol=1e-6)
ridge_ref = Ridge(alpha=1.5, fit_intercept=False).fit(Phi, yt).coef_
np.testing.assert_allclose(actual["ridge"], ridge_ref, atol=1e-8)

for X, result in zip(cases, actual["pca"]):
    if np.ptp(X) == 0:
        np.testing.assert_allclose(result["values"], 0, atol=1e-12)
        continue
    reference = PCA(2).fit(X)
    np.testing.assert_allclose(result["values"][:2], reference.explained_variance_, atol=1e-8)
    A = np.array(result["components"])
    B = reference.components_
    np.testing.assert_allclose(A.T @ A, B.T @ B, atol=1e-8)

# Ensure rendering variants does not change computed artists or image pixels.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT))
from dl4hl.plotting import set_style
from dl4hl.theming import THEME, figure_html

set_style()
fig, ax = plt.subplots()
line, = ax.plot(x, y)
im = ax.imshow(np.arange(4).reshape(2, 2), cmap="viridis")
color = line.get_color()
pixels = im.get_array().copy()
html = figure_html(fig)
assert html.count("<img ") == 3 and line.get_color() == color
np.testing.assert_array_equal(im.get_array(), pixels)
for mode in ("light", "dark"):
    assert len(set(THEME[mode][k] for k in THEME["categorical"])) == 12

print("PASS: OLS, Ridge, tied ranks, Spearman, ReLU, GELU, temperature Softmax, cosine similarity, PCA, plot preservation, and 12 distinct category colors")
