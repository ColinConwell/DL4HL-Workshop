/* ══════════════════════════════════════════════════════════════════
   Figure 3 — What makes a deep network deep?
   Directly compares Part 1's 0-hidden-layer Linear Perceptron against
   1-hidden-layer and 2-hidden-layer MLPs, while letting learners toggle
   Nonlinear (ReLU / GELU) vs. Linear (No σ) activations.
   Key lesson:
   - Depth WITHOUT nonlinearity collapses algebraically to one flat line
     (W₃·W₂·W₁ = W_eff), no matter how many layers are stacked.
   - Nonlinearity WITHOUT depth places one round of flat creases across
     input space (enough for XOR, strained on multi-island parity).
   - Depth + Nonlinearity composes folds of folds, carving intricate
     multi-region geometry cleanly with compact layers.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('deep_mlp', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 1,
    title: 'What makes a deep network deep: stacking layers vs. adding bends',
    hint: 'Compare 0 Hidden (Part-1 Perceptron), 1 Hidden, and 2 Hidden layers &middot; switch Unit bend to Linear (no bend) to see stacked layers collapse into one straight line &middot; press play to watch training from scratch'
  });

  var seed = cfg.seed || 23;
  var depth = cfg.depth != null ? cfg.depth : 2;          // 0, 1, or 2 hidden layers
  var width = cfg.width || 4;                              // hidden units per layer (3..8)
  var actKind = cfg.activation || 'relu';                  // 'relu' | 'gelu' | 'linear'
  var pattern = cfg.pattern || 'bullseye';                 // 'separable' | 'xor' | 'bullseye'

  var X = [], y = [];
  var W = [], B = [];        // list of weight matrices and bias vectors per layer
  var mW = [], vW = [], mB = [], vB = [], adamT = 0;
  var stepCount = 0, accNow = 0, lossNow = 0;

  function act(z) {
    if (actKind === 'linear') return z;
    if (actKind === 'gelu') return M.gelu(z);
    return z > 0 ? z : 0.03 * z; // tiny leaky slope prevents dead hinges on small width
  }
  function dAct(z) {
    if (actKind === 'linear') return 1;
    if (actKind === 'gelu') {
      var e = 1e-4;
      return (M.gelu(z + e) - M.gelu(z - e)) / (2 * e);
    }
    return z > 0 ? 1 : 0.03;
  }

  function buildData() {
    var rng = M.rng(seed);
    X = []; y = [];
    if (pattern === 'separable') {
      for (var c = 0; c < 2; c++) {
        var cx = c ? 0.56 : -0.56, cy = c ? 0.44 : -0.44;
        for (var i = 0; i < 36; i++) {
          X.push([cx + rng.normal(0, 0.24), cy + rng.normal(0, 0.24)]);
          y.push(c);
        }
      }
    } else if (pattern === 'xor') {
      [[-0.62, -0.62, 0], [0.62, 0.62, 0], [-0.62, 0.62, 1], [0.62, -0.62, 1]].forEach(function (q) {
        for (var k = 0; k < 18; k++) {
          X.push([q[0] + rng.normal(0, 0.16), q[1] + rng.normal(0, 0.16)]);
          y.push(q[2]);
        }
      });
    } else {
      /* Nested bullseye: inner core (class 1) + middle moat (class 0) + outer ring (class 1).
         4 straight creases in 1 hidden layer can enclose one zone, whereas a 2nd hidden layer
         folds the radial cone a second time to carve both concentric boundaries cleanly! */
      for (var bi = 0; bi < 30; bi++) {
        var a0 = (2 * Math.PI * bi) / 30 + rng.normal(0, 0.05);
        var r0 = Math.sqrt(rng.next()) * 0.27;
        X.push([r0 * Math.cos(a0), r0 * Math.sin(a0)]); y.push(1);
      }
      for (var bm = 0; bm < 44; bm++) {
        var a1 = (2 * Math.PI * bm) / 44 + rng.normal(0, 0.03);
        var r1 = 0.56 + rng.normal(0, 0.045);
        X.push([r1 * Math.cos(a1), r1 * Math.sin(a1)]); y.push(0);
      }
      for (var bo = 0; bo < 46; bo++) {
        var a2 = (2 * Math.PI * bo) / 46 + rng.normal(0, 0.03);
        var r2 = 0.96 + rng.normal(0, 0.045);
        X.push([r2 * Math.cos(a2), r2 * Math.sin(a2)]); y.push(1);
      }
    }
    initNetwork();
  }

  function layerDims() {
    if (depth === 0) return [2, 1];
    if (depth === 1) return [2, width, 1];
    return [2, width, width, 1];
  }

  function paramCount() {
    var dims = layerDims(), total = 0;
    for (var l = 0; l < dims.length - 1; l++) total += dims[l] * dims[l + 1] + dims[l + 1];
    return total;
  }

  function initNetwork() {
    var dims = layerDims();
    var rng = M.rng(seed * 31 + depth * 13 + width * 7 + (pattern === 'bullseye' ? 5 : pattern === 'xor' ? 2 : 0));
    W = []; B = []; mW = []; vW = []; mB = []; vB = [];
    adamT = 0; stepCount = 0;

    for (var l = 0; l < dims.length - 1; l++) {
      var dIn = dims[l], dOut = dims[l + 1];
      var scale = Math.sqrt(2.0 / dIn) * 0.95;
      var wl = new Float64Array(dOut * dIn);
      var bl = new Float64Array(dOut);
      if (l === 0 && dOut > 1) {
        /* Spread initial Layer-1 orientations symmetrically around the origin */
        for (var r = 0; r < dOut; r++) {
          var ang = (2 * Math.PI * r) / dOut + 0.28;
          wl[r * 2] = 1.7 * Math.cos(ang);
          wl[r * 2 + 1] = 1.7 * Math.sin(ang);
          bl[r] = rng.uniform(-0.32, 0.32);
        }
      } else {
        for (var k = 0; k < wl.length; k++) wl[k] = rng.normal(0, scale);
        for (var r2 = 0; r2 < dOut; r2++) bl[r2] = rng.normal(0, 0.15);
      }
      W.push(wl); B.push(bl);
      mW.push(new Float64Array(wl.length)); vW.push(new Float64Array(wl.length));
      mB.push(new Float64Array(bl.length)); vB.push(new Float64Array(bl.length));
    }
    evaluate();
  }

  function forwardFull(pt) {
    var dims = layerDims(), L = W.length;
    var acts = [pt], zs = [];
    var cur = pt;
    for (var l = 0; l < L; l++) {
      var dIn = dims[l], dOut = dims[l + 1], wl = W[l], bl = B[l];
      var z = new Float64Array(dOut), nxt = new Float64Array(dOut);
      var isLast = l === L - 1;
      for (var r = 0; r < dOut; r++) {
        var s = bl[r], off = r * dIn;
        for (var c = 0; c < dIn; c++) s += wl[off + c] * cur[c];
        z[r] = s;
        nxt[r] = isLast ? M.sigmoid(s) : act(s);
      }
      zs.push(z); acts.push(nxt); cur = nxt;
    }
    return { acts: acts, zs: zs, prob: cur[0], logit: zs[L - 1][0] };
  }

  function predictProb(x0, x1) {
    var dims = layerDims(), L = W.length;
    var cur0 = x0, cur1 = x1;
    if (L === 1) return M.sigmoid(W[0][0] * cur0 + W[0][1] * cur1 + B[0][0]);
    var cur = [cur0, cur1];
    for (var l = 0; l < L; l++) {
      var dIn = dims[l], dOut = dims[l + 1], wl = W[l], bl = B[l], isLast = l === L - 1;
      var nxt = new Array(dOut);
      for (var r = 0; r < dOut; r++) {
        var s = bl[r], off = r * dIn;
        for (var c = 0; c < dIn; c++) s += wl[off + c] * cur[c];
        nxt[r] = isLast ? M.sigmoid(s) : act(s);
      }
      cur = nxt;
    }
    return cur[0];
  }

  /* Exact collapsed linear vector [w1, w2, b] when actKind === 'linear' */
  function effectiveLinear() {
    var dims = layerDims(), L = W.length;
    var v0 = forwardFull([0, 0]).logit;
    var vx = forwardFull([1, 0]).logit - v0;
    var vy = forwardFull([0, 1]).logit - v0;
    return [vx, vy, v0];
  }

  function trainSteps(iters) {
    var dims = layerDims(), L = W.length, n = X.length;
    var lr = depth === 0 ? 0.08 : (actKind === 'linear' ? 0.03 : 0.038);
    var b1 = 0.9, b2 = 0.999, eps = 1e-8;

    for (var it = 0; it < iters; it++) {
      var gW = W.map(function (w) { return new Float64Array(w.length); });
      var gB = B.map(function (b) { return new Float64Array(b.length); });

      for (var i = 0; i < n; i++) {
        var fw = forwardFull(X[i]);
        var dOutVec = [fw.prob - y[i]]; // d(BCE)/d(logit)

        for (var l = L - 1; l >= 0; l--) {
          var dIn = dims[l], dOut = dims[l + 1], wl = W[l], gw = gW[l], gb = gB[l];
          var prevA = fw.acts[l];
          var dPrev = l > 0 ? new Float64Array(dIn) : null;

          for (var r = 0; r < dOut; r++) {
            var dz = dOutVec[r] / n;
            gb[r] += dz;
            var off = r * dIn;
            for (var c = 0; c < dIn; c++) {
              gw[off + c] += dz * prevA[c];
              if (dPrev) dPrev[c] += dOutVec[r] * wl[off + c];
            }
          }
          if (dPrev) {
            var zPrev = fw.zs[l - 1];
            for (var c2 = 0; c2 < dIn; c2++) dPrev[c2] *= dAct(zPrev[c2]);
            dOutVec = dPrev;
          }
        }
      }

      adamT++;
      var c1 = 1 - Math.pow(b1, adamT), c2 = 1 - Math.pow(b2, adamT);
      for (var l2 = 0; l2 < L; l2++) {
        for (var k = 0; k < W[l2].length; k++) {
          var g = gW[l2][k] + 3e-4 * W[l2][k];
          mW[l2][k] = b1 * mW[l2][k] + (1 - b1) * g;
          vW[l2][k] = b2 * vW[l2][k] + (1 - b2) * g * g;
          W[l2][k] -= lr * (mW[l2][k] / c1) / (Math.sqrt(vW[l2][k] / c2) + eps);
        }
        for (var k2 = 0; k2 < B[l2].length; k2++) {
          var gbVal = gB[l2][k2];
          mB[l2][k2] = b1 * mB[l2][k2] + (1 - b1) * gbVal;
          vB[l2][k2] = b2 * vB[l2][k2] + (1 - b2) * gbVal * gbVal;
          B[l2][k2] -= lr * (mB[l2][k2] / c1) / (Math.sqrt(vB[l2][k2] / c2) + eps);
        }
      }
      stepCount++;
    }
    evaluate();
  }

  function evaluate() {
    var ok = 0, loss = 0, n = X.length;
    for (var i = 0; i < n; i++) {
      var p = M.clamp(predictProb(X[i][0], X[i][1]), 1e-6, 1 - 1e-6);
      if ((p >= 0.5 ? 1 : 0) === y[i]) ok++;
      loss -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
    }
    accNow = ok / (n || 1);
    lossNow = loss / (n || 1);
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.04);
  M.mini(split.a, 'network structure: stacked layers & unit bends');
  M.mini(split.b, 'decision boundary in input space');
  var hf = function (w) { return M.clamp(w * 0.82, 240, 330); };
  var netStage = M.stage(split.a, { height: hf, ref: 410, label: 'Neural network diagram comparing 0, 1, and 2 hidden layers with bent or straight-line units' });
  var planeStage = M.stage(split.b, { height: function () { return netStage.h || 275; }, ref: 410, label: '2D input plane showing learned decision regions and crease lines' });

  M.legend(root, [
    { color: C.blue, label: 'category A (0)' },
    { color: C.orange, label: 'category B (1)' },
    { color: C.green, label: 'positive weight', line: true },
    { color: C.red, label: 'negative weight', line: true },
    { color: C.ink, label: '50% decision boundary', line: true }
  ]);

  function fitInstant() {
    initNetwork();
    trainSteps(260);
  }

  var row1 = M.row(root);
  var playBtn = M.button(row1, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Animate training from random initialization' });
  M.button(row1, 'step', function () { pause(); if (stepCount >= 260) initNetwork(); trainSteps(10); drawAll(); }, { aria: 'Run 10 update steps' });
  var depthT = M.toggle(row1, {
    label: 'Depth', value: String(depth),
    options: [
      ['0', '0 Hidden (Perceptron)'],
      ['1', '1 Hidden Layer'],
      ['2', '2 Hidden Layers']
    ],
    onChange: function (v) { pause(); depth = +v; widthS.input.disabled = (depth === 0); widthS.set(width, true); fitInstant(); drawAll(); }
  });

  var row2 = M.row(root);
  var actT = M.toggle(row2, {
    label: 'Unit bend', value: actKind,
    options: [
      ['relu', 'ReLU (hinge)'],
      ['gelu', 'GELU (smooth)'],
      ['linear', 'Linear (no bend)']
    ],
    onChange: function (v) { pause(); actKind = v; fitInstant(); drawAll(); }
  });
  var patT = M.toggle(row2, {
    label: 'Pattern', value: pattern,
    options: [
      ['separable', 'Separable'],
      ['xor', 'XOR'],
      ['bullseye', 'Nested Rings']
    ],
    onChange: function (v) { pause(); pattern = v; buildData(); trainSteps(260); drawAll(); }
  });

  var row3 = M.row(root);
  var widthS = M.slider(row3, {
    label: 'Width / layer', min: 3, max: 8, step: 1, value: width, accent: 'blue',
    fmt: function (v) { return depth === 0 ? 'n/a' : (Math.round(v) + ' units'); },
    onInput: function (v) { pause(); width = Math.round(v); if (depth > 0) { fitInstant(); drawAll(); } }
  });
  widthS.input.disabled = (depth === 0);
  M.spacer(row3);
  M.resetButton(row3, function () {
    pause();
    depth = cfg.depth != null ? cfg.depth : 2;
    width = cfg.width || 4;
    actKind = cfg.activation || 'relu';
    pattern = cfg.pattern || 'bullseye';
    depthT.set(String(depth), true);
    actT.set(actKind, true);
    patT.set(pattern, true);
    widthS.input.disabled = (depth === 0);
    widthS.set(width, true);
    buildData(); trainSteps(260); drawAll();
  });
  var read = M.readout(root);

  var loop = M.loop(function () {
    trainSteps(5);
    drawAll();
    if (stepCount >= 260 || (accNow >= 0.995 && lossNow < 0.05)) { pause(); return false; }
    return true;
  });
  function play() {
    initNetwork();
    loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on');
  }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  /* ─── Draw Left: Network Architecture Diagram ─── */
  function drawNodeGlyph(ctx, x, y, r, kind, s) {
    ctx.save();
    ctx.strokeStyle = kind === 'linear' ? C.red : C.orange;
    ctx.lineWidth = 1.6 * s.sc;
    ctx.beginPath();
    var g = r * 0.52;
    if (kind === 'linear') {
      ctx.moveTo(x - g, y + g * 0.75);
      ctx.lineTo(x + g, y - g * 0.75);
    } else if (kind === 'gelu') {
      for (var k = 0; k <= 12; k++) {
        var u = -2.2 + (k / 12) * 4.4;
        var px = x + (u / 2.2) * g;
        var py = y + g * 0.55 - (M.gelu(u) / 2.2) * g * 1.15;
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
    } else {
      ctx.moveTo(x - g, y + g * 0.45);
      ctx.lineTo(x - g * 0.1, y + g * 0.45);
      ctx.lineTo(x + g * 0.85, y - g * 0.75);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawNet() {
    var ctx = netStage.ctx, s = netStage;
    if (!s.w) return;
    M.clear(ctx, s);
    var dims = layerDims(), nCols = dims.length;
    var padX = 32 * s.sc, topY = 38 * s.sc, botY = s.h - 32 * s.sc;
    var colX = [];
    for (var c = 0; c < nCols; c++) {
      colX.push(padX + (c / (nCols - 1)) * (s.w - 2 * padX));
    }

    var coords = [];
    for (var l = 0; l < nCols; l++) {
      var cnt = dims[l], arr = [];
      for (var j = 0; j < cnt; j++) {
        var frac = cnt === 1 ? 0.5 : (j + 0.5) / cnt;
        if (l === 0) frac = j === 0 ? 0.30 : 0.70;
        arr.push({ x: colX[l], y: topY + frac * (botY - topY) });
      }
      coords.push(arr);
    }

    /* Draw weighted edges */
    for (var el = 0; el < W.length; el++) {
      var wl = W[el], dIn = dims[el], dOut = dims[el + 1];
      var maxAbs = 0.6;
      for (var m = 0; m < wl.length; m++) if (Math.abs(wl[m]) > maxAbs) maxAbs = Math.abs(wl[m]);
      for (var r = 0; r < dOut; r++) {
        for (var ci = 0; ci < dIn; ci++) {
          var wVal = wl[r * dIn + ci];
          var mag = M.clamp(Math.abs(wVal) / maxAbs, 0.06, 1);
          ctx.strokeStyle = M.rgba(wVal >= 0 ? C.green : C.red, 0.18 + 0.66 * mag);
          ctx.lineWidth = (0.7 + 3.4 * mag) * s.sc;
          ctx.beginPath();
          ctx.moveTo(coords[el][ci].x, coords[el][ci].y);
          ctx.lineTo(coords[el + 1][r].x, coords[el + 1][r].y);
          ctx.stroke();
        }
      }
    }

    /* Column labels & nodes */
    var colTitles = depth === 0
      ? ['Inputs', 'Perceptron']
      : depth === 1
        ? ['Inputs', 'Layer 1', 'Output']
        : ['Inputs', 'Layer 1', 'Layer 2', 'Output'];

    for (var cl = 0; cl < nCols; cl++) {
      ctx.fillStyle = C.muted;
      ctx.font = M.font(s, 8.2, 600);
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(colTitles[cl].toUpperCase(), colX[cl], topY - 12 * s.sc);

      var rad = (cl === 0 || cl === nCols - 1) ? 13.5 * s.sc : M.clamp((botY - topY) / (dims[cl] * 2.7), 8.5 * s.sc, 13 * s.sc);
      for (var ni = 0; ni < coords[cl].length; ni++) {
        var nd = coords[cl][ni];
        var isHid = cl > 0 && cl < nCols - 1;
        var borderC = isHid ? (actKind === 'linear' ? C.red : C.ink) : C.ink;
        M.dot(ctx, nd.x, nd.y, rad, C.surface, borderC, 1.45 * s.sc);
        if (cl === 0) {
          ctx.fillStyle = C.ink; ctx.font = M.font(s, 10, 600, 'sans');
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(ni === 0 ? 'x1' : 'x2', nd.x, nd.y + 0.5);
        } else if (cl === nCols - 1) {
          ctx.fillStyle = C.ink; ctx.font = M.font(s, 9.5, 600, 'sans');
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText('out', nd.x, nd.y + 0.5);
        } else {
          drawNodeGlyph(ctx, nd.x, nd.y, rad, actKind, s);
        }
      }
    }

    /* Bottom banner clarifying collapse vs composition */
    if (depth > 0 && actKind === 'linear') {
      M.pill(ctx, s, 'No bends: stacked linear layers collapse into 1 straight line!', s.w * 0.5, s.h - 12 * s.sc, {
        align: 'center', color: C.red, size: 8.2
      });
    } else if (depth === 0) {
      M.pill(ctx, s, 'Part-1 Perceptron: 3 weights \u2192 1 straight line', s.w * 0.5, s.h - 12 * s.sc, {
        align: 'center', color: C.muted, size: 8.3
      });
    } else if (depth === 1) {
      M.pill(ctx, s, '1 Hidden Layer: ' + width + ' flat creases combined once', s.w * 0.5, s.h - 12 * s.sc, {
        align: 'center', color: C.blue, size: 8.3
      });
    } else {
      M.pill(ctx, s, '2 Hidden Layers: Layer 2 folds Layer 1\u2019s already-folded space', s.w * 0.5, s.h - 12 * s.sc, {
        align: 'center', color: C.green, size: 8.2
      });
    }
  }

  /* ─── Draw Right: Input Space Decision Regions & Creases ─── */
  var vP = M.view(planeStage, { x: [-1.25, 1.25], y: [-1.25, 1.25], equal: true, pad: { l: 32, r: 12, t: 12, b: 28 } });

  function drawPlane() {
    var ctx = planeStage.ctx, s = planeStage;
    if (!s.w) return;
    vP.setDomain([-1.25, 1.25], [-1.25, 1.25]);
    M.clear(ctx, s);

    var G = 66;
    var gridP = new Float64Array((G + 1) * (G + 1));
    for (var gj = 0; gj <= G; gj++) {
      var gy = vP.y1 - (gj / G) * (vP.y1 - vP.y0);
      for (var gi = 0; gi <= G; gi++) {
        var gx = vP.x0 + (gi / G) * (vP.x1 - vP.x0);
        gridP[gj * (G + 1) + gi] = predictProb(gx, gy);
      }
    }

    M.field(ctx, vP, G, G, function (_x, _y, i, j) {
      var p = gridP[j * (G + 1) + i], col = M.diverge(p);
      return [col[0], col[1], col[2], 145];
    });
    M.axes(ctx, vP, s, { xlabel: 'input 1', ylabel: 'input 2', nx: 3, ny: 3, grid: false });

    ctx.save(); vP.clip(ctx);

    /* Layer-1 straight creases (when depth >= 1 and nonlinear) */
    if (depth >= 1 && actKind !== 'linear') {
      ctx.setLineDash([4 * s.sc, 3 * s.sc]);
      ctx.strokeStyle = M.rgba(C.muted, 0.45);
      ctx.lineWidth = 1.05 * s.sc;
      for (var hIdx = 0; hIdx < width; hIdx++) {
        var wx = W[0][hIdx * 2], wy = W[0][hIdx * 2 + 1], b0 = B[0][hIdx];
        ctx.beginPath();
        if (Math.abs(wy) > Math.abs(wx)) {
          ctx.moveTo(vP.px(vP.x0), vP.py(-(b0 + wx * vP.x0) / wy));
          ctx.lineTo(vP.px(vP.x1), vP.py(-(b0 + wx * vP.x1) / wy));
        } else if (Math.abs(wx) > 1e-6) {
          ctx.moveTo(vP.px(-(b0 + wy * vP.y0) / wx), vP.py(vP.y0));
          ctx.lineTo(vP.px(-(b0 + wy * vP.y1) / wx), vP.py(vP.y1));
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    /* Exact p = 0.5 contour via marching-squares segments (or exact line in linear mode) */
    if (depth === 0 || actKind === 'linear') {
      var eff = effectiveLinear();
      if (Math.abs(eff[0]) + Math.abs(eff[1]) > 1e-5) {
        ctx.strokeStyle = (depth > 0 && actKind === 'linear') ? C.red : C.ink;
        ctx.lineWidth = 2.1 * s.sc;
        ctx.beginPath();
        if (Math.abs(eff[1]) > Math.abs(eff[0])) {
          ctx.moveTo(vP.px(vP.x0), vP.py(-(eff[2] + eff[0] * vP.x0) / eff[1]));
          ctx.lineTo(vP.px(vP.x1), vP.py(-(eff[2] + eff[0] * vP.x1) / eff[1]));
        } else {
          ctx.moveTo(vP.px(-(eff[2] + eff[1] * vP.y0) / eff[0]), vP.py(vP.y0));
          ctx.lineTo(vP.px(-(eff[2] + eff[1] * vP.y1) / eff[0]), vP.py(vP.y1));
        }
        ctx.stroke();
      }
    } else {
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 1.9 * s.sc;
      ctx.beginPath();
      for (var cj = 0; cj < G; cj++) {
        var yTop = vP.y1 - (cj / G) * (vP.y1 - vP.y0);
        var yBot = vP.y1 - ((cj + 1) / G) * (vP.y1 - vP.y0);
        for (var ci2 = 0; ci2 < G; ci2++) {
          var xLeft = vP.x0 + (ci2 / G) * (vP.x1 - vP.x0);
          var xRight = vP.x0 + ((ci2 + 1) / G) * (vP.x1 - vP.x0);
          var v00 = gridP[cj * (G + 1) + ci2] - 0.5;
          var v10 = gridP[cj * (G + 1) + ci2 + 1] - 0.5;
          var v01 = gridP[(cj + 1) * (G + 1) + ci2] - 0.5;
          var v11 = gridP[(cj + 1) * (G + 1) + ci2 + 1] - 0.5;
          var pts = [];
          if (v00 * v10 < 0) pts.push([M.lerp(xLeft, xRight, v00 / (v00 - v10)), yTop]);
          if (v10 * v11 < 0) pts.push([xRight, M.lerp(yTop, yBot, v10 / (v10 - v11))]);
          if (v01 * v11 < 0) pts.push([M.lerp(xLeft, xRight, v01 / (v01 - v11)), yBot]);
          if (v00 * v01 < 0) pts.push([xLeft, M.lerp(yTop, yBot, v00 / (v00 - v01))]);
          if (pts.length >= 2) {
            ctx.moveTo(vP.px(pts[0][0]), vP.py(pts[0][1]));
            ctx.lineTo(vP.px(pts[1][0]), vP.py(pts[1][1]));
          }
        }
      }
      ctx.stroke();
    }

    for (var i = 0; i < X.length; i++) {
      var wrong = (predictProb(X[i][0], X[i][1]) >= 0.5 ? 1 : 0) !== y[i];
      M.dot(ctx, vP.px(X[i][0]), vP.py(X[i][1]), 3.9 * s.sc, y[i] ? C.orange : C.blue, wrong ? C.red : C.surface, wrong ? 1.9 : 1.0);
    }
    ctx.restore();
  }

  function drawAll() {
    drawNet(); drawPlane();
    var pCount = paramCount();
    var msg;
    if (depth > 0 && actKind === 'linear') {
      msg = '<span class="r">Depth without bends collapses: all ' + pCount + ' weights simplify down into a single straight line (' + Math.round(accNow * 100) + '%).</span>';
    } else if (depth === 0) {
      msg = pattern === 'separable'
        ? '<span class="g">Two well-separated groups need zero hidden layers: Part 1\u2019s perceptron solves them cleanly.</span>'
        : '<span class="r">0 hidden layers can only draw one straight cut (' + Math.round(accNow * 100) + '%); overlapping rings or opposite corners stay mixed.</span>';
    } else if (depth === 1) {
      msg = pattern === 'bullseye' && accNow < 0.92
        ? '<span class="o">1 hidden layer (' + width + ' creases) can fence off one ring, not two nested rings at once (' + Math.round(accNow * 100) + '%).</span>'
        : '<span class="g">1 hidden layer of hinges folds the input plane so one straight cut at the end works (' + Math.round(accNow * 100) + '%).</span>';
    } else {
      msg = '<span class="g">Depth + bends: Layer 2 folds Layer 1\u2019s already-folded space, carving nested rings cleanly (' + Math.round(accNow * 100) + '%).</span>';
    }

    read.innerHTML = '<span class="k">hidden layers</span> ' + depth +
      '<span class="sep">&middot;</span><span class="k">unit bend</span> ' + (depth === 0 ? 'none (1 line)' : (actKind === 'linear' ? 'none (linear)' : actKind.toUpperCase())) +
      '<span class="sep">&middot;</span><span class="k">weights</span> ' + pCount +
      '<span class="sep">&middot;</span><span class="k">step</span> ' + stepCount +
      '<span class="sep">&middot;</span><span class="k">accuracy</span> ' + Math.round(accNow * 100) + '%' +
      '<div class="wread-interp">' + msg + '</div>';
  }

  buildData();
  trainSteps(260);
  netStage.draw = drawAll; planeStage.draw = drawAll;
  netStage.request(); planeStage.request();
  return { destroy: pause };
});
