/* ══════════════════════════════════════════════════════════════════
   Figure 2 — Solving XOR: how a hidden layer folds the input plane
   When class membership depends on an interaction (XOR or concentric
   rings), no single straight line separates the classes. One hidden
   ReLU layer creases and folds the plane so opposite corners land on
   the same side of a final linear readout.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('xor_fold', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 2,
    title: 'The XOR problem: folding space so one straight line suffices',
    hint: 'Compare 0 hidden layers (linear) vs. 1 hidden layer &middot; press play to train &middot; scrub Fold morph to watch space crease'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 4;
  var X = [], y = [], H = cfg.hidden != null ? cfg.hidden : 4;
  var W1 = [], b1 = [], W2 = [], b2 = 0, wLin = [0.4, -0.4, 0];
  var stepCount = 0, accNow = 0, morph = 1.0;

  function loadData() {
    var rng = M.rng(seed);
    if (real) {
      X = cfg.data.X.map(function (p) { return p.slice(); });
      y = cfg.data.y.slice();
    } else {
      X = []; y = [];
      var pat = patT ? patT.get() : (cfg.pattern || 'xor');
      if (pat === 'rings') {
        var c = M.sim.circles(rng, { n: 42, noise: 0.07 });
        X = c.X; y = c.y;
      } else {
        [[-0.65, -0.65, 0], [0.65, 0.65, 0], [-0.65, 0.65, 1], [0.65, -0.65, 1]].forEach(function (q) {
          for (var i = 0; i < 16; i++) {
            X.push([q[0] + rng.normal(0, 0.15), q[1] + rng.normal(0, 0.15)]);
            y.push(q[2]);
          }
        });
      }
    }
    initNet();
  }

  function initNet() {
    H = archT ? (archT.get() === 'linear' ? 0 : (unitS ? unitS.get() : 4)) : H;
    var rng = M.rng(seed * 17 + H + 3);
    wLin = [rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 0];
    W1 = []; b1 = []; W2 = []; b2 = 0;
    for (var j = 0; j < Math.max(2, H); j++) {
      var a = (2 * Math.PI * j) / Math.max(2, H) + 0.35;
      W1.push([1.45 * Math.cos(a), 1.45 * Math.sin(a)]);
      b1.push(rng.uniform(-0.25, 0.25));
      W2.push((j % 2 ? 1 : -1) * rng.uniform(0.6, 1.1));
    }
    stepCount = 0;
    evalAcc();
  }

  function hiddenVec(p) {
    var h = new Array(H);
    for (var j = 0; j < H; j++) {
      var z = W1[j][0] * p[0] + W1[j][1] * p[1] + b1[j];
      h[j] = z > 0 ? z : 0.05 * z;
    }
    return h;
  }

  function forward(p) {
    if (H === 0) {
      return { h: [p[0], p[1]], prob: M.sigmoid(wLin[0] * p[0] + wLin[1] * p[1] + wLin[2]) };
    }
    var h = hiddenVec(p), s = b2;
    for (var j = 0; j < H; j++) s += W2[j] * h[j];
    return { h: h, prob: M.sigmoid(s) };
  }

  function trainStep(iters) {
    var lr = 0.2, n = X.length;
    for (var it = 0; it < iters; it++) {
      if (H === 0) {
        var g0 = 0, g1 = 0, g2 = 0;
        for (var i = 0; i < n; i++) {
          var e0 = forward(X[i]).prob - y[i];
          g0 += e0 * X[i][0]; g1 += e0 * X[i][1]; g2 += e0;
        }
        wLin[0] -= lr * (g0 / n); wLin[1] -= lr * (g1 / n); wLin[2] -= lr * (g2 / n);
      } else {
        var gW1 = W1.map(function () { return [0, 0]; }), gb1 = new Float64Array(H);
        var gW2 = new Float64Array(H), gb2 = 0;
        for (var k = 0; k < n; k++) {
          var fw = forward(X[k]), err = fw.prob - y[k];
          gb2 += err;
          for (var j = 0; j < H; j++) {
            gW2[j] += err * fw.h[j];
            var z = W1[j][0] * X[k][0] + W1[j][1] * X[k][1] + b1[j];
            var dh = err * W2[j] * (z > 0 ? 1 : 0.05);
            gW1[j][0] += dh * X[k][0];
            gW1[j][1] += dh * X[k][1];
            gb1[j] += dh;
          }
        }
        b2 -= lr * (gb2 / n);
        for (var j2 = 0; j2 < H; j2++) {
          W2[j2] -= lr * (gW2[j2] / n + 0.001 * W2[j2]);
          W1[j2][0] -= lr * (gW1[j2][0] / n + 0.001 * W1[j2][0]);
          W1[j2][1] -= lr * (gW1[j2][1] / n + 0.001 * W1[j2][1]);
          b1[j2] -= lr * (gb1[j2] / n);
        }
      }
      stepCount++;
    }
    evalAcc();
  }

  function evalAcc() {
    var c = 0;
    for (var i = 0; i < X.length; i++) if ((forward(X[i]).prob >= 0.5 ? 1 : 0) === y[i]) c++;
    accNow = c / (X.length || 1);
  }

  /* Project hidden features to 2D where axis 1 is the readout direction W2 */
  function folded2D() {
    if (H === 0) return { pts: X.map(function (p) { return p.slice(); }), sepX: null };
    var hs = X.map(hiddenVec);
    var wNorm = Math.sqrt(M.sum(W2.slice(0, H).map(function (v) { return v * v; }))) || 1;
    var u1 = W2.slice(0, H).map(function (v) { return v / wNorm; });
    /* Orthogonal second direction via PCA of residual */
    var res = hs.map(function (h) {
      var dot = 0;
      for (var j = 0; j < H; j++) dot += h[j] * u1[j];
      return h.map(function (v, j) { return v - dot * u1[j]; });
    });
    var u2 = res[0].map(function (_, j) { return j === 1 ? 1 : 0; });
    if (H >= 2) {
      var p = M.pca(res, 1);
      if (p.components.length) u2 = p.components[0];
    }
    var raw = hs.map(function (h) {
      var a = 0, b = 0;
      for (var j = 0; j < H; j++) { a += h[j] * u1[j]; b += h[j] * u2[j]; }
      return [a, b];
    });
    /* Normalize scale nicely to [-1.1, 1.1] for smooth morph with input X */
    var bb = M.bounds(raw, 0.02);
    var cx = (bb.x[0] + bb.x[1]) / 2, cy = (bb.y[0] + bb.y[1]) / 2;
    var sx = (bb.x[1] - bb.x[0]) / 1.9 || 1, sy = (bb.y[1] - bb.y[0]) / 1.9 || 1;
    var sMax = Math.max(sx, sy);
    var normPts = raw.map(function (q, idx) {
      var tx = (q[0] - cx) / sMax, ty = (q[1] - cy) / sMax;
      return [M.lerp(X[idx][0], tx, morph), M.lerp(X[idx][1], ty, morph)];
    });
    var sepRaw = -b2 / wNorm;
    return { pts: normPts, sepX: (sepRaw - cx) / sMax };
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, 'input space (x\u2081, x\u2082): hinge creases + decision region');
  M.mini(split.b, 'hidden representation: folded space + linear readout');
  var hf = function (w) { return M.clamp(w * 0.84, 225, 325); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: 'Input plane with XOR quadrants and hidden hinge creases' });
  var right = M.stage(split.b, { height: hf, ref: 400, label: 'Folded hidden representation showing linear separability' });

  M.legend(root, [
    { color: C.blue, label: 'class A (same signs)' },
    { color: C.orange, label: 'class B (opposite signs)' },
    { color: C.muted, label: 'hidden ReLU creases', line: true },
    { color: C.ink, label: 'linear readout boundary', line: true }
  ]);

  var row1 = M.row(root);
  var playBtn = M.button(row1, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Train or pause network' });
  M.button(row1, 'step', function () { pause(); trainStep(8); drawAll(); }, { aria: 'Run 8 gradient steps' });
  var archT = M.toggle(row1, {
    label: 'Model', value: H === 0 ? 'linear' : 'mlp',
    options: [['linear', '0 hidden (linear)'], ['mlp', '1 hidden layer']],
    onChange: function () { pause(); initNet(); drawAll(); }
  });
  var patT = null;
  if (!real) {
    patT = M.toggle(row1, {
      label: 'Data', value: cfg.pattern || 'xor',
      options: [['xor', 'XOR'], ['rings', 'Rings']],
      onChange: function () { pause(); loadData(); drawAll(); }
    });
  }

  var row2 = M.row(root);
  var unitS = M.slider(row2, {
    label: 'Hidden units', min: 2, max: 8, step: 1, value: Math.max(2, H || 4), accent: 'blue',
    onInput: function () { if (archT.get() === 'mlp') { pause(); initNet(); drawAll(); } }
  });
  var morphS = M.slider(row2, {
    label: 'Fold morph', min: 0, max: 1, step: 0.02, value: 1, accent: 'orange',
    fmt: function (v) { return Math.round(v * 100) + '%'; },
    onInput: function (v) { morph = v; drawAll(); }
  });
  M.spacer(row2);
  M.resetButton(row2, function () {
    pause();
    archT.set('mlp', true);
    unitS.set(4, true);
    morphS.set(1, true);
    morph = 1;
    if (patT) patT.set(cfg.pattern || 'xor', true);
    loadData();
    drawAll();
  });
  var read = M.readout(root);

  var loop = M.loop(function () {
    trainStep(6);
    drawAll();
    if (stepCount >= 420 || accNow >= 0.995) { pause(); return false; }
    return true;
  });
  function play() { if (stepCount >= 420 || accNow >= 0.995) initNet(); loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  var vL = M.view(left, { x: [-1.35, 1.35], y: [-1.35, 1.35], equal: true, pad: { l: 34, r: 12, t: 12, b: 30 } });
  var vR = M.view(right, { x: [-1.35, 1.35], y: [-1.35, 1.35], equal: true, pad: { l: 34, r: 12, t: 12, b: 30 } });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain([-1.35, 1.35], [-1.35, 1.35]);
    M.clear(ctx, s);
    M.field(ctx, vL, 60, 60, function (xx, yy) {
      var p = forward([xx, yy]).prob, c = M.diverge(p);
      return [c[0], c[1], c[2], 145];
    });
    M.axes(ctx, vL, s, { xlabel: 'feature x\u2081', ylabel: 'feature x\u2082', nx: 3, ny: 3, grid: false });

    ctx.save(); vL.clip(ctx);
    if (H > 0) {
      ctx.setLineDash([4 * s.sc, 3 * s.sc]);
      ctx.lineWidth = 1.2 * s.sc;
      for (var j = 0; j < H; j++) {
        var w = W1[j], b = b1[j];
        ctx.strokeStyle = M.rgba(C.ink, 0.42);
        ctx.beginPath();
        if (Math.abs(w[1]) > Math.abs(w[0])) {
          ctx.moveTo(vL.px(vL.x0), vL.py(-(b + w[0] * vL.x0) / w[1]));
          ctx.lineTo(vL.px(vL.x1), vL.py(-(b + w[0] * vL.x1) / w[1]));
        } else if (Math.abs(w[0]) > 1e-6) {
          ctx.moveTo(vL.px(-(b + w[1] * vL.y0) / w[0]), vL.py(vL.y0));
          ctx.lineTo(vL.px(-(b + w[1] * vL.y1) / w[0]), vL.py(vL.y1));
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    for (var i = 0; i < X.length; i++) {
      var wrong = (forward(X[i]).prob >= 0.5 ? 1 : 0) !== y[i];
      M.dot(ctx, vL.px(X[i][0]), vL.py(X[i][1]), 4.0 * s.sc, y[i] ? C.orange : C.blue, wrong ? C.red : C.surface, wrong ? 2 : 1);
    }
    ctx.restore();
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    vR.setDomain([-1.35, 1.35], [-1.35, 1.35]);
    M.clear(ctx, s);
    M.axes(ctx, vR, s, {
      xlabel: H === 0 ? 'unchanged x\u2081' : 'readout direction w\u00b7h',
      ylabel: H === 0 ? 'unchanged x\u2082' : 'orthogonal hidden axis',
      nx: 3, ny: 3
    });

    var fd = folded2D();
    ctx.save(); vR.clip(ctx);
    if (H > 0 && fd.sepX != null && morph > 0.15) {
      var sx = vR.px(fd.sepX);
      ctx.strokeStyle = M.rgba(C.ink, Math.min(0.85, morph));
      ctx.lineWidth = 2.0 * s.sc;
      ctx.beginPath(); ctx.moveTo(sx, vR.T); ctx.lineTo(sx, vR.B); ctx.stroke();
    } else if (H === 0 && (Math.abs(wLin[0]) + Math.abs(wLin[1]) > 1e-4)) {
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 2.0 * s.sc;
      ctx.beginPath();
      if (Math.abs(wLin[1]) > Math.abs(wLin[0])) {
        ctx.moveTo(vR.px(vR.x0), vR.py(-(wLin[2] + wLin[0] * vR.x0) / wLin[1]));
        ctx.lineTo(vR.px(vR.x1), vR.py(-(wLin[2] + wLin[0] * vR.x1) / wLin[1]));
      } else {
        ctx.moveTo(vR.px(-(wLin[2] + wLin[1] * vR.y0) / wLin[0]), vR.py(vR.y0));
        ctx.lineTo(vR.px(-(wLin[2] + wLin[1] * vR.y1) / wLin[0]), vR.py(vR.y1));
      }
      ctx.stroke();
    }
    for (var i = 0; i < fd.pts.length; i++) {
      M.dot(ctx, vR.px(fd.pts[i][0]), vR.py(fd.pts[i][1]), 4.0 * s.sc, y[i] ? C.orange : C.blue, C.surface, 1);
    }
    ctx.restore();
  }

  function drawAll() {
    drawLeft(); drawRight();
    var msg = H === 0
      ? '<span class="r">0 hidden layers: a single straight line cannot separate opposite XOR quadrants</span>'
      : accNow >= 0.96
        ? '<span class="g">solved: hidden ReLU creases fold opposite quadrants onto the same side of the readout</span>'
        : '<span class="b">press play to move the hidden creases until the folded space separates cleanly</span>';
    read.innerHTML = '<span class="k">architecture</span> ' + (H === 0 ? 'linear perceptron' : ('2-layer MLP (' + H + ' ReLU units)')) +
      '<span class="sep">&middot;</span><span class="k">step</span> ' + stepCount +
      '<span class="sep">&middot;</span><span class="k">accuracy</span> ' + Math.round(accNow * 100) + '%' +
      '<span class="sep">&middot;</span>' + msg;
  }

  loadData();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: pause };
});
