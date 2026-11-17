/* ══════════════════════════════════════════════════════════════════
   Figure 1 — Bending the line: nonlinear hinges vs. linear collapse
   Stacking linear units only ever produces another straight line.
   Inserting a nonlinear activation (ReLU, GELU, Tanh) turns each hidden
   unit into a localized hinge or wave whose weighted sum bends to fit
   curved developmental and physical trajectories.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('activation_splines', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 1,
    title: 'Bending the line: nonlinear activations vs. linear collapse',
    hint: 'Switch activation between Linear, ReLU, GELU, and Tanh &middot; press play to optimize hidden hinges or drag knot handles along the axis'
  });

  var real = cfg.data && cfg.data.x && cfg.data.y;
  var seed = cfg.seed || 7;
  var xs = [], ys = [], xDom = [0, 1], yDom = [-1.2, 1.4];
  var K = cfg.units || 6;
  var actKind = cfg.activation || 'relu';
  var knots = [], slopes = [], weights = [], bias = 0, stepCount = 0, mseNow = 0;
  var dragIdx = -1;

  function actFn(z) {
    if (actKind === 'linear') return z;
    if (actKind === 'gelu') return M.gelu(z);
    if (actKind === 'tanh') return Math.tanh(z);
    return z > 0 ? z : 0;
  }
  function dActFn(z) {
    if (actKind === 'linear') return 1;
    if (actKind === 'tanh') { var t = Math.tanh(z); return 1 - t * t; }
    if (actKind === 'gelu') {
      var eps = 1e-4;
      return (M.gelu(z + eps) - M.gelu(z - eps)) / (2 * eps);
    }
    return z > 0 ? 1 : 0;
  }

  function loadData() {
    var rng = M.rng(seed);
    if (real) {
      xs = cfg.data.x.slice();
      ys = cfg.data.y.slice();
      var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
      var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
      var xpad = (xmax - xmin || 1) * 0.06, ypad = (ymax - ymin || 1) * 0.14;
      xDom = [xmin - xpad, xmax + xpad];
      yDom = [ymin - ypad, ymax + ypad];
    } else {
      xs = []; ys = [];
      var n = 64;
      for (var i = 0; i < n; i++) {
        var u = (i + 0.5) / n;
        var yTrue = 0.12 + 0.82 / (1 + Math.exp(-11 * (u - 0.48))) + 0.16 * Math.sin(u * Math.PI * 2.5);
        xs.push(u);
        ys.push(yTrue + rng.normal(0, 0.065));
      }
      xDom = [-0.04, 1.04];
      yDom = [-0.18, 1.22];
    }
    initParams();
  }

  function initParams() {
    K = unitS ? unitS.get() : K;
    actKind = actT ? actT.get() : actKind;
    knots = []; slopes = [];
    var lo = xDom[0] + 0.1 * (xDom[1] - xDom[0]);
    var hi = xDom[1] - 0.1 * (xDom[1] - xDom[0]);
    var span = (hi - lo) || 1;
    for (var j = 0; j < K; j++) {
      knots.push(K === 1 ? (lo + hi) / 2 : lo + (j / (K - 1)) * span);
      slopes.push((j % 2 === 0 ? 1 : 1) * (2.6 / span));
    }
    stepCount = 0;
    solveReadout();
  }

  function basisRow(x) {
    var row = [1];
    for (var j = 0; j < K; j++) {
      var z = slopes[j] * (x - knots[j]);
      row.push(actFn(z));
    }
    return row;
  }

  function solveReadout() {
    var Phi = xs.map(basisRow);
    var lam = actKind === 'linear' ? 1e-3 : 1e-4;
    var wAll = M.lstsq(Phi, ys, lam);
    bias = wAll[0];
    weights = wAll.slice(1);
    computeMSE();
  }

  function predict(x) {
    var s = bias;
    for (var j = 0; j < K; j++) s += weights[j] * actFn(slopes[j] * (x - knots[j]));
    return s;
  }

  function computeMSE() {
    var s = 0;
    for (var i = 0; i < xs.length; i++) {
      var e = predict(xs[i]) - ys[i];
      s += e * e;
    }
    mseNow = s / (xs.length || 1);
  }

  function optimizeStep(iters) {
    var span = (xDom[1] - xDom[0]) || 1;
    var lrK = 0.045 * span, lrS = 0.25 / span;
    for (var it = 0; it < iters; it++) {
      var gk = new Float64Array(K), gs = new Float64Array(K);
      for (var i = 0; i < xs.length; i++) {
        var err = predict(xs[i]) - ys[i];
        for (var j = 0; j < K; j++) {
          var z = slopes[j] * (xs[i] - knots[j]);
          var da = dActFn(z);
          gk[j] += 2 * err * weights[j] * da * (-slopes[j]) / xs.length;
          gs[j] += 2 * err * weights[j] * da * (xs[i] - knots[j]) / xs.length;
        }
      }
      for (var j2 = 0; j2 < K; j2++) {
        knots[j2] = M.clamp(knots[j2] - lrK * gk[j2], xDom[0], xDom[1]);
        slopes[j2] = M.clamp(slopes[j2] - lrS * gs[j2], 0.4 / span, 9.0 / span);
      }
      solveReadout();
      stepCount++;
    }
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.45);
  M.mini(split.a, 'summed network prediction + individual hidden basis curves');
  M.mini(split.b, 'single hidden unit response \u03c3(z)');
  var hf = function (w) { return M.clamp(w * 0.66, 225, 310); };
  var left = M.stage(split.a, { height: hf, ref: 460, label: 'Data scatter with hidden basis hinges and summed network fit' });
  var right = M.stage(split.b, { height: function () { return left.h || 250; }, ref: 320, label: 'Activation function shape for a single hidden unit' });

  M.legend(root, [
    { color: C.muted, label: real ? 'observed child vocabulary' : 'observed data' },
    { color: C.orange, label: 'weighted hidden units w\u2c7c\u00b7\u03c3(z\u2c7c)', line: true },
    { color: C.blue, label: 'network sum \u0177(x)', line: true }
  ]);

  var row = M.row(root);
  var playBtn = M.button(row, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Optimize hidden knots and slopes' });
  M.button(row, 'step', function () { pause(); optimizeStep(6); drawAll(); }, { aria: 'Advance 6 gradient steps' });
  var actT = M.toggle(row, {
    label: 'Activation', value: actKind,
    options: [['linear', 'Linear'], ['relu', 'ReLU'], ['gelu', 'GELU'], ['tanh', 'Tanh']],
    onChange: function (v) { pause(); actKind = v; solveReadout(); drawAll(); }
  });
  var unitS = M.slider(row, {
    label: 'Hidden units', min: 1, max: 12, step: 1, value: K, accent: 'blue',
    onInput: function () { pause(); initParams(); drawAll(); }
  });
  M.spacer(row);
  M.resetButton(row, function () {
    pause();
    actKind = cfg.activation || 'relu';
    actT.set(actKind, true);
    unitS.set(cfg.units || 6, true);
    loadData();
    drawAll();
  });
  var read = M.readout(root);

  var loop = M.loop(function () {
    optimizeStep(3);
    drawAll();
    if (stepCount >= 180) { pause(); return false; }
    return true;
  });
  function play() { if (stepCount >= 180) stepCount = 0; loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  var vL = M.view(left, { x: xDom, y: yDom, pad: { l: 46, r: 14, t: 14, b: 36 } });
  var vR = M.view(right, { x: [-3, 3], y: [-1.4, 2.6], pad: { l: 38, r: 14, t: 14, b: 36 } });

  M.pointer(left.cv, {
    down: function (p) {
      pause();
      vL.update();
      var best = -1, bestD = 18 * left.sc;
      var yAxisPx = vL.B - 8 * left.sc;
      for (var j = 0; j < K; j++) {
        var dx = p.x - vL.px(knots[j]), dy = p.y - yAxisPx;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestD) { bestD = d; best = j; }
      }
      dragIdx = best;
      return best >= 0;
    },
    drag: function (p) {
      if (dragIdx < 0) return;
      knots[dragIdx] = M.clamp(vL.ix(p.x), xDom[0], xDom[1]);
      solveReadout();
      drawAll();
    },
    up: function () { dragIdx = -1; }
  });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain(xDom, yDom);
    M.clear(ctx, s);
    M.axes(ctx, vL, s, {
      xlabel: real ? (cfg.data.xlabel || 'input x') : 'input x',
      ylabel: real ? (cfg.data.ylabel || 'outcome y') : 'outcome y',
      nx: 5, ny: 4
    });

    ctx.save(); vL.clip(ctx);
    /* Observations */
    for (var i = 0; i < xs.length; i++) {
      M.dot(ctx, vL.px(xs[i]), vL.py(ys[i]), 3.1 * s.sc, M.rgba(C.muted, 0.55));
    }

    /* Individual weighted hidden basis functions (centered around bias/K) */
    var G = 120;
    ctx.setLineDash([4 * s.sc, 3 * s.sc]);
    ctx.lineWidth = 1.2 * s.sc;
    for (var j = 0; j < K; j++) {
      ctx.strokeStyle = M.rgba(C.orange, 0.45);
      ctx.beginPath();
      for (var g = 0; g <= G; g++) {
        var xx = xDom[0] + (g / G) * (xDom[1] - xDom[0]);
        var yj = (bias / Math.max(1, K)) + weights[j] * actFn(slopes[j] * (xx - knots[j]));
        var px = vL.px(xx), py = vL.py(yj);
        if (g === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    /* Combined network curve */
    ctx.strokeStyle = C.blue;
    ctx.lineWidth = 2.6 * s.sc;
    ctx.beginPath();
    for (var g2 = 0; g2 <= G; g2++) {
      var xx2 = xDom[0] + (g2 / G) * (xDom[1] - xDom[0]);
      var px2 = vL.px(xx2), py2 = vL.py(predict(xx2));
      if (g2 === 0) ctx.moveTo(px2, py2); else ctx.lineTo(px2, py2);
    }
    ctx.stroke();

    /* Draggable knot handles near bottom axis */
    var hy = vL.B - 8 * s.sc;
    for (var k = 0; k < K; k++) {
      M.dot(ctx, vL.px(knots[k]), hy, (dragIdx === k ? 5.6 : 4.3) * s.sc, C.orange, C.surface, 1.4 * s.sc);
    }
    ctx.restore();
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    vR.update();
    M.clear(ctx, s);
    M.axes(ctx, vR, s, { xlabel: 'pre-activation z', ylabel: 'activation \u03c3(z)', nx: 4, ny: 4 });
    ctx.save(); vR.clip(ctx);
    ctx.strokeStyle = M.rgba(C.muted, 0.45);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(vR.L, vR.py(0)); ctx.lineTo(vR.R, vR.py(0));
    ctx.moveTo(vR.px(0), vR.T); ctx.lineTo(vR.px(0), vR.B);
    ctx.stroke();

    ctx.strokeStyle = actKind === 'linear' ? C.red : C.orange;
    ctx.lineWidth = 2.5 * s.sc;
    ctx.beginPath();
    for (var i = 0; i <= 100; i++) {
      var z = -3 + (i / 100) * 6;
      var px = vR.px(z), py = vR.py(actFn(z));
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    if (actKind === 'relu') M.dot(ctx, vR.px(0), vR.py(0), 4.5 * s.sc, C.orange, C.surface, 1.4 * s.sc);
    ctx.restore();

    var tag = actKind === 'linear' ? 'Linear: sum of lines = 1 line' :
      actKind === 'relu' ? 'ReLU: sharp hinge at z = 0' :
      actKind === 'gelu' ? 'GELU: smooth probabilistic bend' : 'Tanh: saturating S-curve';
    M.pill(ctx, s, tag, vR.L + 6 * s.sc, vR.T + 14 * s.sc, { color: actKind === 'linear' ? C.red : C.ink });
  }

  function drawAll() {
    drawLeft(); drawRight();
    var rmse = Math.sqrt(mseNow);
    var note = actKind === 'linear'
      ? '<span class="r">linear collapse: ' + K + ' linear units still sum to a single straight line</span>'
      : '<span class="g">' + K + ' ' + actKind.toUpperCase() + ' hinges combine to bend locally around the data</span>';
    read.innerHTML = '<span class="k">activation</span> ' + actKind.toUpperCase() +
      '<span class="sep">&middot;</span><span class="k">hidden units</span> ' + K +
      '<span class="sep">&middot;</span><span class="k">RMSE</span> ' + M.f(rmse, 3) +
      '<span class="sep">&middot;</span><span class="k">GD steps</span> ' + stepCount +
      '<span class="sep">&middot;</span>' + note;
  }

  loadData();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: pause };
});
