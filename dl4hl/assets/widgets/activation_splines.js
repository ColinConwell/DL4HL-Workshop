/* ══════════════════════════════════════════════════════════════════
   Figure 1 — Bending the line: nonlinear hinges vs. linear collapse
   Stacking linear units only ever produces another straight line.
   Inserting a nonlinear activation (ReLU, GELU, Tanh) turns each hidden
   unit into a localized hinge or wave whose ensemble mean bends to fit
   curved developmental and physical trajectories.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('activation_splines', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 3,
    title: 'Bending the line: nonlinear hinges vs. linear collapse',
    hint: 'Switch activation between Linear, ReLU, GELU, and Tanh &middot; press play to fit hidden hinges &middot; drag knot handles along the axis'
  });

  var real = cfg.data && cfg.data.x && cfg.data.y;
  var seed = cfg.seed || 7;
  var xs = [], ys = [], xDom = [0, 1], yDom = [-1.2, 1.4];
  var xMean = 0.5, xScale = 0.5, yMean = 0.5, yScale = 0.5;
  var xn = [], yn = [];
  var K = cfg.units || 6;
  var actKind = cfg.activation || 'relu';

  /* Normalized ensemble-branch parameters:
     u_j(x) = yMean + yScale * (c[j] + v[j] * act(s[j] * (xNorm - kNorm[j])))
     yhat(x) = (1 / K) * sum_j u_j(x)
     Algebraically identical to b_out + sum_j w_j * act(a_j * x + b_1j),
     with explicit hidden + output biases and direct visual averaging. */
  var kNorm = [], slopes = [], weights = [], offsets = [];
  var adamM = null, adamV = null, adamT = 0;
  var stepCount = 0, mseNow = 0;
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

  function toXNorm(x) { return (x - xMean) / xScale; }
  function fromXNorm(u) { return xMean + u * xScale; }

  function resetAdam() {
    var dim = 4 * K;
    adamM = new Float64Array(dim);
    adamV = new Float64Array(dim);
    adamT = 0;
  }

  function loadData() {
    var rng = M.rng(seed);
    if (real) {
      xs = cfg.data.x.slice();
      ys = cfg.data.y.slice();
      var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
      var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
      var xpad = (xmax - xmin || 1) * 0.06, ypad = (ymax - ymin || 1) * 0.16;
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
      yDom = [-0.22, 1.28];
    }
    xMean = M.mean(xs);
    xScale = M.std(xs) || 1;
    yMean = M.mean(ys);
    yScale = M.std(ys) || 1;
    xn = xs.map(toXNorm);
    yn = ys.map(function (y) { return (y - yMean) / yScale; });
    initParams();
  }

  /* Closed-form readout solve for (offsets c_j, weights v_j) given current knots & slopes,
     used when dragging knots or after resetting, with optional initial dispersion at step 0. */
  function solveReadoutGivenKnots(alphaVal) {
    var alpha = alphaVal == null ? 0.025 : alphaVal;
    var P = 2 * K;
    var A = [], b = new Array(P).fill(0);
    for (var p = 0; p < P; p++) A.push(new Array(P).fill(0));
    var n = xn.length;
    for (var i = 0; i < n; i++) {
      var zRow = new Array(K);
      for (var j = 0; j < K; j++) zRow[j] = actFn(slopes[j] * (xn[i] - kNorm[j]));
      var rEns = new Array(P);
      for (var j2 = 0; j2 < K; j2++) {
        rEns[j2] = 1 / K;
        rEns[K + j2] = zRow[j2] / K;
      }
      var wEns = (1 - alpha) / n;
      for (var a = 0; a < P; a++) {
        b[a] += wEns * rEns[a] * yn[i];
        for (var c = a; c < P; c++) A[a][c] += wEns * rEns[a] * rEns[c];
      }
      var wInd = alpha / (K * n);
      for (var j3 = 0; j3 < K; j3++) {
        var idxC = j3, idxV = K + j3, zv = zRow[j3];
        b[idxC] += wInd * yn[i];
        b[idxV] += wInd * zv * yn[i];
        A[idxC][idxC] += wInd;
        A[idxC][idxV] += wInd * zv;
        A[idxV][idxV] += wInd * zv * zv;
      }
    }
    for (var d = 0; d < P; d++) {
      A[d][d] += d >= K ? 4e-4 : 1e-5;
      for (var e = 0; e < d; e++) A[d][e] = A[e][d];
    }
    var sol = M.solve(A, b);
    offsets = sol.slice(0, K);
    weights = sol.slice(K, 2 * K);
  }

  function initParams() {
    K = unitS ? unitS.get() : K;
    actKind = actT ? actT.get() : actKind;
    var rng = M.rng(seed * 13 + K);
    kNorm = []; slopes = []; offsets = []; weights = [];
    var uLo = toXNorm(xDom[0] + 0.12 * (xDom[1] - xDom[0]));
    var uHi = toXNorm(xDom[1] - 0.12 * (xDom[1] - xDom[0]));
    for (var j = 0; j < K; j++) {
      var kj = K === 1 ? 0 : uLo + (j / (K - 1)) * (uHi - uLo);
      kNorm.push(kj);
      slopes.push(actKind === 'linear' ? (0.85 + 0.3 * rng.next()) : 1.55);
    }
    solveReadoutGivenKnots(0.035);
    /* Add gentle zero-mean initial dispersion at step 0 so individual units are
       visibly distinct before training and converge cleanly over 180 GD steps. */
    for (var j2 = 0; j2 < K; j2++) {
      var sign = j2 % 2 === 0 ? 1 : -1;
      offsets[j2] += sign * 0.14 + rng.normal(0, 0.05);
      weights[j2] *= (1.0 + sign * 0.18 + rng.normal(0, 0.06));
    }
    stepCount = 0;
    resetAdam();
    computeMSE();
  }

  function unitCurve(j, x) {
    var u = toXNorm(x);
    return yMean + yScale * (offsets[j] + weights[j] * actFn(slopes[j] * (u - kNorm[j])));
  }

  function predict(x) {
    var u = toXNorm(x), s = 0;
    for (var j = 0; j < K; j++) {
      s += offsets[j] + weights[j] * actFn(slopes[j] * (u - kNorm[j]));
    }
    return yMean + yScale * (s / Math.max(1, K));
  }

  function computeMSE() {
    var s = 0;
    for (var i = 0; i < xs.length; i++) {
      var e = predict(xs[i]) - ys[i];
      s += e * e;
    }
    mseNow = s / (xs.length || 1);
  }

  /* Genuine Adam gradient descent over all 4K parameters (offsets, weights, slopes, knots)
     with bias terms included so by step 180 all units converge cleanly (and in Linear mode
     every red unit line converges directly onto the fitted OLS line). */
  function optimizeStep(iters) {
    var n = xn.length;
    var uMin = toXNorm(xDom[0]), uMax = toXNorm(xDom[1]);
    var alpha = actKind === 'linear' ? 0.06 : 0.022;
    var lr = 0.045;

    for (var it = 0; it < iters; it++) {
      var gC = new Float64Array(K), gV = new Float64Array(K);
      var gS = new Float64Array(K), gK = new Float64Array(K);

      for (var i = 0; i < n; i++) {
        var uPts = new Float64Array(K), aPts = new Float64Array(K), zPts = new Float64Array(K);
        var ens = 0;
        for (var j = 0; j < K; j++) {
          var z = slopes[j] * (xn[i] - kNorm[j]);
          var a = actFn(z);
          var uj = offsets[j] + weights[j] * a;
          zPts[j] = z; aPts[j] = a; uPts[j] = uj;
          ens += uj;
        }
        ens /= K;
        var dEns = 2 * (1 - alpha) * (ens - yn[i]) / (K * n);
        for (var j2 = 0; j2 < K; j2++) {
          var dInd = 2 * alpha * (uPts[j2] - yn[i]) / (K * n);
          var dU = dEns + dInd;
          var da = dActFn(zPts[j2]);
          gC[j2] += dU;
          gV[j2] += dU * aPts[j2];
          gS[j2] += dU * weights[j2] * da * (xn[i] - kNorm[j2]);
          gK[j2] += dU * weights[j2] * da * (-slopes[j2]);
        }
      }

      for (var j3 = 0; j3 < K; j3++) gV[j3] += 6e-4 * weights[j3];

      adamT++;
      var b1 = 0.9, b2 = 0.999, eps = 1e-8;
      var c1 = 1 - Math.pow(b1, adamT), c2 = 1 - Math.pow(b2, adamT);
      var grads = [gC, gV, gS, gK];
      for (var gIdx = 0; gIdx < 4; gIdx++) {
        for (var mIdx = 0; mIdx < K; mIdx++) {
          var flat = gIdx * K + mIdx;
          var gVal = grads[gIdx][mIdx];
          adamM[flat] = b1 * adamM[flat] + (1 - b1) * gVal;
          adamV[flat] = b2 * adamV[flat] + (1 - b2) * gVal * gVal;
          var step = lr * (adamM[flat] / c1) / (Math.sqrt(adamV[flat] / c2) + eps);
          if (gIdx === 0) offsets[mIdx] -= step;
          else if (gIdx === 1) weights[mIdx] -= step;
          else if (gIdx === 2) slopes[mIdx] = M.clamp(slopes[mIdx] - step, 0.25, 5.5);
          else kNorm[mIdx] = M.clamp(kNorm[mIdx] - step * 0.65, uMin, uMax);
        }
      }
      stepCount++;
    }
    computeMSE();
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.45);
  M.mini(split.a, 'combined network curve + individual hidden unit bends');
  M.mini(split.b, 'what a single hidden unit does');
  var hf = function (w) { return M.clamp(w * 0.66, 225, 310); };
  var left = M.stage(split.a, { height: hf, ref: 460, label: 'Data scatter with hidden basis hinges and summed network fit' });
  var right = M.stage(split.b, { height: function () { return left.h || 250; }, ref: 320, label: 'Activation function shape for a single hidden unit' });

  M.legend(root, [
    { color: C.muted, label: real ? 'observed child vocabulary' : 'observed data' },
    { color: C.orange, label: 'individual hidden unit bends', line: true },
    { color: C.blue, label: 'combined network curve', line: true }
  ]);

  var row = M.row(root);
  var playBtn = M.button(row, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Optimize hidden knots, slopes, and biases' });
  M.button(row, 'step', function () { pause(); optimizeStep(8); drawAll(); }, { aria: 'Advance 8 gradient steps' });
  var actT = M.toggle(row, {
    label: 'Activation', value: actKind,
    options: [['linear', 'Linear'], ['relu', 'ReLU'], ['gelu', 'GELU'], ['tanh', 'Tanh']],
    onChange: function (v) { pause(); actKind = v; initParams(); drawAll(); }
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
  function play() { if (stepCount >= 180) initParams(); loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
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
        var dx = p.x - vL.px(fromXNorm(kNorm[j])), dy = p.y - yAxisPx;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestD) { bestD = d; best = j; }
      }
      dragIdx = best;
      return best >= 0;
    },
    drag: function (p) {
      if (dragIdx < 0) return;
      var xx = M.clamp(vL.ix(p.x), xDom[0], xDom[1]);
      kNorm[dragIdx] = toXNorm(xx);
      solveReadoutGivenKnots(0.03);
      resetAdam();
      computeMSE();
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
    for (var i = 0; i < xs.length; i++) {
      M.dot(ctx, vL.px(xs[i]), vL.py(ys[i]), 3.1 * s.sc, M.rgba(C.muted, 0.55));
    }

    /* Individual hidden unit curves u_j(x) whose mean equals yhat(x) */
    var G = 120;
    ctx.setLineDash([4 * s.sc, 3 * s.sc]);
    ctx.lineWidth = 1.25 * s.sc;
    for (var j = 0; j < K; j++) {
      ctx.strokeStyle = M.rgba(C.orange, 0.52);
      ctx.beginPath();
      for (var g = 0; g <= G; g++) {
        var xx = xDom[0] + (g / G) * (xDom[1] - xDom[0]);
        var px = vL.px(xx), py = vL.py(unitCurve(j, xx));
        if (g === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    /* Combined network readout curve */
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
      M.dot(ctx, vL.px(fromXNorm(kNorm[k])), hy, (dragIdx === k ? 5.6 : 4.3) * s.sc, C.orange, C.surface, 1.4 * s.sc);
    }
    ctx.restore();
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    vR.update();
    M.clear(ctx, s);
    M.axes(ctx, vR, s, { xlabel: 'weighted input', ylabel: 'unit output', nx: 4, ny: 4 });
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

    var tag = actKind === 'linear' ? 'Linear: average of lines = 1 line' :
      actKind === 'relu' ? 'ReLU: sharp hinge at threshold' :
      actKind === 'gelu' ? 'GELU: smooth gradual bend' : 'Tanh: smooth S-curve';
    M.pill(ctx, s, tag, vR.L + 6 * s.sc, vR.T + 14 * s.sc, { color: actKind === 'linear' ? C.red : C.ink });
  }

  function drawAll() {
    drawLeft(); drawRight();
    var rmse = Math.sqrt(mseNow);
    var note = actKind === 'linear'
      ? '<span class="r">Linear collapse: even as GD aligns all ' + K + ' linear units onto the OLS slope, their combination remains a single straight line.</span>'
      : '<span class="g">' + K + ' ' + actKind.toUpperCase() + ' hidden units optimize their knots, slopes, and biases to bend locally around the trajectory.</span>';
    read.innerHTML = '<span class="k">activation</span> ' + actKind.toUpperCase() +
      '<span class="sep">&middot;</span><span class="k">hidden units</span> ' + K +
      '<span class="sep">&middot;</span><span class="k">RMSE</span> ' + M.f(rmse, 3) +
      '<span class="sep">&middot;</span><span class="k">GD steps</span> ' + stepCount +
      '<div class="wread-interp">' + note + '</div>';
  }

  loadData();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: pause };
});
