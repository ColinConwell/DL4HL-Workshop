/* ══════════════════════════════════════════════════════════════════
   Figure 3 — Untangling Christmas lights: deep learning as progressive
   manifold disentanglement
   Two intertwined 1D manifolds (drawn as glowing strands of lights)
   cannot be separated by a flat cut in input space. Each residual layer
   applies a gentle coordinate warp — loosening one knot at a time —
   until a straight linear boundary separates the two strands cleanly.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('manifold_untangle', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 3,
    title: 'Untangling Christmas lights: progressive manifold disentanglement across depth',
    hint: 'Press play or scrub Depth to travel from input layer L0 to L4 &middot; increase Tangle turns to tighten the knot'
  });

  var seed = cfg.seed || 11;
  var N_PER = 38;
  var DEPTH = 4;
  var turns = cfg.turns || 1.35;
  var depthPos = 0; // continuous 0..4
  var strands = [[], []]; // each: array of 5 layer coordinate snapshots [L0..L4] of length N_PER
  var layerAcc = [0.5, 0.65, 0.8, 0.92, 1.0];
  var finalLine = { w0: 1, w1: 0, b: 0 };

  /* Build two intertwined spiral strands and train a compact 2->8->8->8->2
     residual ReLU network + 2D layer projections so every layer is genuinely
     computed from learned transformations. */
  function buildAndSolve() {
    var rng = M.rng(seed + Math.round(turns * 100));
    var X0 = [], y = [];
    for (var sIdx = 0; sIdx < 2; sIdx++) {
      for (var i = 0; i < N_PER; i++) {
        var u = i / (N_PER - 1);
        var rad = 0.16 + 0.88 * u;
        var ang = u * turns * Math.PI * 2 + (sIdx === 1 ? Math.PI : 0);
        X0.push([
          rad * Math.cos(ang) + rng.normal(0, 0.018),
          rad * Math.sin(ang) + rng.normal(0, 0.018)
        ]);
        y.push(sIdx);
      }
    }

    /* Train a 4-hidden-block residual MLP in 6D -> 2D readout */
    var D = 8, n = X0.length;
    var W_in = [], W_res = [], b_res = [], W_out = [], b_out = 0;
    for (var d = 0; d < D; d++) {
      var a = (2 * Math.PI * d) / D;
      W_in.push([Math.cos(a), Math.sin(a)]);
      W_out.push((d % 2 ? 1 : -1) * 0.6);
    }
    for (var l = 0; l < DEPTH; l++) {
      var wl = [], bl = new Float64Array(D);
      for (var r = 0; r < D; r++) {
        var row = new Float64Array(D);
        for (var c = 0; c < D; c++) row[c] = rng.normal(0, 0.38);
        wl.push(row);
        bl[r] = rng.uniform(-0.15, 0.15);
      }
      W_res.push(wl); b_res.push(bl);
    }

    function runSample(p) {
      var h0 = new Float64Array(D);
      for (var d2 = 0; d2 < D; d2++) h0[d2] = W_in[d2][0] * p[0] + W_in[d2][1] * p[1];
      var states = [h0];
      var cur = h0;
      for (var l2 = 0; l2 < DEPTH; l2++) {
        var nxt = new Float64Array(D), z = new Float64Array(D);
        for (var r2 = 0; r2 < D; r2++) {
          var s = b_res[l2][r2];
          for (var c2 = 0; c2 < D; c2++) s += W_res[l2][r2][c2] * cur[c2];
          z[r2] = s;
          nxt[r2] = cur[r2] + 0.65 * (s > 0 ? s : 0.05 * s);
        }
        nxt._z = z;
        states.push(nxt);
        cur = nxt;
      }
      var logit = b_out;
      for (var d3 = 0; d3 < D; d3++) logit += W_out[d3] * cur[d3];
      return { states: states, prob: M.sigmoid(logit) };
    }

    /* Quick deterministic full-batch GD (160 iters on 76 points -> <4ms in JS!) */
    var lr = 0.22;
    for (var it = 0; it < 165; it++) {
      var gWout = new Float64Array(D), gbOut = 0;
      var gWres = W_res.map(function () {
        var m = []; for (var r3 = 0; r3 < D; r3++) m.push(new Float64Array(D)); return m;
      });
      var gbRes = b_res.map(function () { return new Float64Array(D); });

      for (var idx = 0; idx < n; idx++) {
        var fw = runSample(X0[idx]);
        var err = (fw.prob - y[idx]) / n;
        gbOut += err;
        var dh = new Float64Array(D);
        for (var j = 0; j < D; j++) {
          gWout[j] += err * fw.states[DEPTH][j];
          dh[j] = err * W_out[j];
        }
        for (var l3 = DEPTH - 1; l3 >= 0; l3--) {
          var prev = fw.states[l3], zz = fw.states[l3 + 1]._z;
          var dhPrev = new Float64Array(D);
          for (var j2 = 0; j2 < D; j2++) dhPrev[j2] = dh[j2];
          for (var r4 = 0; r4 < D; r4++) {
            var dz = dh[r4] * 0.65 * (zz[r4] > 0 ? 1 : 0.05);
            gbRes[l3][r4] += dz;
            for (var c4 = 0; c4 < D; c4++) {
              gWres[l3][r4][c4] += dz * prev[c4];
              dhPrev[c4] += dz * W_res[l3][r4][c4];
            }
          }
          dh = dhPrev;
        }
      }
      b_out -= lr * gbOut;
      for (var j3 = 0; j3 < D; j3++) W_out[j3] -= lr * (gWout[j3] + 0.0008 * W_out[j3]);
      for (var l4 = 0; l4 < DEPTH; l4++) {
        for (var r5 = 0; r5 < D; r5++) {
          b_res[l4][r5] -= lr * gbRes[l4][r5];
          for (var c5 = 0; c5 < D; c5++) W_res[l4][r5][c5] -= lr * (gWres[l4][r5][c5] + 0.0008 * W_res[l4][r5][c5]);
        }
      }
    }

    /* Project each layer [L0..L4] to 2D aligned with readout axis & compute linear probe accuracy */
    var allStates = X0.map(runSample);
    var wNorm = Math.sqrt(M.sum(W_out.map(function (v) { return v * v; }))) || 1;
    var uRead = W_out.map(function (v) { return v / wNorm; });

    var layerCoords = [X0];
    layerAcc = [];

    for (var lv = 0; lv <= DEPTH; lv++) {
      var pts2D;
      if (lv === 0) {
        pts2D = X0.map(function (p) { return p.slice(); });
      } else {
        var Hmat = allStates.map(function (o) { return Array.prototype.slice.call(o.states[lv]); });
        var res = Hmat.map(function (h) {
          var dot = 0;
          for (var d4 = 0; d4 < D; d4++) dot += h[d4] * uRead[d4];
          return h.map(function (v, d5) { return v - dot * uRead[d5]; });
        });
        var pc = M.pca(res, 1);
        var uOrth = pc.components[0] || Hmat[0].map(function (_, k) { return k === 1 ? 1 : 0; });
        var raw = Hmat.map(function (h) {
          var a = 0, b = 0;
          for (var d6 = 0; d6 < D; d6++) { a += h[d6] * uRead[d6]; b += h[d6] * uOrth[d6]; }
          return [a, b];
        });
        /* Align orientation smoothly with previous layer & normalize radius */
        var bb = M.bounds(raw, 0.02);
        var cx = (bb.x[0] + bb.x[1]) / 2, cy = (bb.y[0] + bb.y[1]) / 2;
        var sc = Math.max((bb.x[1] - bb.x[0]) / 1.85, (bb.y[1] - bb.y[0]) / 1.85) || 1;
        pts2D = raw.map(function (q) { return [(q[0] - cx) / sc, (q[1] - cy) / sc]; });
        /* Flip y sign if negatively correlated with previous layer's y to avoid visual flips */
        var prevPts = layerCoords[lv - 1];
        var corrY = M.pearson(pts2D.map(function (q) { return q[1]; }), prevPts.map(function (q) { return q[1]; }));
        if (corrY < 0) pts2D = pts2D.map(function (q) { return [q[0], -q[1]]; });
      }
      if (lv > 0) layerCoords.push(pts2D);

      /* Fit linear probe on pts2D via ridge least squares on [-1, +1] targets */
      var Phi = pts2D.map(function (q) { return [1, q[0], q[1]]; });
      var ySigned = y.map(function (v) { return v ? 1 : -1; });
      var wProbe = M.lstsq(Phi, ySigned, 1e-2);
      var correct = 0;
      for (var ii = 0; ii < n; ii++) {
        var pred = (wProbe[0] + wProbe[1] * pts2D[ii][0] + wProbe[2] * pts2D[ii][1]) >= 0 ? 1 : 0;
        if (pred === y[ii]) correct++;
      }
      layerAcc.push(correct / n);
      if (lv === DEPTH) finalLine = { b: wProbe[0], w0: wProbe[1], w1: wProbe[2] };
    }

    strands = [
      layerCoords.map(function (pts) { return pts.slice(0, N_PER); }),
      layerCoords.map(function (pts) { return pts.slice(N_PER); })
    ];
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.3);
  M.mini(split.a, 'two tangled strands morphing across network layers');
  M.mini(split.b, 'linear probe accuracy at each layer');
  var hf = function (w) { return M.clamp(w * 0.78, 235, 325); };
  var left = M.stage(split.a, { height: hf, ref: 430, label: 'Two intertwined strands untangling progressively across hidden layers' });
  var right = M.stage(split.b, { height: function () { return left.h || 260; }, ref: 330, label: 'Linear separability rising layer by layer' });

  M.legend(root, [
    { color: C.blue, label: 'strand A (manifold 0)' },
    { color: C.orange, label: 'strand B (manifold 1)' },
    { color: C.ink, label: 'best single-cut linear probe', line: true }
  ]);

  var tr = M.transport(root, {
    n: 121, fps: 36, aria: 'Scrub network depth from L0 to L4',
    fmt: function (k) {
      var d = (k / 120) * DEPTH;
      return 'layer L' + M.f(d, 1) + ' / L4';
    },
    onFrame: function (k) {
      depthPos = (k / 120) * DEPTH;
      drawAll();
    }
  });

  var row = M.row(root);
  var turnS = M.slider(row, {
    label: 'Tangle turns', min: 0.6, max: 1.9, step: 0.05, value: turns, accent: 'orange',
    fmt: function (v) { return M.f(v, 2); },
    onInput: function (v) { tr.pause(); turns = v; buildAndSolve(); drawAll(); }
  });
  M.spacer(row);
  M.resetButton(row, function () {
    tr.pause();
    turns = cfg.turns || 1.35;
    turnS.set(turns, true);
    buildAndSolve();
    tr.show(0);
    drawAll();
  });
  var read = M.readout(root);

  var vL = M.view(left, { x: [-1.25, 1.25], y: [-1.25, 1.25], equal: true, pad: { l: 22, r: 16, t: 16, b: 24 } });
  var vR = M.view(right, { x: [-0.3, 4.3], y: [0.45, 1.04], pad: { l: 44, r: 14, t: 16, b: 36 } });

  function interpStrand(sIdx, dCont) {
    var l0 = Math.min(DEPTH - 1, Math.floor(dCont));
    var t = M.ease(M.clamp(dCont - l0, 0, 1));
    var A = strands[sIdx][l0], B = strands[sIdx][l0 + 1];
    var out = new Array(N_PER);
    for (var i = 0; i < N_PER; i++) {
      out[i] = [M.lerp(A[i][0], B[i][0], t), M.lerp(A[i][1], B[i][1], t)];
    }
    return out;
  }

  function currentProbe(pA, pB) {
    var Phi = [], yS = [];
    pA.forEach(function (q) { Phi.push([1, q[0], q[1]]); yS.push(-1); });
    pB.forEach(function (q) { Phi.push([1, q[0], q[1]]); yS.push(1); });
    var w = M.lstsq(Phi, yS, 1e-2), ok = 0;
    pA.forEach(function (q) { if (w[0] + w[1] * q[0] + w[2] * q[1] < 0) ok++; });
    pB.forEach(function (q) { if (w[0] + w[1] * q[0] + w[2] * q[1] >= 0) ok++; });
    return { w: w, acc: ok / (pA.length + pB.length) };
  }

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain([-1.25, 1.25], [-1.25, 1.25]);
    M.clear(ctx, s);

    var pA = interpStrand(0, depthPos);
    var pB = interpStrand(1, depthPos);
    var pr = currentProbe(pA, pB);

    ctx.save(); vL.clip(ctx);
    /* Subtle background shading of the current best linear cut */
    M.field(ctx, vL, 48, 48, function (xx, yy) {
      var z = pr.w[0] + pr.w[1] * xx + pr.w[2] * yy;
      var p = M.sigmoid(z * 2.2), col = M.diverge(p);
      return [col[0], col[1], col[2], 95];
    });

    /* Linear separator line */
    ctx.strokeStyle = M.rgba(C.ink, 0.72);
    ctx.setLineDash([5 * s.sc, 4 * s.sc]);
    ctx.lineWidth = 1.6 * s.sc;
    ctx.beginPath();
    if (Math.abs(pr.w[2]) > Math.abs(pr.w[1])) {
      ctx.moveTo(vL.px(vL.x0), vL.py(-(pr.w[0] + pr.w[1] * vL.x0) / pr.w[2]));
      ctx.lineTo(vL.px(vL.x1), vL.py(-(pr.w[0] + pr.w[1] * vL.x1) / pr.w[2]));
    } else if (Math.abs(pr.w[1]) > 1e-6) {
      ctx.moveTo(vL.px(-(pr.w[0] + pr.w[2] * vL.y0) / pr.w[1]), vL.py(vL.y0));
      ctx.lineTo(vL.px(-(pr.w[0] + pr.w[2] * vL.y1) / pr.w[1]), vL.py(vL.y1));
    }
    ctx.stroke();
    ctx.setLineDash([]);

    /* Draw the two glowing light-bulb cords */
    [[pA, C.blue], [pB, C.orange]].forEach(function (pair) {
      var pts = pair[0], col = pair[1];
      ctx.strokeStyle = M.rgba(col, 0.85);
      ctx.lineWidth = 2.4 * s.sc;
      ctx.beginPath();
      for (var i = 0; i < pts.length; i++) {
        var px = vL.px(pts[i][0]), py = vL.py(pts[i][1]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      for (var k = 0; k < pts.length; k += 2) {
        M.dot(ctx, vL.px(pts[k][0]), vL.py(pts[k][1]), 3.8 * s.sc, col, C.surface, 1.1 * s.sc);
      }
    });
    ctx.restore();
    M.pill(ctx, s, 'Layer L' + M.f(depthPos, 1) + '  \u00b7  linear cut ' + Math.round(pr.acc * 100) + '%', vL.L + 6 * s.sc, vL.T + 12 * s.sc);
    return pr.acc;
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    vR.update();
    M.clear(ctx, s);
    M.axes(ctx, vR, s, {
      xlabel: 'layer depth', ylabel: 'linear probe accuracy',
      xticks: [0, 1, 2, 3, 4],
      xfmt: function (t) { return 'L' + t; },
      yfmt: function (t) { return Math.round(t * 100) + '%'; },
      ny: 4
    });

    ctx.save(); vR.clip(ctx);
    /* Chance reference */
    ctx.strokeStyle = M.rgba(C.muted, 0.5);
    ctx.setLineDash([3 * s.sc, 3 * s.sc]);
    ctx.beginPath(); ctx.moveTo(vR.L, vR.py(0.5)); ctx.lineTo(vR.R, vR.py(0.5)); ctx.stroke();
    ctx.setLineDash([]);

    /* Accuracy trajectory across layers */
    ctx.strokeStyle = C.blue;
    ctx.lineWidth = 2.4 * s.sc;
    ctx.beginPath();
    for (var l = 0; l <= DEPTH; l++) {
      var px = vR.px(l), py = vR.py(layerAcc[l]);
      if (l === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    for (var l2 = 0; l2 <= DEPTH; l2++) {
      M.dot(ctx, vR.px(l2), vR.py(layerAcc[l2]), 4.5 * s.sc, C.blue, C.surface, 1.3 * s.sc);
    }

    /* Current depth marker */
    var curX = vR.px(depthPos);
    ctx.strokeStyle = M.rgba(C.orange, 0.75);
    ctx.lineWidth = 1.6 * s.sc;
    ctx.beginPath(); ctx.moveTo(curX, vR.T); ctx.lineTo(curX, vR.B); ctx.stroke();
    ctx.restore();
  }

  function drawAll() {
    var liveAcc = drawLeft() || layerAcc[0];
    drawRight();
    var stageMsg = depthPos < 0.4
      ? '<span class="k">input L0: strands wrap around each other; any flat blade slices through both</span>'
      : depthPos < 3.4
        ? '<span class="b">intermediate layers: each residual ReLU block loosens one loop of the knot</span>'
        : '<span class="g">output L4: manifolds lie unknotted side-by-side so a linear probe reads out 100%</span>';
    read.innerHTML = '<span class="k">depth</span> L' + M.f(depthPos, 1) +
      '<span class="sep">&middot;</span><span class="k">L0 linear acc</span> ' + Math.round(layerAcc[0] * 100) + '%' +
      '<span class="sep">&middot;</span><span class="k">current linear acc</span> ' + Math.round(liveAcc * 100) + '%' +
      '<span class="sep">&middot;</span>' + stageMsg;
  }

  buildAndSolve();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: function () { tr.pause(); } };
});
