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
    fig: 2,
    title: 'Deep learning as disentanglement: unknotting tangled curves layer by layer',
    hint: 'Press play or scrub Depth to travel from input layer L0 to L4 &middot; increase Tangle turns to tighten the spiral'
  });

  var seed = cfg.seed || 11;
  var N_PER = 38;
  var DEPTH = 4;
  var turns = cfg.turns || 1.35;
  var depthPos = 0; // continuous 0..4
  var ready = false;
  var strands = [[], []];
  var layerAcc = [0.6, 0.76, 0.9, 0.97, 1.0];
  var solveCache = {};

  /* Build two intertwined spiral strands and train a 4-block residual ReLU
     network with Adam + progressive auxiliary heads so every layer genuinely
     loosens the tangle and L4 achieves clean visual & linear separation. */
  function buildAndSolve() {
    var cacheKey = turns.toFixed(2);
    if (solveCache[cacheKey]) {
      strands = solveCache[cacheKey].strands;
      layerAcc = solveCache[cacheKey].layerAcc;
      return;
    }

    var rng = M.rng(seed + Math.round(turns * 100));
    var X0 = [], y = [];
    for (var sIdx = 0; sIdx < 2; sIdx++) {
      for (var i = 0; i < N_PER; i++) {
        var u = i / (N_PER - 1);
        var rad = 0.16 + 0.88 * u;
        var ang = u * turns * Math.PI * 2 + (sIdx === 1 ? Math.PI : 0);
        X0.push([
          rad * Math.cos(ang) + rng.normal(0, 0.015),
          rad * Math.sin(ang) + rng.normal(0, 0.015)
        ]);
        y.push(sIdx);
      }
    }

    var D = 14, n = X0.length;
    /* Linear lift 2D -> D (preserves exact input linear non-separability at L0) */
    var W_in = [];
    for (var d = 0; d < D; d++) {
      var a = (Math.PI * d) / D;
      W_in.push([1.15 * Math.cos(a), 1.15 * Math.sin(a)]);
    }

    var params = [];
    function alloc(sz, scale) {
      var arr = new Float64Array(sz);
      for (var k = 0; k < sz; k++) arr[k] = scale ? rng.normal(0, scale) : 0;
      params.push(arr);
      return arr;
    }

    var W_res1 = [], b_res1 = [], W_res2 = [], b_res2 = [], W_head = [], b_head = [];
    for (var l = 0; l < DEPTH; l++) {
      W_res1.push(alloc(D * D, 0.55));
      b_res1.push(alloc(D, 0.22));
      W_res2.push(alloc(D * D, 0.45));
      b_res2.push(alloc(D, 0.05));
      W_head.push(alloc(D, 0.4));
      b_head.push(alloc(1, 0));
    }

    var mAdam = params.map(function (p) { return new Float64Array(p.length); });
    var vAdam = params.map(function (p) { return new Float64Array(p.length); });
    var stepScales = [0.24, 0.34, 0.48, 0.68];
    var lossWeights = [0.05, 0.12, 0.28, 1.6];

    for (var it = 1; it <= 210; it++) {
      var grads = params.map(function (p) { return new Float64Array(p.length); });

      for (var idx = 0; idx < n; idx++) {
        var x0 = X0[idx][0], x1 = X0[idx][1], yi = y[idx];
        var h0 = new Float64Array(D);
        for (var d0 = 0; d0 < D; d0++) h0[d0] = W_in[d0][0] * x0 + W_in[d0][1] * x1;

        var hs = [h0], z1s = [], a1s = [];
        var cur = h0;
        for (var lf = 0; lf < DEPTH; lf++) {
          var z1 = new Float64Array(D), a1 = new Float64Array(D), nxt = new Float64Array(D);
          var w1 = W_res1[lf], b1 = b_res1[lf], w2 = W_res2[lf], b2 = b_res2[lf], sc = stepScales[lf];
          for (var r = 0; r < D; r++) {
            var s1 = b1[r], off = r * D;
            for (var c = 0; c < D; c++) s1 += w1[off + c] * cur[c];
            z1[r] = s1;
            a1[r] = s1 > 0 ? s1 : 0.05 * s1;
          }
          for (var r2 = 0; r2 < D; r2++) {
            var s2 = b2[r2], off2 = r2 * D;
            for (var c2 = 0; c2 < D; c2++) s2 += w2[off2 + c2] * a1[c2];
            nxt[r2] = cur[r2] + sc * s2;
          }
          z1s.push(z1); a1s.push(a1); hs.push(nxt); cur = nxt;
        }

        var dh = new Float64Array(D);
        for (var lb = DEPTH - 1; lb >= 0; lb--) {
          var baseIdx = lb * 6;
          var gW1 = grads[baseIdx], gB1 = grads[baseIdx + 1];
          var gW2 = grads[baseIdx + 2], gB2 = grads[baseIdx + 3];
          var gWh = grads[baseIdx + 4], gBh = grads[baseIdx + 5];
          var wh = W_head[lb], bh = b_head[lb][0], hl = hs[lb + 1];

          /* Margin-encouraging logistic loss so L4 opens a wide visual gap */
          var logit = bh;
          for (var d2 = 0; d2 < D; d2++) logit += wh[d2] * hl[d2];
          var prob = M.sigmoid(logit);
          var dLogit = lossWeights[lb] * (prob - yi) / n;
          gBh[0] += dLogit;
          for (var d3 = 0; d3 < D; d3++) {
            gWh[d3] += dLogit * hl[d3];
            dh[d3] += dLogit * wh[d3];
          }

          var prev = hs[lb], z1b = z1s[lb], a1b = a1s[lb];
          var w1b = W_res1[lb], w2b = W_res2[lb], scb = stepScales[lb];
          var dhPrev = new Float64Array(dh);
          var da1 = new Float64Array(D);

          for (var r3 = 0; r3 < D; r3++) {
            var ds2 = dh[r3] * scb;
            gB2[r3] += ds2;
            var off3 = r3 * D;
            for (var c3 = 0; c3 < D; c3++) {
              gW2[off3 + c3] += ds2 * a1b[c3];
              da1[c3] += ds2 * w2b[off3 + c3];
            }
          }
          for (var r4 = 0; r4 < D; r4++) {
            var dz1 = da1[r4] * (z1b[r4] > 0 ? 1 : 0.05);
            gB1[r4] += dz1;
            var off4 = r4 * D;
            for (var c4 = 0; c4 < D; c4++) {
              gW1[off4 + c4] += dz1 * prev[c4];
              dhPrev[c4] += dz1 * w1b[off4 + c4];
            }
          }
          dh = dhPrev;
        }
      }

      var lr = 0.026, beta1 = 0.9, beta2 = 0.999;
      var c1 = 1 - Math.pow(beta1, it), c2 = 1 - Math.pow(beta2, it);
      for (var pIdx = 0; pIdx < params.length; pIdx++) {
        var pa = params[pIdx], gr = grads[pIdx], ma = mAdam[pIdx], va = vAdam[pIdx];
        for (var k2 = 0; k2 < pa.length; k2++) {
          var gVal = gr[k2] + 2e-4 * pa[k2];
          ma[k2] = beta1 * ma[k2] + (1 - beta1) * gVal;
          va[k2] = beta2 * va[k2] + (1 - beta2) * gVal * gVal;
          pa[k2] -= lr * (ma[k2] / c1) / (Math.sqrt(va[k2] / c2) + 1e-8);
        }
      }
    }

    /* Extract hidden states across L0..L4 and project each layer to 2D
       with horizontal axis aligned to that layer's linear readout direction */
    var statesByLayer = [[], [], [], [], []];
    for (var ii = 0; ii < n; ii++) {
      statesByLayer[0].push(X0[ii].slice());
      var curH = new Float64Array(D);
      for (var dd = 0; dd < D; dd++) curH[dd] = W_in[dd][0] * X0[ii][0] + W_in[dd][1] * X0[ii][1];
      for (var ll = 0; ll < DEPTH; ll++) {
        var aTmp = new Float64Array(D), nxtH = new Float64Array(D);
        for (var rr = 0; rr < D; rr++) {
          var ss = b_res1[ll][rr];
          for (var cc = 0; cc < D; cc++) ss += W_res1[ll][rr * D + cc] * curH[cc];
          aTmp[rr] = ss > 0 ? ss : 0.05 * ss;
        }
        for (var rr2 = 0; rr2 < D; rr2++) {
          var ss2 = b_res2[ll][rr2];
          for (var cc2 = 0; cc2 < D; cc2++) ss2 += W_res2[ll][rr2 * D + cc2] * aTmp[cc2];
          nxtH[rr2] = curH[rr2] + stepScales[ll] * ss2;
        }
        curH = nxtH;
        statesByLayer[ll + 1].push(Array.prototype.slice.call(curH));
      }
    }

    var layerCoords = [];
    layerAcc = [];
    var ySigned = y.map(function (v) { return v ? 1 : -1; });

    for (var lv = 0; lv <= DEPTH; lv++) {
      var pts2D;
      if (lv === 0) {
        pts2D = statesByLayer[0].map(function (p) { return p.slice(); });
      } else {
        var Hmat = statesByLayer[lv];
        /* Fit linear readout direction on Hmat so x-axis is always the optimal separator (blue left, orange right) */
        var PhiD = Hmat.map(function (h) { return [1].concat(h); });
        var wFull = M.lstsq(PhiD, ySigned, 0.08).slice(1);
        var wNorm = Math.sqrt(M.sum(wFull.map(function (v) { return v * v; }))) || 1;
        var uRead = wFull.map(function (v) { return v / wNorm; });

        var res = Hmat.map(function (h) {
          var dot = 0;
          for (var d4 = 0; d4 < D; d4++) dot += h[d4] * uRead[d4];
          return h.map(function (v, d5) { return v - dot * uRead[d5]; });
        });
        var pc = M.pca(res, 1);
        var uOrth = pc.components[0] || Hmat[0].map(function (_, k3) { return k3 === 1 ? 1 : 0; });
        var raw = Hmat.map(function (h) {
          var ax = 0, ay = 0;
          for (var d6 = 0; d6 < D; d6++) { ax += h[d6] * uRead[d6]; ay += h[d6] * uOrth[d6]; }
          return [ax, ay];
        });

        var bb = M.bounds(raw, 0.02);
        var cx = (bb.x[0] + bb.x[1]) / 2, cy = (bb.y[0] + bb.y[1]) / 2;
        var sx = (bb.x[1] - bb.x[0]) / 1.82 || 1;
        var sy = (bb.y[1] - bb.y[0]) / 1.75 || 1;
        pts2D = raw.map(function (q) { return [(q[0] - cx) / sx, (q[1] - cy) / sy]; });

        /* Preserve smooth vertical continuity from previous layer */
        var prevPts = layerCoords[lv - 1];
        var corrY = M.pearson(pts2D.map(function (q) { return q[1]; }), prevPts.map(function (q) { return q[1]; }));
        if (corrY < 0) pts2D = pts2D.map(function (q) { return [q[0], -q[1]]; });
      }
      layerCoords.push(pts2D);

      var Phi2 = pts2D.map(function (q) { return [1, q[0], q[1]]; });
      var wProbe = M.lstsq(Phi2, ySigned, 1e-3);
      var correct = 0;
      for (var m = 0; m < n; m++) {
        var pred = (wProbe[0] + wProbe[1] * pts2D[m][0] + wProbe[2] * pts2D[m][1]) >= 0 ? 1 : 0;
        if (pred === y[m]) correct++;
      }
      layerAcc.push(correct / n);
    }

    strands = [
      layerCoords.map(function (pts) { return pts.slice(0, N_PER); }),
      layerCoords.map(function (pts) { return pts.slice(N_PER); })
    ];
    solveCache[cacheKey] = { strands: strands, layerAcc: layerAcc.slice() };
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.3);
  M.mini(split.a, 'two tangled strands morphing across network layers');
  M.mini(split.b, 'linear probe accuracy at each layer');
  var hf = function (w) { return M.clamp(w * 0.78, 235, 325); };
  var left = M.stage(split.a, { height: hf, ref: 430, label: 'Two intertwined strands untangling progressively across hidden layers' });
  var right = M.stage(split.b, { height: function () { return left.h || 260; }, ref: 330, label: 'Linear separability rising layer by layer' });

  M.legend(root, [
    { color: C.blue, label: 'strand A (category 0)' },
    { color: C.orange, label: 'strand B (category 1)' },
    { color: C.ink, label: 'best single straight cut', line: true }
  ]);

  var tr = M.transport(root, {
    n: 121, fps: 34, aria: 'Scrub network depth from L0 to L4',
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
    onChange: function (v) { tr.pause(); turns = v; buildAndSolve(); drawAll(); },
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
    var w = M.lstsq(Phi, yS, 1e-3), ok = 0;
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
    M.field(ctx, vL, 48, 48, function (xx, yy) {
      var z = pr.w[0] + pr.w[1] * xx + pr.w[2] * yy;
      var p = M.sigmoid(z * 2.2), col = M.diverge(p);
      return [col[0], col[1], col[2], 95];
    });

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

    [[pA, C.blue], [pB, C.orange]].forEach(function (pair) {
      var pts = pair[0], col = pair[1];
      ctx.strokeStyle = M.rgba(col, 0.88);
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
    ctx.strokeStyle = M.rgba(C.muted, 0.5);
    ctx.setLineDash([3 * s.sc, 3 * s.sc]);
    ctx.beginPath(); ctx.moveTo(vR.L, vR.py(0.5)); ctx.lineTo(vR.R, vR.py(0.5)); ctx.stroke();
    ctx.setLineDash([]);

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

    var curX = vR.px(depthPos);
    ctx.strokeStyle = M.rgba(C.orange, 0.75);
    ctx.lineWidth = 1.6 * s.sc;
    ctx.beginPath(); ctx.moveTo(curX, vR.T); ctx.lineTo(curX, vR.B); ctx.stroke();
    ctx.restore();
  }

  function drawAll() {
    if (!ready) return;
    var liveAcc = drawLeft() || layerAcc[0];
    drawRight();
    var stageMsg = depthPos < 0.4
      ? '<span class="k">Input L0: strands wrap around each other; any flat blade slices through both.</span>'
      : depthPos < 3.4
        ? '<span class="b">Intermediate layers: each residual ReLU block loosens another loop of the knot.</span>'
        : '<span class="g">Output L4: the two strands pull completely apart so a single linear boundary separates them cleanly (' + Math.round(layerAcc[DEPTH] * 100) + '%).</span>';
    read.innerHTML = '<span class="k">depth</span> L' + M.f(depthPos, 1) +
      '<span class="sep">&middot;</span><span class="k">L0 linear acc</span> ' + Math.round(layerAcc[0] * 100) + '%' +
      '<span class="sep">&middot;</span><span class="k">current linear acc</span> ' + Math.round(liveAcc * 100) + '%' +
      '<div class="wread-interp">' + stageMsg + '</div>';
  }

  buildAndSolve();
  left.draw = drawAll; right.draw = drawAll;
  ready = true;
  tr.show(0);
  left.request(); right.request();
  return { destroy: function () { tr.pause(); } };
});
