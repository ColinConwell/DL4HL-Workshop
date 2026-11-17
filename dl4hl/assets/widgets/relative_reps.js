/* ══════════════════════════════════════════════════════════════════
   Figure 8 — Relative representations: anchor-query subspace & linear
   classifier coefficients
   Modeled directly on Colin Conwell's Relative-Reps-Classifier diagram:
   images and natural-language queries are embedded on a shared unit
   sphere (e.g., via CLIP). Projecting each image onto its cosine
   similarities with K anchor queries yields an interpretable subspace
   where a linear classifier's weights read out directly over queries.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('relative_reps', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 8,
    title: 'Relative representations: linear classification over natural-language anchor queries',
    hint: 'Toggle anchor queries in or out of the subspace &middot; drag the 3D sphere to inspect image and query geometry'
  });

  var Q_COLORS = ['#9333ea', '#2563eb', '#0f766e', '#d97706', '#b83a2a'];

  function makeFallback() {
    var rng = M.rng(cfg.seed || 29);
    var queries = ['spiky / jagged', 'smooth / rounded', 'angular starburst', 'curvy cloud', 'symmetrical'];
    var qXYZ = [
      M.normVec([0.78, 0.44, 0.44]),
      M.normVec([-0.76, -0.38, 0.52]),
      M.normVec([0.64, 0.68, -0.35]),
      M.normVec([-0.68, -0.58, -0.45]),
      M.normVec([0.08, 0.88, 0.46])
    ];
    var imgXYZ = [], sim = [], y = [];
    for (var i = 0; i < 70; i++) {
      var cls = i < 35 ? 0 : 1; // 0 = bouba, 1 = kiki
      var base = cls === 1 ? [0.68, 0.42, 0.12] : [-0.68, -0.38, 0.14];
      var v = M.normVec([base[0] + rng.normal(0, 0.34), base[1] + rng.normal(0, 0.34), base[2] + rng.normal(0, 0.42)]);
      imgXYZ.push(v);
      y.push(cls);
      sim.push(qXYZ.map(function (q) { return M.cosine(v, q); }));
    }
    return { queries: queries, query_xyz: qXYZ, img_xyz: imgXYZ, sim: sim, y: y, class_names: ['bouba', 'kiki'] };
  }

  var D = (cfg.data && cfg.data.sim && cfg.data.queries) ? cfg.data : makeFallback();
  var K = D.queries.length;
  var activeMask = D.queries.map(function () { return true; });
  var regLam = cfg.penalty || 0.8;
  var coefs = new Array(K).fill(0);
  var accNow = 0;
  var yaw = -0.38, pitch = 0.28, dragOrigin = null;

  function fitQueryClassifier() {
    var actIdx = [];
    for (var j = 0; j < K; j++) if (activeMask[j]) actIdx.push(j);
    coefs = new Array(K).fill(0);
    if (!actIdx.length) { accNow = 0.5; return; }

    var n = D.sim.length;
    /* Standardize active columns */
    var means = actIdx.map(function (j) {
      var s = 0; for (var i = 0; i < n; i++) s += D.sim[i][j]; return s / n;
    });
    var stds = actIdx.map(function (j, col) {
      var s = 0; for (var i = 0; i < n; i++) { var d = D.sim[i][j] - means[col]; s += d * d; }
      return Math.sqrt(s / Math.max(1, n - 1)) || 1;
    });
    var Phi = [], ySigned = [];
    for (var r = 0; r < n; r++) {
      var row = [1];
      for (var c = 0; c < actIdx.length; c++) row.push((D.sim[r][actIdx[c]] - means[c]) / stds[c]);
      Phi.push(row);
      ySigned.push(D.y[r] ? 1 : -1);
    }
    var w = M.lstsq(Phi, ySigned, regLam);
    for (var c2 = 0; c2 < actIdx.length; c2++) coefs[actIdx[c2]] = w[c2 + 1];

    var ok = 0;
    for (var r2 = 0; r2 < n; r2++) {
      var sc = w[0];
      for (var c3 = 0; c3 < actIdx.length; c3++) sc += w[c3 + 1] * Phi[r2][c3 + 1];
      if ((sc >= 0 ? 1 : 0) === D.y[r2]) ok++;
    }
    accNow = ok / n;
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.05);
  M.mini(split.a, 'image-language sphere + "anchor point" query axes');
  M.mini(split.b, 'linear classifier coefficients over queries');
  var hf = function (w) { return M.clamp(w * 0.78, 235, 320); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: '3D unit sphere with embedded stimuli and natural-language anchor query arrows' });
  var right = M.stage(split.b, { height: function () { return left.h || 255; }, ref: 400, label: 'Signed bar chart of linear classifier weights across natural-language queries' });

  var clsNames = D.class_names || ['class 0', 'class 1'];
  M.legend(root, [
    { color: C.blue, label: clsNames[0] + ' (-1)' },
    { color: C.orange, label: clsNames[1] + ' (+1)' },
    { color: C.purple, label: 'language anchor queries', line: true }
  ]);

  var row1 = M.row(root);
  M.label(row1, 'Active queries');
  var qBtns = D.queries.map(function (q, j) {
    var b = M.button(row1, 'Q' + (j + 1) + ': ' + q.split(' ')[0], function () {
      var countOn = activeMask.filter(Boolean).length;
      if (activeMask[j] && countOn <= 1) return; // keep at least 1 query active
      activeMask[j] = !activeMask[j];
      b.classList.toggle('on', activeMask[j]);
      fitQueryClassifier();
      drawAll();
    }, { cls: 'on', aria: 'Toggle anchor query ' + q });
    return b;
  });

  var row2 = M.row(root);
  var lamS = M.slider(row2, {
    label: 'Ridge penalty \u03bb', min: 0.1, max: 12, step: 0.2, value: regLam, accent: 'blue',
    fmt: function (v) { return M.f(v, 1); },
    onInput: function (v) { regLam = v; fitQueryClassifier(); drawAll(); }
  });
  M.spacer(row2);
  M.resetButton(row2, function () {
    for (var j = 0; j < K; j++) { activeMask[j] = true; qBtns[j].classList.add('on'); }
    regLam = cfg.penalty || 0.8;
    lamS.set(regLam, true);
    yaw = -0.38; pitch = 0.28;
    fitQueryClassifier(); drawAll();
  });
  var read = M.readout(root);

  M.pointer(left.cv, {
    down: function (p) { dragOrigin = { x: p.x, y: p.y, yaw: yaw, pitch: pitch }; return true; },
    drag: function (p) {
      if (!dragOrigin) return;
      yaw = dragOrigin.yaw + (p.x - dragOrigin.x) * 0.012;
      pitch = M.clamp(dragOrigin.pitch + (p.y - dragOrigin.y) * 0.012, -1.1, 1.1);
      drawAll();
    },
    up: function () { dragOrigin = null; }
  });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    M.clear(ctx, s);
    var R = Math.min(s.w, s.h) * 0.39;
    var cx = s.w * 0.5, cy = s.h * 0.52;
    M.drawSphere(ctx, cx, cy, R, yaw, pitch, s);

    /* Embedded stimuli dots */
    for (var i = 0; i < D.img_xyz.length; i++) {
      var p = M.sphereProject(D.img_xyz[i], cx, cy, R, yaw, pitch);
      var alpha = p.z >= 0 ? 0.88 : 0.30;
      M.dot(ctx, p.x, p.y, (p.z >= 0 ? 3.3 : 2.5) * s.sc, M.rgba(D.y[i] ? C.orange : C.blue, alpha));
    }

    /* Anchor query vectors radiating from sphere center */
    for (var j = 0; j < K; j++) {
      var qp = M.sphereProject(D.query_xyz[j], cx, cy, R * 1.08, yaw, pitch);
      var qCol = Q_COLORS[j % Q_COLORS.length];
      var on = activeMask[j];
      M.arrow(ctx, cx, cy, qp.x, qp.y, M.rgba(qCol, on ? 0.95 : 0.22), (on ? 2.3 : 1.1) * s.sc, 7 * s.sc);
      if (on) {
        M.pill(ctx, s, 'Q' + (j + 1), qp.x, qp.y - 9 * s.sc, { align: 'center', color: qCol, size: 8.2 });
      }
    }
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);

    var maxAbs = Math.max(0.5, Math.max.apply(null, coefs.map(Math.abs)) * 1.18);
    var labelW = 108 * s.sc;
    var xLeft = labelW + 10 * s.sc, xRight = s.w - 16 * s.sc;
    var xZero = (xLeft + xRight) / 2, halfSpan = (xRight - xLeft) / 2;
    var topY = 24 * s.sc, botY = s.h - 34 * s.sc;
    var rowH = (botY - topY) / K;

    /* Zero axis */
    ctx.strokeStyle = C.ink;
    ctx.setLineDash([3 * s.sc, 3 * s.sc]);
    ctx.lineWidth = 1.1 * s.sc;
    ctx.beginPath(); ctx.moveTo(xZero, topY - 4 * s.sc); ctx.lineTo(xZero, botY + 4 * s.sc); ctx.stroke();
    ctx.setLineDash([]);

    for (var j = 0; j < K; j++) {
      var cy = topY + (j + 0.5) * rowH;
      var w = coefs[j];
      var on = activeMask[j];
      var qCol = Q_COLORS[j % Q_COLORS.length];

      ctx.fillStyle = on ? C.ink : C.muted;
      ctx.font = M.font(s, 8.8, on ? 600 : 400, 'sans');
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('"' + D.queries[j] + '"', labelW, cy);

      if (on) {
        var bw = (w / maxAbs) * halfSpan;
        ctx.fillStyle = M.rgba(w >= 0 ? C.orange : C.blue, 0.85);
        var bx = bw >= 0 ? xZero : xZero + bw;
        ctx.fillRect(bx, cy - rowH * 0.28, Math.abs(bw), rowH * 0.56);
        ctx.strokeStyle = qCol; ctx.lineWidth = 1.3 * s.sc;
        ctx.strokeRect(bx, cy - rowH * 0.28, Math.abs(bw), rowH * 0.56);

        ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.2);
        ctx.textAlign = bw >= 0 ? 'left' : 'right';
        ctx.fillText((w >= 0 ? '+' : '') + M.f(w, 2), xZero + bw + (bw >= 0 ? 4 : -4) * s.sc, cy);
      } else {
        ctx.fillStyle = C.muted; ctx.font = M.font(s, 8);
        ctx.textAlign = 'center';
        ctx.fillText('(excluded)', xZero, cy);
      }
    }

    /* Bottom polarity axis label */
    ctx.fillStyle = C.muted; ctx.font = M.font(s, 8.5, 600, 'sans');
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left'; ctx.fillText('\u2190 \u22121 (' + clsNames[0] + ')', xLeft, botY + 8 * s.sc);
    ctx.textAlign = 'center'; ctx.fillText('0', xZero, botY + 8 * s.sc);
    ctx.textAlign = 'right'; ctx.fillText('+1 (' + clsNames[1] + ') \u2192', xRight, botY + 8 * s.sc);
  }

  function drawAll() {
    drawLeft(); drawRight();
    var nOn = activeMask.filter(Boolean).length;
    var bestJ = M.argmax(coefs.map(Math.abs));
    read.innerHTML = '<span class="k">query subspace</span> ' + nOn + ' / ' + K + ' anchors' +
      '<span class="sep">&middot;</span><span class="k">classifier accuracy</span> ' + Math.round(accNow * 100) + '%' +
      '<span class="sep">&middot;</span><span class="k">strongest cue</span> "' + M.esc(D.queries[bestJ]) + '" (' + (coefs[bestJ] >= 0 ? '+' : '') + M.f(coefs[bestJ], 2) + ')' +
      '<span class="sep">&middot;</span><span class="g">projecting onto language anchors turns opaque embeddings into readable weights</span>';
  }

  fitQueryClassifier();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return {};
});
