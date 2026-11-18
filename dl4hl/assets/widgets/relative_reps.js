/* ══════════════════════════════════════════════════════════════════
   Figure 8 — Relative representations: anchor-query subspace & linear
   classifier coefficients
   Modeled directly on Colin Conwell's Relative-Reps-Classifier diagram:
   images and natural-language queries are aligned in a shared semantic
   space (with per-modality baseline offsets removed). Active anchor
   queries span a geodesic patch on the sphere and induce a readable
   great-circle decision plane over natural-language descriptors.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('relative_reps', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 8,
    title: 'Relative representations: linear classification over natural-language anchor queries',
    hint: 'Toggle anchor queries to grow or shrink the spanned region on the sphere &middot; watch the great-circle decision boundary &amp; accuracy update &middot; drag the 3D sphere'
  });

  var Q_COLORS = ['#9333ea', '#2563eb', '#0f766e', '#d97706', '#b83a2a'];

  function makeFallback() {
    var rng = M.rng(cfg.seed || 29);
    var queries = ['spiky / jagged', 'smooth / rounded', 'angular starburst', 'curvy cloud', 'symmetrical'];
    var qBase = [
      M.normVec([0.84, 0.34, 0.42]),
      M.normVec([-0.82, -0.28, 0.50]),
      M.normVec([0.38, 0.84, -0.38]),
      M.normVec([-0.42, -0.78, 0.46]),
      M.normVec([0.08, 0.92, 0.38])
    ];
    var sim = [], y = [];
    for (var i = 0; i < 80; i++) {
      var cls = i < 40 ? 0 : 1; // 0 = bouba, 1 = kiki
      var center = cls === 1 ? [0.72, 0.22, 0.25] : [-0.72, -0.20, 0.25];
      var v = M.normVec([center[0] + rng.normal(0, 0.38), center[1] + rng.normal(0, 0.45), center[2] + rng.normal(0, 0.45)]);
      y.push(cls);
      sim.push(qBase.map(function (q) { return M.cosine(v, q); }));
    }
    return { queries: queries, sim: sim, y: y, class_names: ['bouba', 'kiki'] };
  }

  var D = (cfg.data && cfg.data.sim && cfg.data.queries) ? cfg.data : makeFallback();
  var K = D.queries.length;
  var activeMask = D.queries.map(function () { return true; });
  var regLam = cfg.penalty || 0.8;
  var coefs = new Array(K).fill(0);
  var accNow = 0;
  var yaw = -0.25, pitch = 0.24, dragOrigin = null;

  /* Compute modality-gap-free joint 3D spherical coordinates directly from
     column-standardized D.sim so queries spread naturally across the sphere
     around the image clouds they describe! */
  var imgXYZ = [], queryXYZ = [], Zstd = [];
  (function buildJointSphere() {
    var n = D.sim.length;
    var means = new Array(K).fill(0), stds = new Array(K).fill(1);
    for (var j = 0; j < K; j++) {
      var s = 0;
      for (var i = 0; i < n; i++) s += D.sim[i][j];
      means[j] = s / n;
      var v = 0;
      for (var i2 = 0; i2 < n; i2++) { var d = D.sim[i2][j] - means[j]; v += d * d; }
      stds[j] = Math.sqrt(v / Math.max(1, n - 1)) || 1;
    }
    Zstd = D.sim.map(function (row) {
      return row.map(function (val, j) { return (val - means[j]) / stds[j]; });
    });

    /* Axis 1: class-discriminative direction in standardized query space;
       Axes 2 & 3: top 2 orthogonal PCA directions of query variation. */
    var m0 = new Array(K).fill(0), m1 = new Array(K).fill(0), n0 = 0, n1 = 0;
    for (var r = 0; r < n; r++) {
      if (D.y[r]) { n1++; for (var c = 0; c < K; c++) m1[c] += Zstd[r][c]; }
      else { n0++; for (var c2 = 0; c2 < K; c2++) m0[c2] += Zstd[r][c2]; }
    }
    var b0 = new Array(K);
    for (var k = 0; k < K; k++) b0[k] = (m1[k] / Math.max(1, n1)) - (m0[k] / Math.max(1, n0));
    b0 = M.normVec(b0);

    var Zorth = Zstd.map(function (row) {
      var dot = 0;
      for (var k2 = 0; k2 < K; k2++) dot += row[k2] * b0[k2];
      return row.map(function (v, k3) { return v - dot * b0[k3]; });
    });
    var pcaRes = M.pca(Zorth, 2);
    var b1 = pcaRes.components[0] || b0.map(function (_, i3) { return i3 === 1 ? 1 : 0; });
    var b2 = pcaRes.components[1] || b0.map(function (_, i4) { return i4 === 2 ? 1 : 0; });

    imgXYZ = Zstd.map(function (row) {
      var x = 0, y = 0, z = 0;
      for (var k4 = 0; k4 < K; k4++) {
        x += row[k4] * b0[k4];
        y += row[k4] * b1[k4];
        z += row[k4] * b2[k4];
      }
      return M.normVec([x, y, z + 0.18]);
    });

    /* Project each query j via its empirical correlation profile across Zstd */
    queryXYZ = [];
    for (var qj = 0; qj < K; qj++) {
      var corrVec = new Array(K).fill(0);
      for (var m = 0; m < K; m++) {
        var sc = 0;
        for (var rr = 0; rr < n; rr++) sc += Zstd[rr][qj] * Zstd[rr][m];
        corrVec[m] = sc / n;
      }
      var qx = 0, qy = 0, qz = 0;
      for (var m2 = 0; m2 < K; m2++) {
        qx += corrVec[m2] * b0[m2];
        qy += corrVec[m2] * b1[m2];
        qz += corrVec[m2] * b2[m2];
      }
      queryXYZ.push(M.normVec([qx, qy, qz + 0.22]));
    }
  })();

  function fitQueryClassifier() {
    var actIdx = [];
    for (var j = 0; j < K; j++) if (activeMask[j]) actIdx.push(j);
    coefs = new Array(K).fill(0);
    if (!actIdx.length) { accNow = 0.5; return; }

    var n = Zstd.length;
    var Phi = [], ySigned = [];
    for (var r = 0; r < n; r++) {
      var row = [1];
      for (var c = 0; c < actIdx.length; c++) row.push(Zstd[r][actIdx[c]]);
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

  /* Spherical slerp between two 3D unit vectors */
  function slerp(a, b, t) {
    return M.normVec([M.lerp(a[0], b[0], t), M.lerp(a[1], b[1], t), M.lerp(a[2], b[2], t)]);
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.05);
  M.mini(split.a, 'shared sphere: query hull coverage + great-circle separator');
  M.mini(split.b, 'linear classifier coefficients over queries');
  var hf = function (w) { return M.clamp(w * 0.78, 240, 325); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: '3D unit sphere with embedded stimuli, spherical hull over active language anchors, and great-circle decision boundary' });
  var right = M.stage(split.b, { height: function () { return left.h || 260; }, ref: 400, label: 'Signed bar chart of linear classifier weights across natural-language queries' });

  var clsNames = D.class_names || ['class 0', 'class 1'];
  M.legend(root, [
    { color: C.blue, label: clsNames[0] + ' (\u22121)' },
    { color: C.orange, label: clsNames[1] + ' (+1)' },
    { color: C.purple, label: 'spanned query hull', line: true },
    { color: C.ink, label: 'induced decision equator', line: true }
  ]);

  var row1 = M.row(root);
  M.label(row1, 'Active queries');
  var qBtns = D.queries.map(function (q, j) {
    var b = M.button(row1, 'Q' + (j + 1) + ': ' + q.split(' ')[0], function () {
      var countOn = activeMask.filter(Boolean).length;
      if (activeMask[j] && countOn <= 1) return;
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
    yaw = -0.25; pitch = 0.24;
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

    /* 1. Spherical surface coverage (geodesic hull / arcs) over active anchor queries */
    var actIdx = [];
    for (var j0 = 0; j0 < K; j0++) if (activeMask[j0]) actIdx.push(j0);

    if (actIdx.length >= 3) {
      /* Order active anchors cyclically around their spherical centroid to trace a clean geodesic hull */
      var meanQ = [0, 0, 0];
      actIdx.forEach(function (idx) {
        meanQ[0] += queryXYZ[idx][0]; meanQ[1] += queryXYZ[idx][1]; meanQ[2] += queryXYZ[idx][2];
      });
      meanQ = M.normVec(meanQ);
      var refUp = Math.abs(meanQ[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
      var tU = M.normVec([
        refUp[1] * meanQ[2] - refUp[2] * meanQ[1],
        refUp[2] * meanQ[0] - refUp[0] * meanQ[2],
        refUp[0] * meanQ[1] - refUp[1] * meanQ[0]
      ]);
      var tV = [
        meanQ[1] * tU[2] - meanQ[2] * tU[1],
        meanQ[2] * tU[0] - meanQ[0] * tU[2],
        meanQ[0] * tU[1] - meanQ[1] * tU[0]
      ];
      var sorted = actIdx.slice().sort(function (a, b) {
        var qa = queryXYZ[a], qb = queryXYZ[b];
        var angA = Math.atan2(qa[0] * tV[0] + qa[1] * tV[1] + qa[2] * tV[2], qa[0] * tU[0] + qa[1] * tU[1] + qa[2] * tU[2]);
        var angB = Math.atan2(qb[0] * tV[0] + qb[1] * tV[1] + qb[2] * tV[2], qb[0] * tU[0] + qb[1] * tU[1] + qb[2] * tU[2]);
        return angA - angB;
      });

      /* Fill geodesic fans from meanQ to each edge so the patch wraps smoothly over the sphere */
      ctx.fillStyle = M.rgba(C.purple || C.blue, 0.16);
      for (var e = 0; e < sorted.length; e++) {
        var vA = queryXYZ[sorted[e]], vB = queryXYZ[sorted[(e + 1) % sorted.length]];
        ctx.beginPath();
        var pC = M.sphereProject(meanQ, cx, cy, R, yaw, pitch);
        ctx.moveTo(pC.x, pC.y);
        for (var st = 0; st <= 14; st++) {
          var pe = M.sphereProject(slerp(vA, vB, st / 14), cx, cy, R, yaw, pitch);
          ctx.lineTo(pe.x, pe.y);
        }
        ctx.closePath();
        ctx.fill();
      }
      /* Geodesic hull boundary arcs */
      ctx.strokeStyle = M.rgba(C.purple || C.blue, 0.68);
      ctx.lineWidth = 1.6 * s.sc;
      ctx.beginPath();
      for (var e2 = 0; e2 < sorted.length; e2++) {
        var vA2 = queryXYZ[sorted[e2]], vB2 = queryXYZ[sorted[(e2 + 1) % sorted.length]];
        for (var st2 = 0; st2 <= 16; st2++) {
          var pe2 = M.sphereProject(slerp(vA2, vB2, st2 / 16), cx, cy, R, yaw, pitch);
          if (e2 === 0 && st2 === 0) ctx.moveTo(pe2.x, pe2.y); else ctx.lineTo(pe2.x, pe2.y);
        }
      }
      ctx.closePath();
      ctx.stroke();
    } else if (actIdx.length === 2) {
      ctx.strokeStyle = M.rgba(C.purple || C.blue, 0.78);
      ctx.lineWidth = 2.0 * s.sc;
      ctx.beginPath();
      for (var st3 = 0; st3 <= 20; st3++) {
        var pe3 = M.sphereProject(slerp(queryXYZ[actIdx[0]], queryXYZ[actIdx[1]], st3 / 20), cx, cy, R, yaw, pitch);
        if (st3 === 0) ctx.moveTo(pe3.x, pe3.y); else ctx.lineTo(pe3.x, pe3.y);
      }
      ctx.stroke();
    } else if (actIdx.length === 1) {
      /* Spherical cap ring around single active anchor */
      var qOnly = queryXYZ[actIdx[0]];
      var up1 = Math.abs(qOnly[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
      var eA = M.normVec([up1[1] * qOnly[2] - up1[2] * qOnly[1], up1[2] * qOnly[0] - up1[0] * qOnly[2], up1[0] * qOnly[1] - up1[1] * qOnly[0]]);
      var eB = [qOnly[1] * eA[2] - qOnly[2] * eA[1], qOnly[2] * eA[0] - qOnly[0] * eA[2], qOnly[0] * eA[1] - qOnly[1] * eA[0]];
      ctx.fillStyle = M.rgba(C.purple || C.blue, 0.16);
      ctx.strokeStyle = M.rgba(C.purple || C.blue, 0.70);
      ctx.lineWidth = 1.4 * s.sc;
      ctx.beginPath();
      for (var ca = 0; ca <= 36; ca++) {
        var ang = (ca / 36) * Math.PI * 2, rCap = 0.36;
        var cv = M.normVec([
          Math.cos(rCap) * qOnly[0] + Math.sin(rCap) * (Math.cos(ang) * eA[0] + Math.sin(ang) * eB[0]),
          Math.cos(rCap) * qOnly[1] + Math.sin(rCap) * (Math.cos(ang) * eA[1] + Math.sin(ang) * eB[1]),
          Math.cos(rCap) * qOnly[2] + Math.sin(rCap) * (Math.cos(ang) * eA[2] + Math.sin(ang) * eB[2])
        ]);
        var pc2 = M.sphereProject(cv, cx, cy, R, yaw, pitch);
        if (ca === 0) ctx.moveTo(pc2.x, pc2.y); else ctx.lineTo(pc2.x, pc2.y);
      }
      ctx.fill(); ctx.stroke();
    }

    /* 2. Induced Great-Circle Decision Boundary {x on S^2 : w3D . x = 0} */
    var w3D = [0, 0, 0];
    for (var jq = 0; jq < K; jq++) {
      if (!activeMask[jq]) continue;
      w3D[0] += coefs[jq] * queryXYZ[jq][0];
      w3D[1] += coefs[jq] * queryXYZ[jq][1];
      w3D[2] += coefs[jq] * queryXYZ[jq][2];
    }
    var wn = Math.sqrt(w3D[0] * w3D[0] + w3D[1] * w3D[1] + w3D[2] * w3D[2]);
    if (wn > 1e-4) {
      var nVec = [w3D[0] / wn, w3D[1] / wn, w3D[2] / wn];
      var ref = Math.abs(nVec[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
      var g1 = M.normVec([ref[1] * nVec[2] - ref[2] * nVec[1], ref[2] * nVec[0] - ref[0] * nVec[2], ref[0] * nVec[1] - ref[1] * nVec[0]]);
      var g2 = [nVec[1] * g1[2] - nVec[2] * g1[1], nVec[2] * g1[0] - nVec[0] * g1[2], nVec[0] * g1[1] - nVec[1] * g1[0]];
      for (var pass = 0; pass < 2; pass++) {
        ctx.beginPath();
        ctx.setLineDash(pass === 0 ? [3 * s.sc, 3 * s.sc] : []);
        ctx.strokeStyle = pass === 0 ? M.rgba(C.ink, 0.30) : M.rgba(C.ink, 0.85);
        ctx.lineWidth = (pass === 0 ? 1.1 : 1.9) * s.sc;
        var started = false;
        for (var gi = 0; gi <= 72; gi++) {
          var th = (gi / 72) * Math.PI * 2;
          var gv = [Math.cos(th) * g1[0] + Math.sin(th) * g2[0], Math.cos(th) * g1[1] + Math.sin(th) * g2[1], Math.cos(th) * g1[2] + Math.sin(th) * g2[2]];
          var gp = M.sphereProject(gv, cx, cy, R, yaw, pitch);
          var front = gp.z >= -0.02;
          if ((pass === 0 && !front) || (pass === 1 && front)) {
            if (!started) { ctx.moveTo(gp.x, gp.y); started = true; } else ctx.lineTo(gp.x, gp.y);
          } else started = false;
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    /* 3. Embedded stimuli dots */
    for (var i = 0; i < imgXYZ.length; i++) {
      var p = M.sphereProject(imgXYZ[i], cx, cy, R, yaw, pitch);
      var alpha = p.z >= 0 ? 0.88 : 0.28;
      M.dot(ctx, p.x, p.y, (p.z >= 0 ? 3.3 : 2.4) * s.sc, M.rgba(D.y[i] ? C.orange : C.blue, alpha), p.z >= 0 ? C.surface : null, 0.7);
    }

    /* 4. Anchor query pins on the sphere surface */
    for (var j = 0; j < K; j++) {
      var qp = M.sphereProject(queryXYZ[j], cx, cy, R, yaw, pitch);
      var qCol = Q_COLORS[j % Q_COLORS.length];
      var on = activeMask[j];
      M.dot(ctx, qp.x, qp.y, (on ? 5.2 : 3.2) * s.sc, on ? qCol : M.rgba(qCol, 0.28), C.surface, (on ? 1.5 : 0.8) * s.sc);
      if (on) {
        M.pill(ctx, s, 'Q' + (j + 1), qp.x, qp.y - 12 * s.sc, { align: 'center', color: qCol, size: 8.0 });
      }
    }
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);

    var maxAbs = Math.max(0.5, Math.max.apply(null, coefs.map(Math.abs)) * 1.18);
    var labelW = 112 * s.sc;
    var xLeft = labelW + 10 * s.sc, xRight = s.w - 16 * s.sc;
    var xZero = (xLeft + xRight) / 2, halfSpan = (xRight - xLeft) / 2;
    var topY = 24 * s.sc, botY = s.h - 34 * s.sc;
    var rowH = (botY - topY) / K;

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
      ctx.font = M.font(s, 8.6, on ? 600 : 400, 'sans');
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('Q' + (j + 1) + ' "' + D.queries[j] + '"', labelW, cy);

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
    var interp = accNow >= 0.85
      ? '<span class="g">Active anchors span both shape hemispheres, aligning the great-circle decision boundary cleanly between ' + clsNames[0] + ' and ' + clsNames[1] + '.</span>'
      : '<span class="o">Without opposing shape anchors on the sphere, the spanned subspace collapses and classification accuracy drops.</span>';
    read.innerHTML = '<span class="k">query subspace</span> ' + nOn + ' / ' + K + ' anchors' +
      '<span class="sep">&middot;</span><span class="k">classifier accuracy</span> ' + Math.round(accNow * 100) + '%' +
      '<span class="sep">&middot;</span><span class="k">strongest cue</span> "' + M.esc(D.queries[bestJ]) + '" (' + (coefs[bestJ] >= 0 ? '+' : '') + M.f(coefs[bestJ], 2) + ')' +
      '<div class="wread-interp">' + interp + '</div>';
  }

  fitQueryClassifier();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return {};
});
