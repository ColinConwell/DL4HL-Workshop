/* ══════════════════════════════════════════════════════════════════
   Figure 6 — (GPT-Style) LLMs: from subword tokens to a trajectory on
   the unit sphere pointing toward [NEXT]
   Inspired directly by Colin Conwell's LLM-Explainer schematic:
   words split into subword [Tokens] -> integer [IDs] -> dense vectors ->
   hierarchical multi-head causal attention -> a residual trajectory on
   the hypersphere converging onto the predicted next token.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('llm_explainer', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 7,
    title: '(GPT-Style) LLMs: from subword tokens to next-token trajectories',
    hint: 'Press play to trace tokens through attention layers &middot; slide Attention head across all 12 heads &middot; drag the 3D sphere or adjust Temperature'
  });

  function makeFallbackHeads(baseA, baseB, n) {
    var out = [];
    for (var h = 0; h < 12; h++) {
      var rng = M.rng(71 + h * 19 + n);
      var row = new Array(n), s = 0;
      for (var i = 0; i < n; i++) {
        var mix = (h % 2 === 0 ? baseA[i] : baseB[i]) || (1 / n);
        var focus = (h % 4 === 2 && i === n - 2) ? 0.35 : (h % 4 === 3 && i === 0) ? 0.30 : 0;
        row[i] = Math.max(0.01, mix * (0.65 + 0.7 * rng.next()) + focus);
        s += row[i];
      }
      out.push(row.map(function (v) { return +(v / s).toFixed(3); }));
    }
    return out;
  }

  var FALLBACK_CASES = [
    {
      prompt: 'Words become tokens that point toward',
      tokens: ['Words', ' be', 'come', ' to', 'kens', ' that', ' point', ' toward'],
      ids: [24162, 307, 2925, 284, 34022, 326, 1291, 6260],
      candidates: ['the', 'a', 'their', 'each', 'specific', 'an'],
      logits: [0, -0.78, -2.14, -2.85, -3.40, -3.52],
      cand_xyz: [[0.74, 0.26, 0.62], [0.52, 0.62, 0.59], [0.22, -0.58, 0.78], [-0.14, 0.76, 0.63], [0.10, 0.88, -0.46], [-0.42, -0.45, 0.79]],
      traj_xyz: [[-0.66, -0.38, 0.65], [-0.30, -0.16, 0.94], [0.14, 0.06, 0.99], [0.52, 0.20, 0.83], [0.71, 0.31, 0.63]],
      attentions: [
        makeFallbackHeads([0.42, 0.06, 0.06, 0.07, 0.09, 0.08, 0.10, 0.12], [0.10, 0.12, 0.18, 0.10, 0.12, 0.11, 0.13, 0.14], 8),
        makeFallbackHeads([0.18, 0.05, 0.08, 0.06, 0.28, 0.07, 0.16, 0.12], [0.08, 0.04, 0.06, 0.05, 0.22, 0.12, 0.28, 0.15], 8),
        makeFallbackHeads([0.14, 0.03, 0.04, 0.04, 0.24, 0.06, 0.29, 0.16], [0.11, 0.03, 0.05, 0.04, 0.19, 0.08, 0.32, 0.18], 8)
      ]
    },
    {
      prompt: 'In clinical interviews, anxious patients often describe',
      tokens: ['In', ' clinical', ' interviews', ',', ' anxious', ' patients', ' often', ' describe'],
      ids: [818, 7603, 11791, 11, 18332, 5827, 1690, 6901],
      candidates: ['a', 'their', 'feeling', 'the', 'how', 'symptoms'],
      logits: [0, -0.18, -0.92, -1.05, -1.45, -1.95],
      cand_xyz: [[0.74, 0.26, 0.62], [0.64, 0.50, 0.58], [0.42, -0.46, 0.78], [0.35, 0.72, 0.60], [-0.08, -0.66, 0.75], [-0.34, 0.56, 0.75]],
      traj_xyz: [[-0.66, -0.38, 0.65], [-0.28, -0.14, 0.95], [0.16, 0.09, 0.98], [0.50, 0.24, 0.83], [0.69, 0.34, 0.64]],
      attentions: [
        makeFallbackHeads([0.30, 0.12, 0.10, 0.08, 0.11, 0.10, 0.09, 0.10], [0.08, 0.14, 0.14, 0.06, 0.18, 0.18, 0.10, 0.12], 8),
        makeFallbackHeads([0.10, 0.16, 0.11, 0.04, 0.27, 0.20, 0.05, 0.07], [0.06, 0.12, 0.09, 0.04, 0.31, 0.22, 0.07, 0.09], 8),
        makeFallbackHeads([0.07, 0.13, 0.08, 0.03, 0.33, 0.21, 0.06, 0.09], [0.05, 0.11, 0.08, 0.03, 0.28, 0.26, 0.08, 0.11], 8)
      ]
    },
    {
      prompt: 'Deep networks untangle knotted data one layer at',
      tokens: ['Deep', ' networks', ' unt', 'angle', ' kn', 'otted', ' data', ' one', ' layer', ' at'],
      ids: [29744, 7686, 4058, 12529, 17205, 19889, 1366, 530, 7679, 379],
      candidates: ['a', 'each', 'the', 'every', 'once', 'time'],
      logits: [0, -1.95, -2.30, -2.65, -2.90, -3.10],
      cand_xyz: [[0.74, 0.26, 0.62], [0.36, 0.72, 0.59], [0.22, -0.62, 0.75], [-0.18, 0.74, 0.65], [-0.38, -0.52, 0.76], [0.05, 0.89, -0.45]],
      traj_xyz: [[-0.66, -0.38, 0.65], [-0.26, -0.14, 0.96], [0.20, 0.08, 0.98], [0.55, 0.21, 0.81], [0.74, 0.26, 0.62]],
      attentions: [
        makeFallbackHeads([0.25, 0.09, 0.08, 0.08, 0.07, 0.07, 0.09, 0.09, 0.09, 0.09], [0.10, 0.10, 0.08, 0.10, 0.08, 0.08, 0.10, 0.14, 0.12, 0.10], 10),
        makeFallbackHeads([0.08, 0.07, 0.06, 0.07, 0.05, 0.05, 0.09, 0.24, 0.19, 0.10], [0.06, 0.08, 0.08, 0.09, 0.06, 0.06, 0.10, 0.22, 0.17, 0.08], 10),
        makeFallbackHeads([0.05, 0.05, 0.04, 0.05, 0.04, 0.04, 0.07, 0.34, 0.22, 0.10], [0.04, 0.06, 0.05, 0.06, 0.04, 0.04, 0.08, 0.31, 0.23, 0.09], 10)
      ]
    }
  ];

  var cases = (cfg.data && cfg.data.cases && cfg.data.cases.length) ? cfg.data.cases : FALLBACK_CASES;
  var caseIdx = 0;
  var stepPos = 4; // 0..4 across [Embed, Attn L1, Attn L6, Attn L12, Readout]
  var headIdx = 0;
  var temp = cfg.temperature || 0.8;
  var yaw = -0.22, pitch = 0.22, dragOrigin = null;
  var ready = false;

  function curCase() { return cases[caseIdx % cases.length]; }
  function numHeads(c) {
    var m0 = c && c.attentions && c.attentions[0];
    return (m0 && m0.length) ? m0.length : 12;
  }

  function lastTokenAttn(c, layerSlot, hSlot) {
    var layerMat = c.attentions[Math.min(c.attentions.length - 1, layerSlot)];
    var head = layerMat[hSlot % layerMat.length];
    var row = Array.isArray(head[0]) ? head[head.length - 1] : head;
    return row.slice(0, c.tokens.length);
  }

  function interpTraj(c, pos) {
    var trPts = c.traj_xyz;
    var i0 = Math.min(trPts.length - 2, Math.floor(pos));
    var t = M.clamp(pos - i0, 0, 1);
    var v = [
      M.lerp(trPts[i0][0], trPts[i0 + 1][0], t),
      M.lerp(trPts[i0][1], trPts[i0 + 1][1], t),
      M.lerp(trPts[i0][2], trPts[i0 + 1][2], t)
    ];
    return M.normVec(v);
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.18);
  M.mini(split.a, '[tokens] \u2192 [IDs] \u2192 vectors \u2192 multi-head causal attention');
  M.mini(split.b, 'residual trajectory on unit sphere \u2192 [NEXT] probabilities');
  var hf = function (w) { return M.clamp(w * 0.80, 255, 340); };
  var left = M.stage(split.a, { height: hf, ref: 430, label: 'Subword tokenization, token IDs, vector dots, and causal attention arcs' });
  var right = M.stage(split.b, { height: function () { return left.h || 275; }, ref: 360, label: '3D unit sphere showing residual stream trajectory pointing to next-token candidates' });

  var tr = M.transport(root, {
    n: 81, fps: 28, aria: 'Transformer depth from token embedding to final next-token readout',
    fmt: function (k) {
      var labels = ['1. Embed', '2. Early L1', '3. Mid L6', '4. Deep L12', '5. [NEXT] readout'];
      var p = (k / 80) * 4;
      return labels[Math.round(p)];
    },
    onFrame: function (k) { stepPos = (k / 80) * 4; drawAll(); }
  });
  tr.show(80);

  var row1 = M.row(root);
  var promptOpts = cases.map(function (c, idx) {
    var words = c.prompt.split(' ').slice(0, 2).join(' ') + '\u2026';
    return [String(idx), words];
  });
  var promptT = M.toggle(row1, {
    label: 'Prompt', value: '0', options: promptOpts,
    onChange: function (v) {
      caseIdx = +v;
      var mxH = numHeads(curCase());
      headIdx = Math.min(headIdx, mxH - 1);
      headS.input.max = mxH;
      headS.set(headIdx + 1, true);
      drawAll();
    }
  });

  var row2 = M.row(root);
  var maxHeadsInit = numHeads(curCase());
  var headS = M.slider(row2, {
    label: 'Attention head', min: 1, max: maxHeadsInit, step: 1, value: 1, accent: 'blue',
    fmt: function (v) { return 'H' + Math.round(v) + ' / ' + numHeads(curCase()); },
    onInput: function (v) { headIdx = Math.round(v) - 1; drawAll(); }
  });
  var tempS = M.slider(row2, {
    label: 'Temperature', min: 0.2, max: 1.8, step: 0.05, value: temp, accent: 'orange',
    fmt: function (v) { return M.f(v, 2); },
    onInput: function (v) { temp = v; drawAll(); }
  });
  M.spacer(row2);
  M.resetButton(row2, function () {
    tr.pause();
    caseIdx = 0; headIdx = 0; temp = 0.8; yaw = -0.22; pitch = 0.22;
    promptT.set('0', true);
    headS.input.max = numHeads(curCase());
    headS.set(1, true);
    tempS.set(0.8, true);
    tr.show(80); drawAll();
  });
  var read = M.readout(root);

  M.pointer(right.cv, {
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
    var c = curCase(), toks = c.tokens, n = toks.length;
    var padX = 12 * s.sc, usableW = s.w - 2 * padX;
    var stepW = usableW / n;

    /* Row 1: [Token] chips, integer [IDs], and 4-dot token vectors */
    var yTok = 24 * s.sc, yId = 46 * s.sc, yVec = 66 * s.sc;
    ctx.textAlign = 'center';
    for (var i = 0; i < n; i++) {
      var cx = padX + (i + 0.5) * stepW;
      var col = M.catColor(i);
      var chipW = Math.max(22 * s.sc, stepW - 4 * s.sc);

      ctx.fillStyle = M.rgba(col, 0.14);
      ctx.strokeStyle = M.rgba(col, 0.72);
      ctx.lineWidth = 1.1 * s.sc;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(cx - chipW / 2, yTok - 10 * s.sc, chipW, 20 * s.sc, 4 * s.sc);
      else ctx.rect(cx - chipW / 2, yTok - 10 * s.sc, chipW, 20 * s.sc);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.6, 600, 'sans');
      ctx.textBaseline = 'middle';
      var lbl = toks[i].trim() || '\u2423';
      if (lbl.length > 7) lbl = lbl.slice(0, 6) + '\u2026';
      ctx.fillText(lbl, cx, yTok);

      ctx.fillStyle = C.muted; ctx.font = M.font(s, 7.6);
      ctx.fillText('[' + c.ids[i] + ']', cx, yId);

      for (var d = -1.5; d <= 1.5; d += 1) {
        M.dot(ctx, cx + d * 4.2 * s.sc, yVec, 1.65 * s.sc, col);
      }
    }

    /* Top-of-band pill explaining what the arcs below represent (well clear of bottom output label!) */
    var attnSlot = stepPos < 1.5 ? 0 : stepPos < 2.8 ? 1 : 2;
    var nH = numHeads(c);
    var hNum = (headIdx % nH) + 1;
    var stageLabel = stepPos < 0.4
      ? 'Step 1 (Embeddings): each token holds its uncontextualized lookup vector'
      : attnSlot === 0
        ? 'Attention arcs (Early L1 \u00b7 H' + hNum + '): final token gathers local neighbor cues'
        : attnSlot === 1
          ? 'Attention arcs (Mid L6 \u00b7 H' + hNum + '): final token binds phrase structure'
          : 'Attention arcs (Deep L12 \u00b7 H' + hNum + '): final token pools semantic context';
    M.pill(ctx, s, stageLabel, padX, 86 * s.sc, { color: C.ink, size: 8.1 });

    /* Row 2: Causal attention arcs routing earlier tokens into the final query token */
    var ySrc = 102 * s.sc, yNode = s.h - 36 * s.sc;
    var weights = lastTokenAttn(c, attnSlot, headIdx);
    var qX = padX + (n - 0.5) * stepW;

    for (var j = 0; j < n; j++) {
      var sx = padX + (j + 0.5) * stepW;
      var w = weights[j] || 0;
      var alpha = stepPos < 0.35 ? 0.12 : M.clamp(0.08 + w * 2.4, 0.08, 0.95);
      ctx.strokeStyle = M.rgba(M.catColor(j), alpha);
      ctx.lineWidth = (0.8 + w * 7.2) * s.sc;
      ctx.beginPath();
      ctx.moveTo(sx, ySrc);
      ctx.bezierCurveTo(sx, (ySrc + yNode) * 0.54, qX, (ySrc + yNode) * 0.54, qX, yNode - 7 * s.sc);
      ctx.stroke();
    }

    /* Explicit pooled residual vector node at convergence point + non-overlapping bottom caption */
    M.dot(ctx, qX, yNode, 5.5 * s.sc, C.blue, C.surface, 1.5 * s.sc);
    M.pill(ctx, s, 'Final token state h_L \u2192 sphere arrow \u2192 predicts [NEXT]', s.w - padX, s.h - 13 * s.sc, {
      align: 'right', color: C.blue, size: 8.2
    });
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);
    var c = curCase();

    var R = Math.min(s.w * 0.28, s.h * 0.34);
    var cx = R + 18 * s.sc, cy = s.h * 0.47;
    M.drawSphere(ctx, cx, cy, R, yaw, pitch, s);

    /* Plot trajectory path across depth + current trajectory vector arrow */
    var curVec = interpTraj(c, stepPos);
    ctx.strokeStyle = M.rgba(C.purple || C.blue, 0.50);
    ctx.lineWidth = 1.7 * s.sc;
    ctx.beginPath();
    for (var k = 0; k <= 24; k++) {
      var u = (k / 24) * stepPos;
      var pv = M.sphereProject(interpTraj(c, u), cx, cy, R, yaw, pitch);
      if (k === 0) ctx.moveTo(pv.x, pv.y); else ctx.lineTo(pv.x, pv.y);
    }
    ctx.stroke();

    /* Candidate token anchors on the sphere (draw before arrow tip so both are crisp) */
    var scaledLogits = c.logits.map(function (l) { return l * (0.25 + 0.75 * (stepPos / 4)); });
    var probs = M.softmax(scaledLogits, temp);
    for (var m = c.candidates.length - 1; m >= 0; m--) {
      var cp = M.sphereProject(c.cand_xyz[m], cx, cy, R, yaw, pitch);
      var isTop = m === 0;
      M.dot(ctx, cp.x, cp.y, (isTop ? 5.2 : 3.3) * s.sc, isTop ? C.orange : M.rgba(C.muted, cp.z >= 0 ? 0.85 : 0.35), C.surface, 1.2 * s.sc);
      if (isTop) {
        M.pill(ctx, s, '[' + c.candidates[0] + ']', cp.x, cp.y - 11 * s.sc, { align: 'center', color: C.orange, size: 8.0 });
      }
    }

    var tip = M.sphereProject(curVec, cx, cy, R, yaw, pitch);
    M.arrow(ctx, cx, cy, tip.x, tip.y, C.blue, 2.5 * s.sc, 8 * s.sc);

    /* Candidate probability bar chart on the right side of right panel */
    var bx0 = cx + R + 16 * s.sc, bwMax = Math.max(45 * s.sc, s.w - bx0 - 12 * s.sc);
    var topY = 26 * s.sc, rowH = (s.h - 44 * s.sc) / c.candidates.length;
    ctx.font = M.font(s, 9, 600);
    ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('P([NEXT] | context)', bx0, topY - 4 * s.sc);

    for (var r = 0; r < c.candidates.length; r++) {
      var yy = topY + r * rowH + rowH * 0.5;
      var p = probs[r];
      ctx.fillStyle = M.rgba(r === 0 ? C.orange : C.blue, 0.22);
      ctx.fillRect(bx0, yy - 6 * s.sc, bwMax, 12 * s.sc);
      ctx.fillStyle = r === 0 ? C.orange : C.blue;
      ctx.fillRect(bx0, yy - 6 * s.sc, bwMax * p, 12 * s.sc);

      ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.8, r === 0 ? 600 : 400);
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText('[' + c.candidates[r] + ']', bx0 + 4 * s.sc, yy);
      ctx.textAlign = 'right';
      ctx.fillText(Math.round(p * 100) + '%', bx0 + bwMax - 3 * s.sc, yy);
    }
  }

  function drawAll() {
    if (!ready) return;
    drawLeft(); drawRight();
    var c = curCase();
    var nH = numHeads(c);
    var scaled = c.logits.map(function (l) { return l * (0.25 + 0.75 * (stepPos / 4)); });
    var probs = M.softmax(scaled, temp);
    read.innerHTML = '<span class="k">tokens</span> ' + c.tokens.length +
      '<span class="sep">&middot;</span><span class="k">head</span> H' + ((headIdx % nH) + 1) + ' / ' + nH +
      '<span class="sep">&middot;</span><span class="k">top [NEXT]</span> <b>[' + M.esc(c.candidates[0]) + ']</b> (' + Math.round(probs[0] * 100) + '% at T=' + M.f(temp, 2) + ')' +
      '<div class="wread-interp"><span class="b">Causal attention heads route earlier token vectors into a residual trajectory converging on the most probable next token.</span></div>';
  }

  left.draw = drawAll; right.draw = drawAll;
  ready = true;
  tr.show(80);
  left.request(); right.request();
  return { destroy: function () { tr.pause(); } };
});
