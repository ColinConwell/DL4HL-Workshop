/* ══════════════════════════════════════════════════════════════════
   Figure 5 — Visual hierarchy & transfer learning: scratch vs. frozen
   backbone vs. full finetuning
   Pretrained vision backbones arrive with general-purpose edge, texture,
   and part detectors already organized across depth. Freezing early
   blocks and fitting a lightweight linear head generalizes from small
   research samples where training from scratch overfits.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('transfer_hierarchy', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 5,
    title: 'Transfer learning: reusing a visual hierarchy instead of starting from noise',
    hint: 'Switch regime between Frozen head, Full finetune, and Train from scratch &middot; vary Labeled sample size N'
  });

  var regime = cfg.regime || 'probe';
  var nTrain = cfg.n || 120;
  var epoch = 10;
  var ready = false;

  var STAGES = [
    { name: 'Stem', sub: 'oriented edges', params: '9.4K' },
    { name: 'Block 1\u20132', sub: 'textures & corners', params: '0.68M' },
    { name: 'Block 3\u20134', sub: '3D parts & shadows', params: '10.5M' },
    { name: 'Linear head', sub: '6 shape classes', params: '3.1K' }
  ];

  function stageStatus(idx) {
    if (regime === 'scratch') return { state: 'random \u2192 train', trainable: true, quality: Math.min(1, 0.18 + (nTrain / 850) * (epoch / 10)) };
    if (regime === 'probe') {
      return idx < 3
        ? { state: 'frozen pretrained', trainable: false, quality: 0.95 }
        : { state: 'trainable head', trainable: true, quality: Math.min(0.98, 0.45 + 0.53 * (1 - Math.exp(-epoch / 2.5))) };
    }
    return idx < 3
      ? { state: 'gentle finetune', trainable: true, quality: Math.min(0.99, 0.88 + 0.1 * (nTrain / 720)) }
      : { state: 'trainable head', trainable: true, quality: Math.min(0.99, 0.50 + 0.48 * (1 - Math.exp(-epoch / 2.2))) };
  }

  function valAccCurve(mode, nVal, ep) {
    var prog = 1 - Math.exp(-ep / (mode === 'probe' ? 2.2 : mode === 'finetune' ? 3.0 : 5.2));
    if (mode === 'probe') {
      var asymP = 0.56 + 0.37 * (1 - Math.exp(-nVal / 95));
      return 0.167 + (asymP - 0.167) * prog;
    }
    if (mode === 'finetune') {
      var overfitPenalty = nVal < 75 ? 0.08 * (1 - nVal / 75) * Math.max(0, (ep - 4) / 8) : 0;
      var asymF = 0.52 + 0.44 * (1 - Math.exp(-nVal / 115));
      return 0.167 + (asymF - 0.167) * prog - overfitPenalty;
    }
    /* Scratch needs much larger N to build invariant edge/part detectors */
    var asymS = 0.19 + 0.64 * (1 - Math.exp(-nVal / 520));
    return 0.167 + (asymS - 0.167) * prog;
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.15);
  M.mini(split.a, 'backbone stages: frozen vs. trainable parameters');
  M.mini(split.b, 'held-out validation accuracy across sample size N');
  var hf = function (w) { return M.clamp(w * 0.72, 230, 310); };
  var left = M.stage(split.a, { height: hf, ref: 420, label: 'Four ResNet stages showing frozen versus trainable layers and feature tiles' });
  var right = M.stage(split.b, { height: function () { return left.h || 255; }, ref: 360, label: 'Sample efficiency curves comparing linear probe, finetuning, and scratch' });

  M.legend(root, [
    { color: C.blue, label: 'frozen backbone + linear head', line: true },
    { color: C.green, label: 'full finetuning', line: true },
    { color: C.orange, label: 'trained from random init (scratch)', line: true }
  ]);

  var tr = M.transport(root, {
    n: 13, fps: 6, aria: 'Training epoch',
    fmt: function (k) { return 'epoch ' + k + ' / 12'; },
    onFrame: function (k) { epoch = k; drawAll(); }
  });
  tr.show(10);

  var row = M.row(root);
  var regT = M.toggle(row, {
    label: 'Strategy', value: regime,
    options: [['probe', 'Frozen head'], ['finetune', 'Full finetune'], ['scratch', 'From scratch']],
    onChange: function (v) { regime = v; drawAll(); }
  });
  var nS = M.slider(row, {
    label: 'Labeled images N', min: 24, max: 720, step: 24, value: nTrain, accent: 'blue',
    onInput: function (v) { nTrain = v; drawAll(); }
  });
  M.spacer(row);
  M.resetButton(row, function () {
    tr.pause();
    regime = 'probe'; nTrain = 120;
    regT.set('probe', true); nS.set(120, true); tr.show(10);
    drawAll();
  });
  var read = M.readout(root);

  var vR = M.view(right, { x: [24, 720], y: [0.12, 1.02], pad: { l: 44, r: 14, t: 16, b: 36 } });

  function drawFeatureGlyph(ctx, cx, cy, r, stageIdx, qual, s) {
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = C.codeBg; ctx.fill();
    ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.stroke();
    ctx.clip();
    var rng = M.rng(101 + stageIdx * 31);
    /* Blend structured feature motif with noise when quality is low (scratch) */
    ctx.globalAlpha = M.clamp(0.2 + 0.8 * qual, 0.2, 1);
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1.6 * s.sc;
    if (stageIdx === 0) {
      /* Oriented Gabor bars */
      for (var k = -2; k <= 2; k++) {
        ctx.beginPath();
        ctx.moveTo(cx - r * 0.7, cy + k * r * 0.28 - r * 0.25);
        ctx.lineTo(cx + r * 0.7, cy + k * r * 0.28 + r * 0.25);
        ctx.stroke();
      }
    } else if (stageIdx === 1) {
      /* Lattice / texture rings */
      for (var a = -1; a <= 1; a++) for (var b = -1; b <= 1; b++) {
        M.dot(ctx, cx + a * r * 0.46, cy + b * r * 0.46, r * 0.16, null, C.blue, 1.3 * s.sc);
      }
    } else if (stageIdx === 2) {
      /* 3D shaded sphere/cone contour */
      var g = ctx.createRadialGradient(cx - r * 0.25, cy - r * 0.25, r * 0.08, cx, cy, r * 0.72);
      g.addColorStop(0, C.surface);
      g.addColorStop(1, C.blue);
      M.dot(ctx, cx, cy, r * 0.62, g, C.ink, 1.1 * s.sc);
    } else {
      /* 6-class readout bars */
      var bars = [0.2, 0.88, 0.32, 0.15, 0.42, 0.24];
      for (var bi = 0; bi < 6; bi++) {
        var bx = cx - r * 0.68 + bi * (r * 0.24);
        var bh = bars[bi] * qual * r * 1.1;
        ctx.fillStyle = bi === 1 ? C.green : C.muted;
        ctx.fillRect(bx, cy + r * 0.55 - bh, r * 0.17, bh);
      }
    }
    if (qual < 0.75) {
      ctx.globalAlpha = (1 - qual) * 0.65;
      for (var n = 0; n < 26; n++) {
        ctx.fillStyle = n % 2 ? C.orange : C.muted;
        ctx.fillRect(cx + rng.uniform(-r, r), cy + rng.uniform(-r, r), 2.5 * s.sc, 2.5 * s.sc);
      }
    }
    ctx.restore();
  }

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    M.clear(ctx, s);
    var padX = 14 * s.sc, gap = 10 * s.sc;
    var rowH = (s.h - 24 * s.sc - 3 * gap) / 4;

    for (var i = 0; i < 4; i++) {
      var st = STAGES[i], info = stageStatus(i);
      var y0 = 12 * s.sc + i * (rowH + gap);
      var x0 = padX, w = s.w - 2 * padX;

      ctx.fillStyle = info.trainable ? M.rgba(C.blue, 0.08) : C.surface;
      ctx.strokeStyle = info.trainable ? (regime === 'scratch' ? C.orange : C.blue) : C.border;
      ctx.lineWidth = (info.trainable ? 1.6 : 1.1) * s.sc;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x0, y0, w, rowH, 5 * s.sc); else ctx.rect(x0, y0, w, rowH);
      ctx.fill(); ctx.stroke();

      var gr = Math.min(rowH * 0.38, 19 * s.sc);
      drawFeatureGlyph(ctx, x0 + 26 * s.sc, y0 + rowH / 2, gr, i, info.quality, s);

      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillStyle = C.ink; ctx.font = M.font(s, 10.5, 600, 'sans');
      ctx.fillText(st.name + ' \u00b7 ' + st.sub, x0 + 54 * s.sc, y0 + rowH * 0.36);

      ctx.fillStyle = C.muted; ctx.font = M.font(s, 9);
      ctx.fillText(st.params + ' params  \u00b7  ' + info.state, x0 + 54 * s.sc, y0 + rowH * 0.70);

      var badgeCol = info.trainable ? (regime === 'scratch' ? C.orange : C.green) : C.muted;
      M.pill(ctx, s, info.trainable ? 'UPDATING' : 'FROZEN', x0 + w - 8 * s.sc, y0 + rowH / 2, {
        align: 'right', color: badgeCol, size: 8.5
      });

      if (i < 3) {
        M.arrow(ctx, x0 + 26 * s.sc, y0 + rowH + 1, x0 + 26 * s.sc, y0 + rowH + gap - 1, C.muted, 1.2 * s.sc, 5 * s.sc);
      }
    }
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    vR.update();
    M.clear(ctx, s);
    M.axes(ctx, vR, s, {
      xlabel: 'labeled training images (N)',
      ylabel: 'held-out accuracy',
      yfmt: function (t) { return Math.round(t * 100) + '%'; },
      nx: 5, ny: 4
    });

    ctx.save(); vR.clip(ctx);
    ctx.setLineDash([3 * s.sc, 3 * s.sc]);
    ctx.strokeStyle = M.rgba(C.muted, 0.5);
    ctx.beginPath(); ctx.moveTo(vR.L, vR.py(1 / 6)); ctx.lineTo(vR.R, vR.py(1 / 6)); ctx.stroke();
    ctx.setLineDash([]);

    var modes = [
      ['probe', C.blue],
      ['finetune', C.green],
      ['scratch', C.orange]
    ];
    modes.forEach(function (pair) {
      var m = pair[0], col = pair[1];
      ctx.strokeStyle = col;
      ctx.lineWidth = (m === regime ? 2.8 : 1.5) * s.sc;
      ctx.globalAlpha = m === regime ? 1 : 0.55;
      ctx.beginPath();
      for (var k = 0; k <= 60; k++) {
        var nn = 24 + (k / 60) * (720 - 24);
        var px = vR.px(nn), py = vR.py(valAccCurve(m, nn, epoch));
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    var curX = vR.px(nTrain), curY = vR.py(valAccCurve(regime, nTrain, epoch));
    ctx.strokeStyle = M.rgba(C.ink, 0.35);
    ctx.lineWidth = 1.2 * s.sc;
    ctx.beginPath(); ctx.moveTo(curX, vR.T); ctx.lineTo(curX, vR.B); ctx.stroke();
    M.dot(ctx, curX, curY, 5.0 * s.sc, regime === 'probe' ? C.blue : regime === 'finetune' ? C.green : C.orange, C.surface, 1.5 * s.sc);
    ctx.restore();
  }

  function drawAll() {
    if (!ready) return;
    drawLeft(); drawRight();
    var acc = valAccCurve(regime, nTrain, epoch);
    var trainableShare = regime === 'probe' ? '0.03% (3.1K / 11.2M)' : '100% (11.2M / 11.2M)';
    var msg = regime === 'probe'
      ? '<span class="g">Linear probing tunes only the 3.1K head weights while keeping pretrained 3D shape features intact.</span>'
      : regime === 'finetune'
        ? '<span class="b">Full finetuning adapts every block, pulling ahead once N is large enough to avoid overfitting.</span>'
        : '<span class="r">Training from scratch must re-invent edges and surfaces from ' + nTrain + ' images alone.</span>';
    read.innerHTML = '<span class="k">epoch</span> ' + epoch +
      '<span class="sep">&middot;</span><span class="k">trainable params</span> ' + trainableShare +
      '<span class="sep">&middot;</span><span class="k">val accuracy (N=' + nTrain + ')</span> ' + Math.round(acc * 100) + '%' +
      '<div class="wread-interp">' + msg + '</div>';
  }

  left.draw = drawAll; right.draw = drawAll;
  ready = true;
  tr.show(10);
  left.request(); right.request();
  return { destroy: function () { tr.pause(); } };
});
