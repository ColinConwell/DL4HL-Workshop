/* ══════════════════════════════════════════════════════════════════
   Figure 9 — Encoding & decoding across network depth (DeepDive)
   Extracting intermediate feature maps across a Vision Transformer
   (compressed via Johnson-Lindenstrauss Sparse Random Projection)
   reveals where along the hierarchy aesthetic ratings (OASIS beauty,
   valence, arousal) and 3D object identity become linearly readable.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('layerwise_readout', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 9,
    title: 'Encoding & decoding across network depth: probing a Vision Transformer',
    hint: 'Press play or scrub Layer to move from the patch stem to the final norm &middot; toggle right view between Beauty scatter and Layer RDM'
  });

  function fallbackData() {
    var layers = ['Patch stem', 'Block 1', 'Block 3', 'Block 6', 'Block 9', 'Block 12', 'Final norm'];
    var beauty = [0.134, 0.285, 0.300, 0.454, 0.620, 0.640, 0.874];
    var valence = [-0.018, 0.186, 0.259, 0.332, 0.532, 0.527, 0.861];
    var arousal = [-0.008, 0.204, 0.271, 0.481, 0.577, 0.532, 0.842];
    var shapes = [0.208, 0.153, 0.292, 0.583, 0.847, 0.736, 0.917];
    var rng = M.rng(77), yTrue = [];
    for (var i = 0; i < 80; i++) yTrue.push(rng.uniform(2.0, 6.4));
    var scat = {}, rdms = {};
    layers.forEach(function (lbl, li) {
      var r = beauty[li];
      var r2 = M.rng(200 + li * 13);
      scat[lbl] = {
        y: yTrue,
        yhat: yTrue.map(function (v) { return 4.2 + r * (v - 4.2) + Math.sqrt(Math.max(0.05, 1 - r * r)) * r2.normal(0, 0.72); })
      };
      var mat = [];
      for (var a = 0; a < 16; a++) {
        var row = [];
        for (var b = 0; b < 16; b++) {
          if (a === b) row.push(0);
          else {
            var sameGroup = Math.floor(a / 4) === Math.floor(b / 4);
            var struct = (li / 6) * (sameGroup ? 0.28 : 0.95) + (1 - li / 6) * 0.62;
            row.push(M.clamp(struct + 0.08 * Math.sin(a * 3.1 + b * 1.7 + li), 0.05, 1.35));
          }
        }
        mat.push(row);
      }
      rdms[lbl] = mat;
    });
    return {
      layers: layers,
      oasis_curves: { beauty: beauty, valence: valence, arousal: arousal },
      shapes_curve: shapes,
      beauty_scatter: scat,
      rdms: rdms
    };
  }

  var D = (cfg.data && cfg.data.layers && cfg.data.oasis_curves) ? cfg.data : fallbackData();
  var L = D.layers.length;
  var curL = L - 1;
  var rightMode = cfg.panel || 'scatter';
  var ready = false;

  /* ─── DOM ─── */
  var split = M.split(root, 1.25);
  M.mini(split.a, 'held-out encoding (r) & shape decoding (acc) across depth');
  var rightTitle = M.mini(split.b, 'held-out beauty predictions vs. human ratings');
  var hf = function (w) { return M.clamp(w * 0.74, 235, 315); };
  var left = M.stage(split.a, { height: hf, ref: 430, label: 'Line plot of OASIS beauty, valence, arousal encoding r and SimpleShapes decoding accuracy across ViT layers' });
  var right = M.stage(split.b, { height: function () { return left.h || 255; }, ref: 340, label: 'Selected layer prediction scatter or representational dissimilarity matrix' });

  M.legend(root, [
    { color: C.blue, label: 'OASIS beauty (r)', line: true },
    { color: C.green, label: 'OASIS valence (r)', line: true },
    { color: C.orange, label: 'OASIS arousal (r)', line: true },
    { color: C.purple, label: 'SimpleShapes decoding (acc)', line: true }
  ]);

  var tr = M.transport(root, {
    n: L, fps: 2, aria: 'Select Vision Transformer layer',
    fmt: function (k) { return D.layers[k] + ' (' + (k + 1) + '/' + L + ')'; },
    onFrame: function (k) { curL = k; drawAll(); }
  });
  tr.show(L - 1);

  var row = M.row(root);
  var modeT = M.toggle(row, {
    label: 'Right panel', value: rightMode,
    options: [['scatter', 'Beauty scatter'], ['rdm', 'Layer RDM (1\u2212r)']],
    onChange: function (v) {
      rightMode = v;
      rightTitle.innerHTML = v === 'scatter'
        ? 'held-out beauty predictions vs. human ratings'
        : 'layer representational dissimilarity matrix (1 \u2212 r)';
      drawAll();
    }
  });
  M.spacer(row);
  M.resetButton(row, function () {
    tr.pause();
    rightMode = 'scatter';
    modeT.set('scatter', true);
    rightTitle.innerHTML = 'held-out beauty predictions vs. human ratings';
    tr.show(L - 1);
    drawAll();
  });
  var read = M.readout(root);

  var vL = M.view(left, { x: [-0.25, L - 0.75], y: [-0.08, 1.02], pad: { l: 44, r: 14, t: 16, b: 42 } });
  var vR = M.view(right, { x: [1.5, 6.8], y: [1.5, 6.8], equal: true, pad: { l: 42, r: 14, t: 16, b: 36 } });

  M.pointer(left.cv, {
    down: function (p) {
      tr.pause();
      vL.update();
      var k = M.clamp(Math.round(vL.ix(p.x)), 0, L - 1);
      tr.show(k);
      return true;
    },
    drag: function (p) {
      var k = M.clamp(Math.round(vL.ix(p.x)), 0, L - 1);
      if (k !== curL) tr.show(k);
    }
  });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain([-0.25, L - 0.75], [-0.08, 1.02]);
    M.clear(ctx, s);
    M.axes(ctx, vL, s, {
      xlabel: 'Vision Transformer depth (shallow \u2192 deep)',
      ylabel: 'held-out score (r or acc)',
      xticks: M.range(L),
      xfmt: function (i) {
        var nm = D.layers[i] || '';
        return nm.replace('Patch stem', 'Stem').replace('Block ', 'B').replace('Final norm', 'Final');
      },
      ny: 5
    });

    ctx.save(); vL.clip(ctx);
    ctx.strokeStyle = M.rgba(C.muted, 0.45);
    ctx.setLineDash([3 * s.sc, 3 * s.sc]);
    ctx.beginPath(); ctx.moveTo(vL.L, vL.py(0)); ctx.lineTo(vL.R, vL.py(0)); ctx.stroke();
    ctx.setLineDash([]);

    /* Selected layer highlight column */
    var selX = vL.px(curL);
    ctx.fillStyle = M.rgba(C.blue, 0.08);
    ctx.fillRect(selX - 12 * s.sc, vL.T, 24 * s.sc, vL.B - vL.T);
    ctx.strokeStyle = M.rgba(C.ink, 0.45);
    ctx.lineWidth = 1.3 * s.sc;
    ctx.beginPath(); ctx.moveTo(selX, vL.T); ctx.lineTo(selX, vL.B); ctx.stroke();

    var series = [
      [D.oasis_curves.beauty, C.blue],
      [D.oasis_curves.valence, C.green],
      [D.oasis_curves.arousal, C.orange],
      [D.shapes_curve, C.purple || C.teal]
    ];
    series.forEach(function (pair) {
      var arr = pair[0], col = pair[1];
      ctx.strokeStyle = col;
      ctx.lineWidth = 2.3 * s.sc;
      ctx.beginPath();
      for (var i = 0; i < L; i++) {
        var px = vL.px(i), py = vL.py(arr[i]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      for (var k = 0; k < L; k++) {
        M.dot(ctx, vL.px(k), vL.py(arr[k]), (k === curL ? 5.0 : 3.4) * s.sc, col, C.surface, 1.2 * s.sc);
      }
    });
    ctx.restore();
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);
    var lbl = D.layers[curL];

    if (rightMode === 'scatter') {
      var sc = D.beauty_scatter[lbl];
      var allV = sc.y.concat(sc.yhat);
      var lo = Math.min.apply(null, allV) - 0.25, hi = Math.max.apply(null, allV) + 0.25;
      vR.setDomain([lo, hi], [lo, hi]);
      M.axes(ctx, vR, s, { xlabel: 'human beauty rating', ylabel: 'RidgeCV predicted beauty', nx: 4, ny: 4 });
      ctx.save(); vR.clip(ctx);
      ctx.strokeStyle = M.rgba(C.muted, 0.55);
      ctx.setLineDash([4 * s.sc, 3 * s.sc]);
      ctx.beginPath(); ctx.moveTo(vR.px(lo), vR.py(lo)); ctx.lineTo(vR.px(hi), vR.py(hi)); ctx.stroke();
      ctx.setLineDash([]);
      for (var i = 0; i < sc.y.length; i++) {
        M.dot(ctx, vR.px(sc.y[i]), vR.py(sc.yhat[i]), 3.3 * s.sc, M.rgba(C.blue, 0.72), C.surface, 0.7);
      }
      ctx.restore();
      M.pill(ctx, s, lbl + '  \u00b7  r = ' + M.f(D.oasis_curves.beauty[curL], 2), vR.L + 6 * s.sc, vR.T + 12 * s.sc);
    } else {
      var mat = D.rdms[lbl], m = mat.length;
      var pad = 22 * s.sc;
      var side = Math.min(s.w - 2 * pad, s.h - 2 * pad - 10 * s.sc);
      var x0 = (s.w - side) / 2, y0 = (s.h - side) / 2 + 4 * s.sc;
      var cell = side / m;
      var vmax = 1.3;
      for (var r = 0; r < m; r++) {
        for (var c = 0; c < m; c++) {
          var t = M.clamp(mat[r][c] / vmax, 0, 1);
          ctx.fillStyle = M.css(M.magma(t), 1);
          ctx.fillRect(x0 + c * cell, y0 + r * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
      ctx.strokeStyle = C.border; ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, side, side);
      M.pill(ctx, s, lbl + ' RDM (' + m + '\u00d7' + m + ' stimuli)', x0, y0 - 11 * s.sc, { size: 8.5 });
    }
  }

  function drawAll() {
    if (!ready) return;
    drawLeft(); drawRight();
    var lbl = D.layers[curL];
    read.innerHTML = '<span class="k">layer</span> <b>' + M.esc(lbl) + '</b>' +
      '<span class="sep">&middot;</span><span class="k">beauty r</span> ' + M.f(D.oasis_curves.beauty[curL], 2) +
      '<span class="sep">&middot;</span><span class="k">valence r</span> ' + M.f(D.oasis_curves.valence[curL], 2) +
      '<span class="sep">&middot;</span><span class="k">arousal r</span> ' + M.f(D.oasis_curves.arousal[curL], 2) +
      '<span class="sep">&middot;</span><span class="k">shape acc</span> ' + Math.round(D.shapes_curve[curL] * 100) + '%';
  }

  left.draw = drawAll; right.draw = drawAll;
  ready = true;
  tr.show(L - 1);
  left.request(); right.request();
  return { destroy: function () { tr.pause(); } };
});
