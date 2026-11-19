/* ══════════════════════════════════════════════════════════════════
   Figure 10 — Contrastive multimodal alignment: joint image & text
   embeddings on the unit sphere (CLIP / SigLIP)
   Before training, separate vision and language encoders map paired
   images and captions to unrelated directions on the hypersphere.
   Contrastive training maximizes cosine similarity along the matched
   diagonal (pulling image_i and text_i together) while minimizing
   off-diagonal similarity (spreading distinct concepts apart),
   unlocking zero-shot probing via dot products.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('clip_alignment', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 10,
    title: 'Multimodal alignment: matching images & text on a shared unit sphere',
    hint: 'Press play or scrub training from random start to aligned CLIP space &middot; click any row in the 5\u00d75 similarity table to highlight its image\u2013caption pair on the 3D sphere &middot; drag the sphere to rotate'
  });

  var PAIRS = [
    { short: 'Spiky', caption: '"jagged spiky star"', glyph: 'kiki', col: '#d97706', target: M.normVec([0.78, 0.38, 0.49]) },
    { short: 'Rounded', caption: '"smooth curvy cloud"', glyph: 'bouba', col: '#2563eb', target: M.normVec([-0.76, 0.34, 0.55]) },
    { short: '3D Cone', caption: '"shaded 3D cone"', glyph: 'cone', col: '#0f766e', target: M.normVec([0.08, -0.84, 0.54]) },
    { short: 'Hearth', caption: '"warm kitchen hearth"', glyph: 'hearth', col: '#b83a2a', target: M.normVec([-0.58, -0.44, -0.68]) },
    { short: 'Coast', caption: '"sunny ocean horizon"', glyph: 'coast', col: '#6a4c93', target: M.normVec([0.54, -0.18, -0.82]) }
  ];
  var N = PAIRS.length;

  var initImg = [], initTxt = [], alignedImg = [], alignedTxt = [];
  (function buildVectors() {
    var rng = M.rng(cfg.seed || 77);
    for (var i = 0; i < N; i++) {
      initImg.push(M.normVec([rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)]));
      initTxt.push(M.normVec([rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)]));
      var t = PAIRS[i].target;
      /* Realistic cross-modal angular offset (~26 deg, cos ~ 0.89) at convergence */
      alignedImg.push(M.normVec([t[0] + 0.17, t[1] - 0.14, t[2] + 0.10]));
      alignedTxt.push(M.normVec([t[0] - 0.16, t[1] + 0.15, t[2] - 0.09]));
    }
  })();

  var prog = 1.0;          // 0..1 contrastive training progress
  var selIdx = 0;          // currently highlighted pair 0..4
  var invTemp = cfg.scale || 8.0; // logit scale 1/tau for softmax readout
  var yaw = -0.24, pitch = 0.24, dragOrigin = null;
  var ready = false;
  var matRect = { x: 0, y: 0, w: 0, h: 0, cellW: 0, cellH: 0 };

  function slerp(a, b, t) {
    return M.normVec([M.lerp(a[0], b[0], t), M.lerp(a[1], b[1], t), M.lerp(a[2], b[2], t)]);
  }

  function currentVecs() {
    var imgs = [], txts = [];
    for (var i = 0; i < N; i++) {
      imgs.push(slerp(initImg[i], alignedImg[i], prog));
      txts.push(slerp(initTxt[i], alignedTxt[i], prog));
    }
    return { imgs: imgs, txts: txts };
  }

  function drawStimulusGlyph(ctx, kind, cx, cy, r, col, s) {
    ctx.save();
    ctx.strokeStyle = col;
    ctx.fillStyle = M.rgba(col, 0.20);
    ctx.lineWidth = 1.3 * s.sc;
    ctx.beginPath();
    if (kind === 'kiki') {
      for (var k = 0; k < 14; k++) {
        var a = -Math.PI / 2 + (k * Math.PI) / 7;
        var rad = k % 2 === 0 ? r : r * 0.38;
        ctx[k === 0 ? 'moveTo' : 'lineTo'](cx + rad * Math.cos(a), cy + rad * Math.sin(a));
      }
      ctx.closePath(); ctx.fill(); ctx.stroke();
    } else if (kind === 'bouba') {
      for (var b = 0; b <= 36; b++) {
        var ab = (b / 36) * Math.PI * 2;
        var rb = r * (0.74 + 0.22 * Math.sin(4 * ab + 0.4));
        ctx[b === 0 ? 'moveTo' : 'lineTo'](cx + rb * Math.cos(ab), cy + rb * Math.sin(ab));
      }
      ctx.closePath(); ctx.fill(); ctx.stroke();
    } else if (kind === 'cone') {
      ctx.moveTo(cx, cy - r * 0.92);
      ctx.lineTo(cx + r * 0.82, cy + r * 0.72);
      ctx.quadraticCurveTo(cx, cy + r * 1.05, cx - r * 0.82, cy + r * 0.72);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    } else if (kind === 'hearth') {
      ctx.rect(cx - r * 0.75, cy - r * 0.15, r * 1.5, r * 0.88);
      ctx.fill(); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - r * 0.35, cy - r * 0.15);
      ctx.quadraticCurveTo(cx, cy - r * 1.05, cx + r * 0.35, cy - r * 0.15);
      ctx.stroke();
    } else {
      ctx.arc(cx, cy - r * 0.22, r * 0.42, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - r * 0.9, cy + r * 0.52);
      ctx.quadraticCurveTo(cx - r * 0.3, cy + r * 0.2, cx, cy + r * 0.52);
      ctx.quadraticCurveTo(cx + r * 0.3, cy + r * 0.84, cx + r * 0.9, cy + r * 0.52);
      ctx.stroke();
    }
    ctx.restore();
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.06);
  M.mini(split.a, '5\u00d75 image\u2013caption similarity table');
  M.mini(split.b, 'shared unit sphere: images (\u25cf) & captions (\u25cb)');
  var hf = function (w) { return M.clamp(w * 0.82, 255, 335); };
  var left = M.stage(split.a, { height: hf, ref: 410, label: '5 by 5 image-caption similarity table showing matched diagonal alignment during CLIP training' });
  var right = M.stage(split.b, { height: function () { return left.h || 280; }, ref: 400, label: '3D unit sphere showing paired image and text embeddings pulling together onto shared semantic directions' });

  M.legend(root, [
    { color: C.ink, label: 'image vector (\u25cf solid)' },
    { color: C.orange, label: 'text caption vector (\u25cb ring)', ring: true },
    { color: C.green, label: 'matched pair link', line: true }
  ]);

  var tr = M.transport(root, {
    n: 61, fps: 30, aria: 'Contrastive training progression from random start to aligned multimodal sphere',
    fmt: function (k) { return 'training ' + Math.round((k / 60) * 100) + '%'; },
    onFrame: function (k) { prog = M.ease(k / 60); drawAll(); }
  });
  tr.show(60);

  var row1 = M.row(root);
  var pairT = M.toggle(row1, {
    label: 'Inspect pair', value: '0',
    options: PAIRS.map(function (p, i) { return [String(i), (i + 1) + ': ' + p.short]; }),
    onChange: function (v) { selIdx = +v; drawAll(); }
  });

  var row2 = M.row(root);
  var tempS = M.slider(row2, {
    label: 'Score sharpness', min: 2, max: 18, step: 1, value: invTemp, accent: 'blue',
    fmt: function (v) { return Math.round(v) + '\u00d7'; },
    onInput: function (v) { invTemp = v; drawAll(); }
  });
  M.spacer(row2);
  M.resetButton(row2, function () {
    tr.pause();
    selIdx = 0; invTemp = 8.0; yaw = -0.24; pitch = 0.24;
    pairT.set('0', true);
    tempS.set(8.0, true);
    tr.show(60); drawAll();
  });
  var read = M.readout(root);

  M.pointer(left.cv, {
    down: function (p) {
      if (p.x >= matRect.x && p.x <= matRect.x + matRect.w && p.y >= matRect.y && p.y <= matRect.y + matRect.h) {
        var rIdx = M.clamp(Math.floor((p.y - matRect.y) / matRect.cellH), 0, N - 1);
        selIdx = rIdx;
        pairT.set(String(selIdx), true);
        drawAll();
      }
      return false;
    },
    hover: function (p) {
      var inside = p.x >= matRect.x && p.x <= matRect.x + matRect.w && p.y >= matRect.y && p.y <= matRect.y + matRect.h;
      left.cv.style.cursor = inside ? 'pointer' : '';
    }
  });

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

  /* ─── Draw Left: Dual-Encoder Header + 5×5 Cosine Matrix ─── */
  function drawLeft(vecs) {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    M.clear(ctx, s);

    var leftPad = 82 * s.sc, topPad = 42 * s.sc, rightPad = 12 * s.sc, botPad = 24 * s.sc;
    var gw = s.w - leftPad - rightPad, gh = s.h - topPad - botPad;
    var cw = gw / N, ch = gh / N;
    matRect = { x: leftPad, y: topPad, w: gw, h: gh, cellW: cw, cellH: ch };

    ctx.fillStyle = C.muted; ctx.font = M.font(s, 8.0, 600);
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('TEXT CAPTIONS \u2192', leftPad + gw * 0.5, 15 * s.sc);

    /* Column labels (text captions T1..T5) */
    for (var j = 0; j < N; j++) {
      var cx = leftPad + (j + 0.5) * cw;
      ctx.fillStyle = j === selIdx ? PAIRS[j].col : C.ink;
      ctx.font = M.font(s, 8.0, j === selIdx ? 700 : 500, 'sans');
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText('T' + (j + 1) + ' ' + PAIRS[j].short, cx, topPad - 5 * s.sc);
    }

    /* Rows: image icon + label + 5 similarity cells */
    for (var i = 0; i < N; i++) {
      var ry = topPad + i * ch, cy = ry + ch * 0.5;
      drawStimulusGlyph(ctx, PAIRS[i].glyph, 16 * s.sc, cy, 8.2 * s.sc, PAIRS[i].col, s);
      ctx.fillStyle = i === selIdx ? PAIRS[i].col : C.ink;
      ctx.font = M.font(s, 8.2, i === selIdx ? 700 : 500, 'sans');
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(PAIRS[i].short, 29 * s.sc, cy);

      for (var c = 0; c < N; c++) {
        var rx = leftPad + c * cw;
        var sim = M.cosine(vecs.imgs[i], vecs.txts[c]);
        var u = M.clamp((sim + 0.75) / 1.75, 0, 1);
        var rgb = M.diverge(u);
        ctx.fillStyle = M.css(rgb, 0.86);
        ctx.fillRect(rx + 1, ry + 1, cw - 2, ch - 2);

        if (i === c) {
          ctx.strokeStyle = M.rgba(C.ink, 0.45);
          ctx.lineWidth = 1.0 * s.sc;
          ctx.strokeRect(rx + 1.5, ry + 1.5, cw - 3, ch - 3);
        }
        ctx.fillStyle = C.ink;
        ctx.font = M.font(s, 8.1, i === c ? 600 : 400);
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText((sim >= 0 ? '+' : '') + M.f(sim, 2), rx + cw * 0.5, cy);
      }

      if (i === selIdx) {
        ctx.strokeStyle = PAIRS[i].col;
        ctx.lineWidth = 2.0 * s.sc;
        ctx.strokeRect(leftPad, ry, gw, ch);
      }
    }

    ctx.fillStyle = C.muted; ctx.font = M.font(s, 7.8);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('Matched pairs sit along the diagonal (click any row to inspect)', leftPad + gw * 0.5, topPad + gh + 6 * s.sc);
  }

  /* ─── Draw Right: Shared 3D Unit Hypersphere ─── */
  function drawRight(vecs) {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);

    var R = Math.min(s.w, s.h - 28 * s.sc) * 0.40;
    var cx = s.w * 0.5, cy = s.h * 0.53;
    M.drawSphere(ctx, cx, cy, R, yaw, pitch, s);

    /* Geodesic arcs & markers for all 5 pairs */
    for (var i = 0; i < N; i++) {
      var col = PAIRS[i].col;
      var isSel = i === selIdx;
      var pImg = M.sphereProject(vecs.imgs[i], cx, cy, R, yaw, pitch);
      var pTxt = M.sphereProject(vecs.txts[i], cx, cy, R, yaw, pitch);
      var front = pImg.z >= -0.05 || pTxt.z >= -0.05;

      /* Geodesic arc between image_i and text_i */
      ctx.strokeStyle = M.rgba(isSel ? C.green : col, isSel ? 0.88 : (front ? 0.42 : 0.16));
      ctx.lineWidth = (isSel ? 2.1 : 1.15) * s.sc;
      ctx.beginPath();
      for (var st = 0; st <= 16; st++) {
        var ps = M.sphereProject(slerp(vecs.imgs[i], vecs.txts[i], st / 16), cx, cy, R, yaw, pitch);
        if (st === 0) ctx.moveTo(ps.x, ps.y); else ctx.lineTo(ps.x, ps.y);
      }
      ctx.stroke();

      /* Selected pair: draw both radial unit vectors from the origin! */
      if (isSel) {
        M.arrow(ctx, cx, cy, pImg.x, pImg.y, col, 2.3 * s.sc, 7.5 * s.sc);
        ctx.save();
        ctx.setLineDash([4 * s.sc, 3 * s.sc]);
        M.arrow(ctx, cx, cy, pTxt.x, pTxt.y, col, 1.9 * s.sc, 7.0 * s.sc);
        ctx.restore();
      }

      /* Image point (solid circle) & Text point (open ring) */
      var alpha = isSel ? 1.0 : (front ? 0.85 : 0.32);
      M.dot(ctx, pImg.x, pImg.y, (isSel ? 5.6 : 4.1) * s.sc, M.rgba(col, alpha), C.surface, 1.3 * s.sc);
      M.dot(ctx, pTxt.x, pTxt.y, (isSel ? 5.8 : 4.2) * s.sc, C.surface, M.rgba(col, alpha), (isSel ? 2.2 : 1.7) * s.sc);

      if (isSel || front) {
        M.pill(ctx, s, PAIRS[i].caption, pTxt.x, pTxt.y - 12 * s.sc, {
          align: 'center', color: col, size: isSel ? 8.1 : 7.2
        });
      }
    }

    var selSim = M.cosine(vecs.imgs[selIdx], vecs.txts[selIdx]);
    var deg = Math.round(Math.acos(M.clamp(selSim, -1, 1)) * (180 / Math.PI));
    var logitsRow = [];
    for (var c2 = 0; c2 < N; c2++) logitsRow.push(invTemp * M.cosine(vecs.imgs[selIdx], vecs.txts[c2]));
    var pMatch = M.softmax(logitsRow, 1.0)[selIdx];

    M.pill(
      ctx, s,
      'Pair ' + (selIdx + 1) + ' (' + PAIRS[selIdx].short + '): cos = ' + (selSim >= 0 ? '+' : '') + M.f(selSim, 2) + ' (' + deg + '\u00b0) \u00b7 P(match) = ' + Math.round(pMatch * 100) + '%',
      s.w * 0.5, 13 * s.sc,
      { align: 'center', color: PAIRS[selIdx].col, size: 7.9 }
    );
  }

  function drawAll() {
    if (!ready) return;
    var vecs = currentVecs();
    drawLeft(vecs);
    drawRight(vecs);

    var diagSum = 0, offSum = 0, hits = 0;
    for (var i = 0; i < N; i++) {
      var bestC = 0, bestS = -Infinity;
      for (var j = 0; j < N; j++) {
        var sc = M.cosine(vecs.imgs[i], vecs.txts[j]);
        if (i === j) diagSum += sc; else offSum += sc;
        if (sc > bestS) { bestS = sc; bestC = j; }
      }
      if (bestC === i) hits++;
    }
    var meanDiag = diagSum / N, meanOff = offSum / (N * (N - 1));
    var top1 = hits / N;
    var msg = prog < 0.15
      ? '<span class="o">Before training, the image and text networks point in unrelated directions; zero-shot matching sits near chance (' + Math.round(top1 * 100) + '%).</span>'
      : '<span class="g">Contrastive training pulls matched image\u2013caption pairs onto the same neighborhood of the sphere (mean similarity ' + (meanDiag >= 0 ? '+' : '') + M.f(meanDiag, 2) + ') while pushing non-matches apart (' + M.f(meanOff, 2) + ').</span>';

    read.innerHTML = '<span class="k">matched similarity</span> ' + (meanDiag >= 0 ? '+' : '') + M.f(meanDiag, 2) +
      '<span class="sep">&middot;</span><span class="k">mismatched similarity</span> ' + (meanOff >= 0 ? '+' : '') + M.f(meanOff, 2) +
      '<span class="sep">&middot;</span><span class="k">zero-shot top-1</span> ' + hits + ' / ' + N + ' (' + Math.round(top1 * 100) + '%)' +
      '<div class="wread-interp">' + msg + '</div>';
  }

  left.draw = drawAll; right.draw = drawAll;
  ready = true;
  tr.show(60);
  left.request(); right.request();
  return { destroy: function () { tr.pause(); } };
});
