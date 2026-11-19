/* ══════════════════════════════════════════════════════════════════
   Figure 8 — Low-Rank Adaptation (LoRA): steering frozen transformer
   trajectories through a rank-r bottleneck
   Melds directly with Figure 7's LLM-Explainer motifs (subword token
   chips + 3D unit-sphere vector readout):
   - Left: Subword [Tokens] -> pooled vector x -> parallel split into
     frozen pretrained matrix W₀ (d×d, 0 gradients) and trainable LoRA
     hourglass B·A (d×r and r×d), summing at ⊕ into h' = W₀x + (α/r)BAx.
   - Right: On the 3D unit sphere, B = 0 at step 0 leaves h' identical
     to the frozen pretrained vector h₀; training only B and A grows a
     low-rank steering vector Δh toward the task readout pole using a
     tiny fraction of parameters (2dr / d²).
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('lora_adapter', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 8,
    title: 'Low-Rank Adaptation (LoRA): steering a frozen model through a narrow bottleneck',
    hint: 'Press play or scrub training from zeroed start to adapted model &middot; slide Bottleneck rank to widen or pinch the side-channel &middot; switch Headline or drag the 3D sphere'
  });

  var D_MODEL = 768;
  var RANKS = [1, 2, 4, 8, 16, 32];
  var rank = cfg.rank || 8;
  var alpha = cfg.alpha || 16;
  var prog = 1.0; // 0..1 training progress (step 0 has B = 0 -> ΔW = 0!)
  var caseIdx = 0;
  var yaw = -0.26, pitch = 0.22, dragOrigin = null;
  var ready = false;

  var CASES = [
    {
      short: 'Grad student\u2026',
      headline: 'Local graduate student weeps on keyboard, again',
      tokens: ['Local', ' grad', ' student', ' weeps', ' on', ' key', 'board'],
      ids: [15883, 9642, 3076, 26139, 2006, 3145, 6977],
      label: 1, // sarcastic
      h0: M.normVec([0.08, 0.28, 0.96]),
      deltaFull: [0.88, 0.16, -0.34]
    },
    {
      short: '5 simple ways\u2026',
      headline: '5 simple ways to organize your family kitchen',
      tokens: ['5', ' simple', ' ways', ' to', ' organ', 'ize', ' kitchen'],
      ids: [1019, 3722, 3971, 2000, 4986, 4697, 3829],
      label: 0, // straight
      h0: M.normVec([-0.06, -0.18, 0.98]),
      deltaFull: [-0.86, -0.20, -0.32]
    },
    {
      short: 'Area man\u2026',
      headline: 'Area man passionate defender of what he imagines constitution to be',
      tokens: ['Area', ' man', ' passion', 'ate', ' defend', 'er', ' constitution'],
      ids: [2181, 2158, 6406, 3686, 6985, 2121, 4552],
      label: 1, // sarcastic
      h0: M.normVec([-0.04, 0.42, 0.91]),
      deltaFull: [0.92, -0.06, -0.30]
    }
  ];

  var POLES = [
    M.normVec([-0.82, -0.22, 0.53]), // [straight]
    M.normVec([0.82, 0.24, 0.52])    // [sarcastic]
  ];

  /* Background corpus cloud on sphere (showing frozen ambiguity vs task poles) */
  var bgCloud = [];
  (function makeCloud() {
    var rng = M.rng(404);
    for (var i = 0; i < 34; i++) {
      var cls = i % 2;
      var c = POLES[cls];
      bgCloud.push({
        cls: cls,
        v: M.normVec([c[0] * 0.72 + rng.normal(0, 0.34), c[1] * 0.72 + rng.normal(0, 0.38), 0.58 + rng.normal(0, 0.28)])
      });
    }
  })();

  /* Rank capacity factor: r=1 captures ~58% of ideal steering direction + slight off-axis residual,
     r=4 captures ~88%, r>=8 captures ~98-100%, illustrating why small r suffices for task steering! */
  function adaptedState(c) {
    var rCap = 1 - 0.46 * Math.exp(-(rank - 1) / 2.6);
    var scaleFactor = M.clamp((alpha / 16) * rCap, 0.15, 1.85) * prog;
    var sideDrift = (1 - rCap) * 0.32 * prog * (c.label ? -1 : 1);
    var raw = [
      c.h0[0] + scaleFactor * c.deltaFull[0],
      c.h0[1] + scaleFactor * c.deltaFull[1] + sideDrift,
      c.h0[2] + scaleFactor * c.deltaFull[2]
    ];
    var hAdapt = M.normVec(raw);
    var logitStraight = 2.9 * M.cosine(hAdapt, POLES[0]);
    var logitSarc = 2.9 * M.cosine(hAdapt, POLES[1]);
    var probs = M.softmax([logitStraight, logitSarc], 1.0);
    return { h0: c.h0, hAdapt: hAdapt, probs: probs, scaleFactor: scaleFactor };
  }

  function slerp(a, b, t) {
    return M.normVec([M.lerp(a[0], b[0], t), M.lerp(a[1], b[1], t), M.lerp(a[2], b[2], t)]);
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.14);
  M.mini(split.a, '[tokens] \u2192 frozen main weights + narrow LoRA side-channel');
  M.mini(split.b, 'unit sphere: frozen vector + LoRA nudge \u2192 prediction');
  var hf = function (w) { return M.clamp(w * 0.80, 265, 345); };
  var left = M.stage(split.a, { height: hf, ref: 430, label: 'Subword tokens flowing through frozen pretrained weights in parallel with a narrow trainable LoRA bottleneck' });
  var right = M.stage(split.b, { height: function () { return left.h || 285; }, ref: 380, label: '3D unit sphere showing frozen embedding vector steered by a small LoRA nudge toward straight or sarcastic pole' });

  M.legend(root, [
    { color: C.blue, label: 'frozen pretrained path / [straight]' },
    { color: C.orange, label: 'trainable LoRA side-channel / [sarcastic]' },
    { color: C.ink, label: 'adapted vector (frozen + nudge)', line: true }
  ]);

  var tr = M.transport(root, {
    n: 61, fps: 30, aria: 'LoRA finetuning steps from zeroed start to adapted readout',
    fmt: function (k) {
      return k === 0 ? 'step 0 (no nudge)' : ('step ' + k + ' / 60');
    },
    onFrame: function (k) { prog = M.ease(k / 60); drawAll(); }
  });
  tr.show(60);

  var row1 = M.row(root);
  var caseT = M.toggle(row1, {
    label: 'Headline', value: '0',
    options: CASES.map(function (c, i) { return [String(i), c.short]; }),
    onChange: function (v) { caseIdx = +v; drawAll(); }
  });

  var row2 = M.row(root);
  var rankIdxInit = Math.max(0, RANKS.indexOf(rank));
  var rankS = M.slider(row2, {
    label: 'Bottleneck rank', min: 0, max: RANKS.length - 1, step: 1, value: rankIdxInit, accent: 'orange',
    fmt: function (v) { return 'rank ' + RANKS[Math.round(v)]; },
    onInput: function (v) { rank = RANKS[Math.round(v)]; drawAll(); }
  });
  var alphaS = M.slider(row2, {
    label: 'Nudge strength', min: 4, max: 32, step: 4, value: alpha, accent: 'blue',
    fmt: function (v) { return Math.round(v) + ''; },
    onInput: function (v) { alpha = Math.round(v); drawAll(); }
  });
  M.spacer(row2);
  M.resetButton(row2, function () {
    tr.pause();
    caseIdx = 0; rank = 8; alpha = 16; yaw = -0.26; pitch = 0.22;
    caseT.set('0', true);
    rankS.set(RANKS.indexOf(8), true);
    alphaS.set(16, true);
    tr.show(60); drawAll();
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

  /* ─── Draw Left: Token Strip + Parallel Frozen W₀ & LoRA Hourglass B·A ─── */
  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    M.clear(ctx, s);
    var c = CASES[caseIdx], toks = c.tokens, n = toks.length;
    var padX = 12 * s.sc, usableW = s.w - 2 * padX;
    var stepW = usableW / n;

    /* 1. Top strip: subword [Token] chips + [IDs] + 4-dot vectors (echoing Figure 7) */
    var yTok = 20 * s.sc, yId = 39 * s.sc, yVec = 52 * s.sc;
    ctx.textAlign = 'center';
    for (var i = 0; i < n; i++) {
      var tx = padX + (i + 0.5) * stepW;
      var col = M.catColor(i);
      var chipW = Math.max(22 * s.sc, stepW - 4 * s.sc);
      ctx.fillStyle = M.rgba(col, 0.13);
      ctx.strokeStyle = M.rgba(col, 0.70);
      ctx.lineWidth = 1.05 * s.sc;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(tx - chipW / 2, yTok - 9 * s.sc, chipW, 18 * s.sc, 4 * s.sc);
      else ctx.rect(tx - chipW / 2, yTok - 9 * s.sc, chipW, 18 * s.sc);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.3, 600, 'sans');
      ctx.textBaseline = 'middle';
      var lbl = toks[i].trim();
      if (lbl.length > 7) lbl = lbl.slice(0, 6) + '\u2026';
      ctx.fillText(lbl, tx, yTok);

      ctx.fillStyle = C.muted; ctx.font = M.font(s, 7.2);
      ctx.fillText('[' + c.ids[i] + ']', tx, yId);

      for (var d = -1.5; d <= 1.5; d += 1) {
        M.dot(ctx, tx + d * 3.8 * s.sc, yVec, 1.45 * s.sc, col);
      }
    }

    /* Converge token vectors into pooled hidden input x (d = 768) */
    var midX = s.w * 0.5, yIn = 74 * s.sc;
    ctx.strokeStyle = M.rgba(C.muted, 0.32);
    ctx.lineWidth = 0.9 * s.sc;
    for (var j = 0; j < n; j++) {
      var sx = padX + (j + 0.5) * stepW;
      ctx.beginPath();
      ctx.moveTo(sx, yVec + 4 * s.sc);
      ctx.quadraticCurveTo(sx, yIn - 4 * s.sc, midX, yIn);
      ctx.stroke();
    }
    M.pill(ctx, s, 'pooled token state (768 numbers)', midX, yIn, { align: 'center', color: C.ink, size: 7.9 });

    /* 2. Parallel branches: Left = Frozen Pretrained Weights; Right = LoRA Hourglass */
    var leftCX = s.w * 0.27, rightCX = s.w * 0.73;
    var boxTop = 98 * s.sc, boxBot = s.h - 52 * s.sc, boxH = boxBot - boxTop;
    var yMerge = s.h - 22 * s.sc;

    /* Split wires from x to both branches */
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.5 * s.sc;
    ctx.beginPath(); ctx.moveTo(midX - 24 * s.sc, yIn + 7 * s.sc); ctx.lineTo(leftCX, boxTop); ctx.stroke();
    ctx.strokeStyle = C.orange; ctx.lineWidth = 1.5 * s.sc;
    ctx.beginPath(); ctx.moveTo(midX + 24 * s.sc, yIn + 7 * s.sc); ctx.lineTo(rightCX, boxTop); ctx.stroke();

    /* Left Branch: Frozen Pretrained Weight Matrix (768 × 768) */
    var w0W = Math.min(132 * s.sc, s.w * 0.38);
    var w0X = leftCX - w0W / 2;
    ctx.fillStyle = M.rgba(C.blue, 0.08);
    ctx.strokeStyle = M.rgba(C.blue, 0.78);
    ctx.lineWidth = 1.4 * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(w0X, boxTop, w0W, boxH, 5 * s.sc);
    else ctx.rect(w0X, boxTop, w0W, boxH);
    ctx.fill(); ctx.stroke();

    /* Subtle frozen weight grid */
    var gRows = 6, gCols = 6;
    var gx0 = w0X + 12 * s.sc, gy0 = boxTop + 26 * s.sc;
    var gw = w0W - 24 * s.sc, gh = Math.max(24 * s.sc, boxH - 52 * s.sc);
    var cw = gw / gCols, ch = gh / gRows;
    for (var gr = 0; gr < gRows; gr++) {
      for (var gc = 0; gc < gCols; gc++) {
        var shade = ((gr * 7 + gc * 5) % 5) / 5;
        ctx.fillStyle = M.rgba(C.blue, 0.10 + 0.22 * shade);
        ctx.fillRect(gx0 + gc * cw + 1, gy0 + gr * ch + 1, cw - 2, ch - 2);
      }
    }
    ctx.fillStyle = C.blue; ctx.font = M.font(s, 8.2, 600, 'sans');
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('Pretrained Weights (768\u00d7768)', leftCX, boxTop + 6 * s.sc);
    M.pill(ctx, s, 'FROZEN \u00b7 589,824 weights', leftCX, boxBot - 12 * s.sc, { align: 'center', color: C.blue, size: 7.4 });

    /* Right Branch: Trainable LoRA Bottleneck Hourglass */
    var topW = Math.min(126 * s.sc, s.w * 0.36);
    var waistFrac = 0.16 + 0.46 * (Math.log2(rank) / 5);
    var waistW = topW * waistFrac;
    var midY = (boxTop + boxBot) * 0.5;
    var gapY = 10 * s.sc;

    /* Down-projection trapezoid (768 -> rank) */
    ctx.fillStyle = M.rgba(C.orange, 0.15);
    ctx.strokeStyle = C.orange;
    ctx.lineWidth = 1.4 * s.sc;
    ctx.beginPath();
    ctx.moveTo(rightCX - topW / 2, boxTop);
    ctx.lineTo(rightCX + topW / 2, boxTop);
    ctx.lineTo(rightCX + waistW / 2, midY - gapY);
    ctx.lineTo(rightCX - waistW / 2, midY - gapY);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = C.ink; ctx.font = M.font(s, 7.9, 600, 'sans');
    ctx.textBaseline = 'middle';
    ctx.fillText('Compress (768 \u2192 ' + rank + ')', rightCX, (boxTop + midY - gapY) * 0.5);

    /* Bottleneck nodes along the waist */
    var nDots = Math.min(rank, 8);
    for (var ri = 0; ri < nDots; ri++) {
      var rx = nDots === 1 ? rightCX : (rightCX - waistW * 0.42 + (ri / (nDots - 1)) * waistW * 0.84);
      M.dot(ctx, rx, midY, 3.1 * s.sc, C.orange, C.surface, 1.0 * s.sc);
    }
    M.pill(ctx, s, 'rank ' + rank, rightCX + waistW * 0.5 + 26 * s.sc, midY, { align: 'center', color: C.orange, size: 7.4 });

    /* Up-projection trapezoid (rank -> 768): faint at step 0 (starts zeroed!), lights up with prog */
    ctx.fillStyle = M.rgba(C.orange, 0.04 + 0.22 * prog);
    ctx.strokeStyle = M.rgba(C.orange, 0.35 + 0.65 * prog);
    ctx.lineWidth = 1.4 * s.sc;
    if (prog < 0.05) ctx.setLineDash([3 * s.sc, 2 * s.sc]);
    ctx.beginPath();
    ctx.moveTo(rightCX - waistW / 2, midY + gapY);
    ctx.lineTo(rightCX + waistW / 2, midY + gapY);
    ctx.lineTo(rightCX + topW / 2, boxBot);
    ctx.lineTo(rightCX - topW / 2, boxBot);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = prog < 0.05 ? C.muted : C.ink;
    ctx.font = M.font(s, 7.9, 600, 'sans');
    ctx.fillText(prog < 0.05 ? 'starts at zero!' : ('Expand (' + rank + ' \u2192 768)'), rightCX, (midY + gapY + boxBot) * 0.5);

    /* Merge wires into + node */
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.6 * s.sc;
    ctx.beginPath(); ctx.moveTo(leftCX, boxBot); ctx.lineTo(midX - 11 * s.sc, yMerge); ctx.stroke();
    ctx.strokeStyle = M.rgba(C.orange, 0.25 + 0.75 * prog); ctx.lineWidth = (1.1 + 1.4 * prog) * s.sc;
    ctx.beginPath(); ctx.moveTo(rightCX, boxBot); ctx.lineTo(midX + 11 * s.sc, yMerge); ctx.stroke();

    M.dot(ctx, midX, yMerge, 10 * s.sc, C.surface, C.ink, 1.5 * s.sc);
    ctx.fillStyle = C.ink; ctx.font = M.font(s, 11, 700, 'sans');
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('+', midX, yMerge + 0.5);
    M.pill(ctx, s, 'frozen output + LoRA nudge', midX + 76 * s.sc, yMerge, { align: 'center', color: C.ink, size: 7.7 });
  }

  /* ─── Draw Right: 3D Unit Sphere Trajectory + Param Budget & Task Readout ─── */
  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);
    var c = CASES[caseIdx];
    var st = adaptedState(c);

    var R = Math.min(s.w * 0.29, s.h * 0.36);
    var cx = R + 18 * s.sc, cy = s.h * 0.50;
    M.drawSphere(ctx, cx, cy, R, yaw, pitch, s);

    /* Background headline cloud */
    for (var i = 0; i < bgCloud.length; i++) {
      var bp = M.sphereProject(bgCloud[i].v, cx, cy, R, yaw, pitch);
      var bCol = bgCloud[i].cls ? C.orange : C.blue;
      M.dot(ctx, bp.x, bp.y, (bp.z >= 0 ? 2.6 : 1.9) * s.sc, M.rgba(bCol, bp.z >= 0 ? 0.42 : 0.14));
    }

    /* Task readout poles: [straight] & [sarcastic] */
    [['[straight]', POLES[0], C.blue], ['[sarcastic]', POLES[1], C.orange]].forEach(function (pole) {
      var pp = M.sphereProject(pole[1], cx, cy, R, yaw, pitch);
      M.star(ctx, pp.x, pp.y, 5.8 * s.sc, pole[2], C.surface);
      M.pill(ctx, s, pole[0], pp.x, pp.y - 13 * s.sc, { align: 'center', color: pole[2], size: 7.8 });
    });

    /* Frozen pretrained vector (dashed slate-blue arrow) */
    var p0 = M.sphereProject(st.h0, cx, cy, R, yaw, pitch);
    ctx.save();
    ctx.setLineDash([4 * s.sc, 3 * s.sc]);
    M.arrow(ctx, cx, cy, p0.x, p0.y, M.rgba(C.blue, 0.72), 1.8 * s.sc, 6.5 * s.sc);
    ctx.restore();
    M.pill(ctx, s, 'frozen base', p0.x, p0.y + 12 * s.sc, { align: 'center', color: C.blue, size: 7.4 });

    /* Geodesic steering arc from frozen base to adapted vector */
    if (prog > 0.02) {
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.2 * s.sc;
      ctx.beginPath();
      for (var k = 0; k <= 20; k++) {
        var sp = M.sphereProject(slerp(st.h0, st.hAdapt, k / 20), cx, cy, R, yaw, pitch);
        if (k === 0) ctx.moveTo(sp.x, sp.y); else ctx.lineTo(sp.x, sp.y);
      }
      ctx.stroke();
      var pMid = M.sphereProject(slerp(st.h0, st.hAdapt, 0.52), cx, cy, R, yaw, pitch);
      M.pill(ctx, s, '+LoRA nudge', pMid.x, pMid.y - 11 * s.sc, { align: 'center', color: C.orange, size: 7.3 });
    }
    var pAdapt = M.sphereProject(st.hAdapt, cx, cy, R, yaw, pitch);
    M.arrow(ctx, cx, cy, pAdapt.x, pAdapt.y, C.ink, 2.5 * s.sc, 8 * s.sc);

    /* Right-hand column: 1) Weight Budget, 2) Class Probability Readout */
    var bx0 = cx + R + 16 * s.sc, bwMax = Math.max(54 * s.sc, s.w - bx0 - 12 * s.sc);
    var frozenParams = D_MODEL * D_MODEL;
    var loraParams = 2 * D_MODEL * rank;
    var share = loraParams / (frozenParams + loraParams);

    ctx.fillStyle = C.muted; ctx.font = M.font(s, 8.4, 600);
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('WEIGHT BUDGET', bx0, 22 * s.sc);

    var pyBar = 40 * s.sc, barH = 14 * s.sc;
    ctx.fillStyle = M.rgba(C.blue, 0.22);
    ctx.fillRect(bx0, pyBar, bwMax, barH);
    var loraW = Math.max(3 * s.sc, bwMax * Math.min(1, share * 4.5));
    ctx.fillStyle = C.orange;
    ctx.fillRect(bx0, pyBar, loraW, barH);
    ctx.strokeStyle = C.border; ctx.lineWidth = 1;
    ctx.strokeRect(bx0, pyBar, bwMax, barH);

    ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.2, 600);
    ctx.fillText('LoRA side-channel: ' + (loraParams / 1000).toFixed(1) + 'k', bx0, pyBar + barH + 6 * s.sc);
    ctx.fillStyle = C.muted; ctx.font = M.font(s, 7.8);
    ctx.fillText('Frozen: 589.8k (' + (share * 100).toFixed(2) + '% trained)', bx0, pyBar + barH + 19 * s.sc);

    /* Task probabilities */
    var probTop = s.h * 0.54;
    ctx.fillStyle = C.muted; ctx.font = M.font(s, 8.4, 600);
    ctx.fillText('PREDICTED CLASS', bx0, probTop);

    var classes = [['[sarcastic]', st.probs[1], C.orange], ['[straight]', st.probs[0], C.blue]];
    for (var ci = 0; ci < 2; ci++) {
      var ry = probTop + 22 * s.sc + ci * 34 * s.sc;
      var pVal = classes[ci][1], colC = classes[ci][2];
      ctx.fillStyle = M.rgba(colC, 0.18);
      ctx.fillRect(bx0, ry, bwMax, 14 * s.sc);
      ctx.fillStyle = colC;
      ctx.fillRect(bx0, ry, bwMax * pVal, 14 * s.sc);
      ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.4, 600);
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(classes[ci][0], bx0, ry - 2 * s.sc);
      ctx.textAlign = 'right';
      ctx.fillText(Math.round(pVal * 100) + '%', bx0 + bwMax, ry - 2 * s.sc);
    }
  }

  function drawAll() {
    if (!ready) return;
    drawLeft(); drawRight();
    var c = CASES[caseIdx];
    var st = adaptedState(c);
    var frozenParams = D_MODEL * D_MODEL, loraParams = 2 * D_MODEL * rank;
    var pct = ((loraParams / (frozenParams + loraParams)) * 100).toFixed(2);
    var msg = prog < 0.04
      ? '<span class="b">At step 0, the LoRA side-channel starts at zero, so predictions match the untouched pretrained model.</span>'
      : '<span class="g">Adjusting just ' + loraParams.toLocaleString() + ' side-channel weights (' + pct + '% of the layer) steers the representation toward [' + (c.label ? 'sarcastic' : 'straight') + '].</span>';
    read.innerHTML = '<span class="k">bottleneck</span> rank ' + rank +
      '<span class="sep">&middot;</span><span class="k">trainable weights</span> ' + loraParams.toLocaleString() + ' / ' + (frozenParams + loraParams).toLocaleString() + ' (' + pct + '%)' +
      '<span class="sep">&middot;</span><span class="k">P(sarcastic)</span> ' + Math.round(st.probs[1] * 100) + '%' +
      '<div class="wread-interp">' + msg + '</div>';
  }

  left.draw = drawAll; right.draw = drawAll;
  ready = true;
  tr.show(60);
  left.request(); right.request();
  return { destroy: function () { tr.pause(); } };
});
