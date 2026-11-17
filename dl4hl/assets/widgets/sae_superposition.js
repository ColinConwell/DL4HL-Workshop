/* ══════════════════════════════════════════════════════════════════
   Figure 7 — Polysemantic superposition & Sparse Autoencoders (SAEs)
   Networks pack many sparse concepts into fewer neurons (superposition),
   making individual neurons polysemantic. A TopK Sparse Autoencoder
   expands dense activations into a wider, sparsely active dictionary
   whose individual latents track coherent concepts.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('sae_superposition', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 7,
    title: 'Polysemantic neurons vs. monosemantic Sparse Autoencoder (SAE) latents',
    hint: 'Click any concept spoke on the pentagon (or mix two) &middot; explore discovered SAE headline latents on the right'
  });

  var CONCEPTS = ['Everyday irony', 'Campaign politics', 'Severe weather', 'Global diplomacy', 'Health & family'];
  var FALLBACK_FEATURES = [
    { id: 46, firing_rate: 0.126, sarcastic_share: 1.0, top_headlines: [
      { text: 'couple just wants small ceremony in public park with close friends and shirtless stranger', act: 3.4, sarcastic: 1 },
      { text: 'area woman fulfills dream of becoming writer by getting job at bookstore', act: 3.1, sarcastic: 1 },
      { text: 'man still worried parents of ex-girlfriend from 7 years ago hate him', act: 2.9, sarcastic: 1 }
    ]},
    { id: 32, firing_rate: 0.163, sarcastic_share: 0.4, top_headlines: [
      { text: 'gingrich urges romney to drop out so he can focus on general election', act: 3.2, sarcastic: 1 },
      { text: 'cruz likely to block trump on a second ballot at gop convention', act: 2.8, sarcastic: 0 },
      { text: 'senate panel advances bipartisan campaign finance disclosure bill', act: 2.5, sarcastic: 0 }
    ]},
    { id: 43, firing_rate: 0.054, sarcastic_share: 0.2, top_headlines: [
      { text: 'multiple tornadoes rip through oklahoma, injuring 7', act: 3.6, sarcastic: 0 },
      { text: 'fema officials panic after accidentally evacuating 1 million residents', act: 3.0, sarcastic: 1 },
      { text: 'death toll from powerful mexico earthquake rises to 91', act: 2.7, sarcastic: 0 }
    ]},
    { id: 22, firing_rate: 0.082, sarcastic_share: 0.2, top_headlines: [
      { text: 'syria ceasefire, backed by russia and turkey, holds after initial clashes', act: 3.3, sarcastic: 0 },
      { text: 'u.s. to slow down relationship with uruguay', act: 2.9, sarcastic: 0 },
      { text: 'security council resolution sparks diplomatic debate at u.n.', act: 2.6, sarcastic: 0 }
    ]},
    { id: 14, firing_rate: 0.098, sarcastic_share: 0.6, top_headlines: [
      { text: 'pediatric sleep study links bedtime routines to calmer mornings', act: 3.1, sarcastic: 0 },
      { text: 'parents confident new baby will fix everything else in marriage', act: 2.9, sarcastic: 1 },
      { text: 'local dad resting eyes for just five minutes on living room rug', act: 2.7, sarcastic: 1 }
    ]}
  ];

  var realFeats = (cfg.data && cfg.data.features && cfg.data.features.length) ? cfg.data.features : FALLBACK_FEATURES;
  var activeIdx = 0;
  var mixSecond = -1;
  var topK = cfg.top_k || 1;

  /* 5 unit vectors arranged in a 2D pentagon (classic Elhage/Fel superposition geometry) */
  var DICT = [0, 1, 2, 3, 4].map(function (i) {
    var a = -Math.PI / 2 + (2 * Math.PI * i) / 5;
    return [Math.cos(a), Math.sin(a)];
  });

  function denseHidden() {
    var h = [DICT[activeIdx][0], DICT[activeIdx][1]];
    if (mixSecond >= 0 && mixSecond !== activeIdx) {
      h[0] += 0.72 * DICT[mixSecond][0];
      h[1] += 0.72 * DICT[mixSecond][1];
    }
    return h;
  }

  function saeDecode(h) {
    var pre = DICT.map(function (d) { return Math.max(0, h[0] * d[0] + h[1] * d[1] - 0.18); });
    var order = M.range(5).sort(function (a, b) { return pre[b] - pre[a]; });
    var codes = [0, 0, 0, 0, 0];
    for (var r = 0; r < topK; r++) codes[order[r]] = pre[order[r]];
    return { pre: pre, codes: codes };
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, '2 polysemantic neurons packing 5 concepts (superposition)');
  M.mini(split.b, 'overcomplete TopK SAE latents + top-activating texts');
  var hf = function (w) { return M.clamp(w * 0.80, 235, 315); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: '2D pentagon showing 5 sparse concepts superimposed inside 2 hidden neurons' });
  var right = M.stage(split.b, { height: function () { return left.h || 255; }, ref: 400, label: 'Sparse autoencoder TopK dictionary activations and top-activating headlines' });

  var row = M.row(root);
  var featOpts = CONCEPTS.slice(0, Math.min(5, realFeats.length)).map(function (nm, idx) {
    return [String(idx), 'F' + (idx + 1)];
  });
  var featT = M.toggle(row, {
    label: 'Concept', value: '0', options: featOpts,
    onChange: function (v) { activeIdx = +v; drawAll(); }
  });
  var mixT = M.toggle(row, {
    label: 'Input sparsity', value: 'single',
    options: [['single', '1 active concept'], ['pair', '2 co-active']],
    onChange: function (v) { mixSecond = v === 'pair' ? (activeIdx + 2) % 5 : -1; drawAll(); }
  });
  var kS = M.slider(row, {
    label: 'SAE Top-K', min: 1, max: 3, step: 1, value: topK, accent: 'blue',
    onInput: function (v) { topK = v; drawAll(); }
  });
  M.spacer(row);
  M.resetButton(row, function () {
    activeIdx = 0; mixSecond = -1; topK = 1;
    featT.set('0', true); mixT.set('single', true); kS.set(1, true);
    drawAll();
  });
  var read = M.readout(root);

  var vL = M.view(left, { x: [-1.45, 1.45], y: [-1.45, 1.45], equal: true, pad: { l: 34, r: 14, t: 14, b: 32 } });

  M.pointer(left.cv, {
    down: function (p) {
      vL.update();
      var best = 0, bestD = Infinity;
      for (var i = 0; i < 5; i++) {
        var dx = p.x - vL.px(DICT[i][0]), dy = p.y - vL.py(DICT[i][1]);
        var d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      }
      activeIdx = best;
      if (mixT.get() === 'pair') mixSecond = (activeIdx + 2) % 5;
      featT.set(String(Math.min(best, featOpts.length - 1)), true);
      drawAll();
      return false;
    }
  });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain([-1.45, 1.45], [-1.45, 1.45]);
    M.clear(ctx, s);
    M.axes(ctx, vL, s, { xlabel: 'polysemantic neuron h\u2081', ylabel: 'polysemantic neuron h\u2082', nx: 3, ny: 3 });

    ctx.save(); vL.clip(ctx);
    /* Unit circle & 5 dictionary directions */
    ctx.strokeStyle = M.rgba(C.muted, 0.35);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(vL.px(0), vL.py(0), Math.abs(vL.px(1) - vL.px(0)), 0, Math.PI * 2);
    ctx.stroke();

    for (var i = 0; i < 5; i++) {
      var col = M.catColor(i);
      var isAct = i === activeIdx || i === mixSecond;
      M.arrow(ctx, vL.px(0), vL.py(0), vL.px(DICT[i][0]), vL.py(DICT[i][1]),
        M.rgba(col, isAct ? 0.95 : 0.42), (isAct ? 2.4 : 1.3) * s.sc, 6.5 * s.sc);
      ctx.fillStyle = isAct ? col : C.muted;
      ctx.font = M.font(s, 8.5, isAct ? 600 : 400, 'sans');
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('F' + (i + 1), vL.px(DICT[i][0] * 1.22), vL.py(DICT[i][1] * 1.22));
    }

    /* Dense activation vector h */
    var h = denseHidden();
    M.dot(ctx, vL.px(h[0]), vL.py(h[1]), 5.2 * s.sc, C.ink, C.surface, 1.5 * s.sc);
    ctx.restore();
    M.pill(ctx, s, 'Neuron h\u2081=' + M.f(h[0], 2) + ', h\u2082=' + M.f(h[1], 2) + ' (mixes concepts!)', vL.L + 6 * s.sc, vL.T + 12 * s.sc);
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);
    var h = denseHidden();
    var dec = saeDecode(h);
    var padX = 14 * s.sc;

    /* Top half: 5 sparse dictionary activation bars */
    var barTop = 18 * s.sc, barH = 13 * s.sc, barGap = 5 * s.sc;
    var barLeft = padX + 96 * s.sc, barW = s.w - barLeft - 40 * s.sc;

    for (var i = 0; i < 5; i++) {
      var yy = barTop + i * (barH + barGap);
      var val = dec.codes[i];
      ctx.fillStyle = val > 0 ? C.ink : C.muted;
      ctx.font = M.font(s, 8.8, val > 0 ? 600 : 400, 'sans');
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('F' + (i + 1) + ' ' + CONCEPTS[i].split(' ')[0], barLeft - 8 * s.sc, yy + barH / 2);

      ctx.fillStyle = C.codeBg;
      ctx.fillRect(barLeft, yy, barW, barH);
      if (val > 0) {
        ctx.fillStyle = M.catColor(i);
        ctx.fillRect(barLeft, yy, barW * M.clamp(val / 1.15, 0, 1), barH);
      }
      ctx.fillStyle = val > 0 ? C.ink : C.muted;
      ctx.font = M.font(s, 8.5);
      ctx.textAlign = 'left';
      ctx.fillText(M.f(val, 2), barLeft + barW + 5 * s.sc, yy + barH / 2);
    }

    /* Bottom half: Real top-activating headlines for the primary selected SAE latent */
    var fObj = realFeats[activeIdx % realFeats.length];
    var cardY = barTop + 5 * (barH + barGap) + 8 * s.sc;
    var cardH = s.h - cardY - 10 * s.sc;
    ctx.fillStyle = C.surface;
    ctx.strokeStyle = C.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(padX, cardY, s.w - 2 * padX, cardH, 5 * s.sc);
    else ctx.rect(padX, cardY, s.w - 2 * padX, cardH);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = M.catColor(activeIdx);
    ctx.font = M.font(s, 9, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('Discovered DistilBERT SAE Latent #' + fObj.id + ' (fires on ' + Math.round(fObj.firing_rate * 100) + '% of headlines)', padX + 8 * s.sc, cardY + 7 * s.sc);

    var lines = fObj.top_headlines.slice(0, 3);
    var lineStep = Math.max(16 * s.sc, (cardH - 28 * s.sc) / 3);
    ctx.font = M.font(s, 8.4, 400, 'sans');
    for (var li = 0; li < lines.length; li++) {
      var ly = cardY + 25 * s.sc + li * lineStep;
      var txt = lines[li].text;
      var maxChars = Math.max(32, Math.floor((s.w - 2 * padX - 48 * s.sc) / (4.7 * s.sc)));
      if (txt.length > maxChars) txt = txt.slice(0, maxChars - 1) + '\u2026';
      ctx.fillStyle = lines[li].sarcastic ? C.orange : C.blue;
      ctx.fillText(lines[li].sarcastic ? '[sarc]' : '[news]', padX + 8 * s.sc, ly);
      ctx.fillStyle = C.ink;
      ctx.fillText(txt, padX + 44 * s.sc, ly);
    }
  }

  function drawAll() {
    drawLeft(); drawRight();
    var h = denseHidden();
    var dec = saeDecode(h);
    var nActive = dec.codes.filter(function (v) { return v > 0; }).length;
    read.innerHTML = '<span class="k">dense neurons</span> 2 (polysemantic)' +
      '<span class="sep">&middot;</span><span class="k">SAE dictionary</span> 5 directions (Top-' + topK + ' active: ' + nActive + ')' +
      '<span class="sep">&middot;</span><span class="g">widening + enforcing sparsity untangles mixed neuron axes into single-theme latents</span>';
  }

  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return {};
});
