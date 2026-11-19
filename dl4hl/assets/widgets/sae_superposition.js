/* ══════════════════════════════════════════════════════════════════
   Figure 7 — Polysemantic superposition & Sparse Autoencoders (SAEs)
   Networks pack many sparse concepts into fewer neurons (superposition),
   making individual neurons polysemantic (responding across unrelated
   topics). A TopK Sparse Autoencoder expands dense activations into a
   wider, sparsely active dictionary whose latents fire selectively
   (monosemantically) for single coherent concepts.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('sae_superposition', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 9,
    title: 'Polysemantic neurons vs. monosemantic Sparse Autoencoder (SAE) latents',
    hint: 'Press play to sweep across topics and watch dense neurons mix vs. SAE latents isolate each theme &middot; click any concept spoke or toggle 2 co-active concepts'
  });

  var CONCEPTS = ['Everyday irony', 'Campaign politics', 'Severe weather', 'Global diplomacy', 'Health & family'];
  var SHORT_CONCEPTS = ['Irony', 'Politics', 'Weather', 'Diplomacy', 'Family'];
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
  var sweepTimer = 0;

  /* 5 unit vectors arranged in a rotated 2D pentagon so every spoke projects onto BOTH h1 and h2 */
  var DICT = [0, 1, 2, 3, 4].map(function (i) {
    var a = -Math.PI / 2 + 0.28 + (2 * Math.PI * i) / 5;
    return [Math.cos(a), Math.sin(a)];
  });

  function denseHidden() {
    var h = [DICT[activeIdx][0], DICT[activeIdx][1]];
    if (mixSecond >= 0 && mixSecond !== activeIdx) {
      h[0] += 0.78 * DICT[mixSecond][0];
      h[1] += 0.78 * DICT[mixSecond][1];
    }
    return h;
  }

  /* Orthogonal Matching Pursuit / Top-K sparse dictionary recovery on the 5-spoke overcomplete basis */
  function saeDecode(h) {
    var r0 = h[0], r1 = h[1];
    var codes = [0, 0, 0, 0, 0];
    var chosen = [];
    for (var step = 0; step < topK; step++) {
      var bestI = -1, bestDot = 0.22; // threshold suppresses interference sidelobes!
      for (var i = 0; i < 5; i++) {
        if (chosen.indexOf(i) >= 0) continue;
        var dot = r0 * DICT[i][0] + r1 * DICT[i][1];
        if (dot > bestDot) { bestDot = dot; bestI = i; }
      }
      if (bestI < 0) break;
      chosen.push(bestI);
      codes[bestI] = bestDot;
      r0 -= bestDot * DICT[bestI][0];
      r1 -= bestDot * DICT[bestI][1];
    }
    return { codes: codes, chosen: chosen };
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, '2 polysemantic neurons packing 5 concepts (superposition)');
  M.mini(split.b, '5 monosemantic Top-K SAE latents + discovered texts');
  var hf = function (w) { return M.clamp(w * 0.84, 255, 335); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: '2D pentagon and polysemantic neuron tuning strips showing 5 sparse concepts superimposed inside 2 hidden neurons' });
  var right = M.stage(split.b, { height: function () { return left.h || 275; }, ref: 400, label: 'Sparse autoencoder monosemantic dictionary activations and top-activating headlines' });

  var row1 = M.row(root);
  var playBtn = M.button(row1, 'play', function () { loop.on ? pause() : play(); }, {
    cls: 'wplay', role: 'play', aria: 'Sweep across single and paired concepts to compare polysemantic neurons with monosemantic SAE latents'
  });
  var featOpts = CONCEPTS.slice(0, 5).map(function (_, idx) {
    return [String(idx), 'F' + (idx + 1) + ': ' + SHORT_CONCEPTS[idx]];
  });
  var featT = M.toggle(row1, {
    label: 'Concept', value: '0', options: featOpts,
    onChange: function (v) {
      pause();
      activeIdx = +v;
      if (mixT.get() === 'pair') mixSecond = (activeIdx + 2) % 5;
      drawAll();
    }
  });

  var row2 = M.row(root);
  var mixT = M.toggle(row2, {
    label: 'Input sparsity', value: 'single',
    options: [['single', '1 active concept'], ['pair', '2 co-active']],
    onChange: function (v) {
      pause();
      mixSecond = v === 'pair' ? (activeIdx + 2) % 5 : -1;
      if (v === 'pair' && topK < 2) { topK = 2; kS.set(2, true); }
      drawAll();
    }
  });
  var kS = M.slider(row2, {
    label: 'SAE Top-K', min: 1, max: 3, step: 1, value: topK, accent: 'blue',
    onInput: function (v) { pause(); topK = v; drawAll(); }
  });
  M.spacer(row2);
  M.resetButton(row2, function () {
    pause();
    activeIdx = 0; mixSecond = -1; topK = 1;
    featT.set('0', true); mixT.set('single', true); kS.set(1, true);
    drawAll();
  });
  var read = M.readout(root);

  /* Sweep schedule: 5 single concepts followed by 2 co-active pairs */
  var SWEEP = [
    [0, -1, 1], [1, -1, 1], [2, -1, 1], [3, -1, 1], [4, -1, 1],
    [0, 2, 2], [1, 4, 2]
  ];
  var sweepStep = 0;
  var loop = M.loop(function (dt) {
    sweepTimer += dt;
    if (sweepTimer >= 1.15) {
      sweepTimer = 0;
      sweepStep = (sweepStep + 1) % SWEEP.length;
      var st = SWEEP[sweepStep];
      activeIdx = st[0]; mixSecond = st[1]; topK = st[2];
      featT.set(String(activeIdx), true);
      mixT.set(mixSecond >= 0 ? 'pair' : 'single', true);
      kS.set(topK, true);
      drawAll();
    }
    return true;
  });
  function play() { sweepTimer = 0; loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  var vL = M.view(left, { x: [-1.48, 1.48], y: [-1.48, 1.48], equal: true, pad: { l: 34, r: 14, t: 12, b: 84 } });

  M.pointer(left.cv, {
    down: function (p) {
      pause();
      vL.update();
      if (p.y > vL.B + 12 * left.sc) return false;
      var best = 0, bestD = Infinity;
      for (var i = 0; i < 5; i++) {
        var dx = p.x - vL.px(DICT[i][0]), dy = p.y - vL.py(DICT[i][1]);
        var d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      }
      activeIdx = best;
      if (mixT.get() === 'pair') mixSecond = (activeIdx + 2) % 5;
      featT.set(String(best), true);
      drawAll();
      return false;
    }
  });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain([-1.48, 1.48], [-1.48, 1.48]);
    M.clear(ctx, s);
    M.axes(ctx, vL, s, { xlabel: 'neuron h\u2081', ylabel: 'neuron h\u2082', nx: 3, ny: 3 });

    var h = denseHidden();
    ctx.save(); vL.clip(ctx);
    ctx.strokeStyle = M.rgba(C.muted, 0.32);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(vL.px(0), vL.py(0), Math.abs(vL.px(1) - vL.px(0)), 0, Math.PI * 2);
    ctx.stroke();

    /* Projection dashed drop-lines onto h1 and h2 axes */
    ctx.setLineDash([3 * s.sc, 3 * s.sc]);
    ctx.strokeStyle = M.rgba(C.ink, 0.35);
    ctx.beginPath();
    ctx.moveTo(vL.px(h[0]), vL.py(h[1])); ctx.lineTo(vL.px(h[0]), vL.py(0));
    ctx.moveTo(vL.px(h[0]), vL.py(h[1])); ctx.lineTo(vL.px(0), vL.py(h[1]));
    ctx.stroke();
    ctx.setLineDash([]);

    for (var i = 0; i < 5; i++) {
      var col = M.catColor(i);
      var isAct = i === activeIdx || i === mixSecond;
      M.arrow(ctx, vL.px(0), vL.py(0), vL.px(DICT[i][0]), vL.py(DICT[i][1]),
        M.rgba(col, isAct ? 0.96 : 0.35), (isAct ? 2.5 : 1.25) * s.sc, 6.5 * s.sc);
      ctx.fillStyle = isAct ? col : C.muted;
      ctx.font = M.font(s, 8.3, isAct ? 600 : 400, 'sans');
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('F' + (i + 1), vL.px(DICT[i][0] * 1.24), vL.py(DICT[i][1] * 1.24));
    }

    M.dot(ctx, vL.px(h[0]), vL.py(h[1]), 5.2 * s.sc, C.ink, C.surface, 1.5 * s.sc);
    ctx.restore();

    /* Bottom polysemantic tuning summary: each dense neuron responds to 4-5 unrelated topics! */
    var stripY = s.h - 48 * s.sc, padX = 12 * s.sc;
    ctx.fillStyle = C.muted; ctx.font = M.font(s, 8.0, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('Polysemantic tuning: each neuron fires across multiple unrelated concepts', padX, stripY - 3 * s.sc);

    for (var nIdx = 0; nIdx < 2; nIdx++) {
      var ry = stripY + nIdx * 20 * s.sc;
      ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.3, 600);
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText('h' + (nIdx + 1) + '=' + M.f(h[nIdx], 2), padX, ry + 7 * s.sc);

      var segX0 = padX + 54 * s.sc, segW = (s.w - segX0 - padX - 16 * s.sc) / 5;
      for (var cIdx = 0; cIdx < 5; cIdx++) {
        var loading = Math.abs(DICT[cIdx][nIdx]); // how strongly concept cIdx drives neuron nIdx
        var isLive = cIdx === activeIdx || cIdx === mixSecond;
        var bx = segX0 + cIdx * (segW + 4 * s.sc);
        ctx.fillStyle = M.rgba(M.catColor(cIdx), 0.15 + 0.65 * loading);
        ctx.fillRect(bx, ry + 1 * s.sc, segW, 12 * s.sc);
        if (isLive) {
          ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4 * s.sc;
          ctx.strokeRect(bx, ry + 1 * s.sc, segW, 12 * s.sc);
        }
        ctx.fillStyle = C.ink; ctx.font = M.font(s, 7.3, isLive ? 600 : 400, 'sans');
        ctx.textAlign = 'center';
        ctx.fillText('F' + (cIdx + 1), bx + segW / 2, ry + 7 * s.sc);
      }
    }
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);
    var h = denseHidden();
    var dec = saeDecode(h);
    var padX = 12 * s.sc;

    /* Header contrasting monosemantic 1-to-1 selectivity */
    ctx.fillStyle = C.green; ctx.font = M.font(s, 8.2, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('Monosemantic SAE dictionary: 1 latent per concept (others = 0)', padX, 7 * s.sc);

    /* Top half: 5 sparse dictionary activation bars */
    var barTop = 24 * s.sc, barH = 14 * s.sc, barGap = 5 * s.sc;
    var barLeft = padX + 86 * s.sc, barW = s.w - barLeft - 62 * s.sc;

    for (var i = 0; i < 5; i++) {
      var yy = barTop + i * (barH + barGap);
      var val = dec.codes[i];
      var isTruth = (i === activeIdx || i === mixSecond);
      ctx.fillStyle = val > 0 ? C.ink : C.muted;
      ctx.font = M.font(s, 8.4, val > 0 ? 600 : 400, 'sans');
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('L' + (i + 1) + ' \u00b7 ' + SHORT_CONCEPTS[i], barLeft - 7 * s.sc, yy + barH / 2);

      ctx.fillStyle = C.codeBg;
      ctx.fillRect(barLeft, yy, barW, barH);
      if (val > 0) {
        ctx.fillStyle = M.catColor(i);
        ctx.fillRect(barLeft, yy, barW * M.clamp(val / 1.15, 0, 1), barH);
      }
      ctx.fillStyle = val > 0 ? C.ink : C.muted;
      ctx.font = M.font(s, 8.0, val > 0 ? 600 : 400);
      ctx.textAlign = 'left';
      var statusTxt = val > 0 ? (M.f(val, 2) + ' \u2713') : (isTruth ? '0.00 (k<' + (mixSecond >= 0 ? 2 : 1) + ')' : '0.00');
      ctx.fillText(statusTxt, barLeft + barW + 5 * s.sc, yy + barH / 2);
    }

    /* Bottom half: Real top-activating headlines for the primary selected SAE latent */
    var fObj = realFeats[activeIdx % realFeats.length];
    var cardY = barTop + 5 * (barH + barGap) + 6 * s.sc;
    var cardH = s.h - cardY - 8 * s.sc;
    ctx.fillStyle = C.surface;
    ctx.strokeStyle = M.rgba(M.catColor(activeIdx), 0.65);
    ctx.lineWidth = 1.2 * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(padX, cardY, s.w - 2 * padX, cardH, 5 * s.sc);
    else ctx.rect(padX, cardY, s.w - 2 * padX, cardH);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = M.catColor(activeIdx);
    ctx.font = M.font(s, 8.6, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('Monosemantic Latent L' + (activeIdx + 1) + ' (DistilBERT #' + fObj.id + ', fires on ' + Math.round(fObj.firing_rate * 100) + '% of corpus): ' + CONCEPTS[activeIdx], padX + 8 * s.sc, cardY + 6 * s.sc);

    var lines = fObj.top_headlines.slice(0, 3);
    var lineStep = Math.max(15 * s.sc, (cardH - 25 * s.sc) / 3);
    ctx.font = M.font(s, 8.2, 400, 'sans');
    for (var li = 0; li < lines.length; li++) {
      var ly = cardY + 23 * s.sc + li * lineStep;
      var txt = lines[li].text;
      var maxChars = Math.max(30, Math.floor((s.w - 2 * padX - 48 * s.sc) / (4.6 * s.sc)));
      if (txt.length > maxChars) txt = txt.slice(0, maxChars - 1) + '\u2026';
      ctx.fillStyle = lines[li].sarcastic ? C.orange : C.blue;
      ctx.fillText(lines[li].sarcastic ? '[sarc]' : '[news]', padX + 8 * s.sc, ly);
      ctx.fillStyle = C.ink;
      ctx.fillText(txt, padX + 43 * s.sc, ly);
    }
  }

  function drawAll() {
    drawLeft(); drawRight();
    var h = denseHidden();
    var dec = saeDecode(h);
    var nTrue = mixSecond >= 0 ? 2 : 1;
    var activeLabels = dec.chosen.map(function (idx) { return 'L' + (idx + 1) + ' (' + SHORT_CONCEPTS[idx] + ')'; }).join(' + ') || 'none';
    var interp = (mixSecond >= 0 && topK < 2)
      ? '<span class="o">2 concepts are co-active, so Top-1 keeps only the stronger latent; raise SAE Top-K to 2 to recover both cleanly.</span>'
      : '<span class="g">While both dense neurons respond across 4\u20135 topics at once, the Top-' + topK + ' SAE isolates ' + activeLabels + ' with 0 elsewhere.</span>';
    read.innerHTML = '<span class="k">active input</span> ' + nTrue + ' concept' + (nTrue > 1 ? 's' : '') +
      '<span class="sep">&middot;</span><span class="k">dense neurons</span> 2 (both active across topics)' +
      '<span class="sep">&middot;</span><span class="k">SAE latents firing</span> ' + dec.chosen.length + ' / 5 (' + activeLabels + ')' +
      '<div class="wread-interp">' + interp + '</div>';
  }

  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: pause };
});
