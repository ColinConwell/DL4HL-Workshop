/* ══════════════════════════════════════════════════════════════════
   Figure 10 — The Neuroconnectionist Trifecta & opportunistic model
   experiments
   Inspired directly by Colin Conwell's Types-Of-DL-Models deck:
   perceptual representations reflect three interacting pressures —
   Sensory Diet (Input), Computational Architecture, and Computational
   Objective (Task). Left: how five canonical objectives sculpt the
   latent hypersphere. Right: holding two pressures constant to isolate
   the third across open-source model zoos.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('trifecta_explorer', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 10,
    title: 'Types of deep learning models: the neuroconnectionist trifecta',
    hint: 'Switch Learning objective and press play to watch points organize on the sphere &middot; vary one Trifecta factor on the right'
  });

  var taskMode = cfg.objective || 'contrastive';
  var varyAxis = cfg.vary || 'task';
  var prog = 1.0; // 0..1 training progression
  var yaw = -0.42, pitch = 0.28, dragOrigin = null;
  var ready = false;

  /* 36 instances (12 in each of 3 latent families, plus an augmented twin per instance) */
  var N = 24, initPts = [], initTwins = [], fam = [];
  (function buildInit() {
    var rng = M.rng(cfg.seed || 53);
    for (var i = 0; i < N; i++) {
      var f = i % 3;
      fam.push(f);
      var v = M.normVec([rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)]);
      var tw = M.normVec([v[0] + rng.normal(0, 0.55), v[1] + rng.normal(0, 0.55), v[2] + rng.normal(0, 0.55)]);
      initPts.push(v);
      initTwins.push(tw);
    }
  })();

  function targetPositions(mode) {
    var rng = M.rng(909);
    var centers = [
      M.normVec([0.82, 0.42, 0.38]),
      M.normVec([-0.78, 0.48, 0.40]),
      M.normVec([0.05, -0.86, 0.50])
    ];
    var pts = [], twins = [], extra = [];
    for (var i = 0; i < N; i++) {
      var f = fam[i], u = i / N;
      if (mode === 'supervised') {
        var c = centers[f];
        var p1 = M.normVec([c[0] + rng.normal(0, 0.12), c[1] + rng.normal(0, 0.12), c[2] + rng.normal(0, 0.12)]);
        pts.push(p1); twins.push(p1);
      } else if (mode === 'autoencode') {
        /* Smooth continuous manifold ribbon across latitude/longitude */
        var lon = (u - 0.5) * Math.PI * 1.7;
        var lat = (f - 1) * 0.48 + 0.12 * Math.sin(u * 9);
        var p2 = M.normVec([Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)]);
        pts.push(p2); twins.push(p2);
      } else if (mode === 'puzzle') {
        /* 4 discrete rotation / jigsaw sectors around equator */
        var rot = i % 4, ang = (rot / 4) * Math.PI * 2 + rng.normal(0, 0.13);
        var p3 = M.normVec([Math.cos(ang), rng.normal(0, 0.16), Math.sin(ang)]);
        pts.push(p3); twins.push(p3);
      } else if (mode === 'contrastive') {
        /* Uniform repulsion across the sphere + augmented views pulled tight to each instance! */
        var phi = Math.acos(1 - 2 * (i + 0.5) / N);
        var theta = Math.PI * (1 + Math.sqrt(5)) * i;
        var p4 = M.normVec([Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta)]);
        var tw4 = M.normVec([p4[0] + rng.normal(0, 0.05), p4[1] + rng.normal(0, 0.05), p4[2] + rng.normal(0, 0.05)]);
        pts.push(p4); twins.push(tw4);
      } else {
        /* CLIP language alignment: image vectors + paired caption vectors meet near semantic anchors */
        var c5 = centers[f];
        var p5 = M.normVec([c5[0] + rng.normal(0, 0.20), c5[1] + rng.normal(0, 0.20), c5[2] + rng.normal(0, 0.20)]);
        var cap = M.normVec([p5[0] + rng.normal(0, 0.06), p5[1] + rng.normal(0, 0.06), p5[2] + rng.normal(0, 0.06)]);
        pts.push(p5); twins.push(cap);
        extra = centers;
      }
    }
    return { pts: pts, twins: twins, anchors: extra };
  }

  var EXPERIMENTS = {
    input: {
      hold: 'Architecture (AlexNet) + Task (IPCL contrastive)',
      vary: 'Sensory Diet (Input stream)',
      question: 'Does object vs. scene vs. face experience alone sculpt category selectivity?',
      arms: [
        { name: 'Open-IPCL \u00b7 ImageNet-1K', detail: '1.2M object-centric photos (Konkle & Alvarez, 2022)', col: 'blue' },
        { name: 'Open-IPCL \u00b7 Places-365', detail: '1.8M navigable indoor & outdoor scenes', col: 'green' },
        { name: 'Open-IPCL \u00b7 VGGFace2', detail: '3.3M human face exemplars across pose & lighting', col: 'orange' }
      ]
    },
    arch: {
      hold: 'Sensory Diet (ImageNet-1K) + Task (Category supervision)',
      vary: 'Computational Architecture',
      question: 'Do local convolutions learn fundamentally different geometry than global attention?',
      arms: [
        { name: 'timm \u00b7 ResNet-50', detail: '2D spatial convolutions with residual skip connections', col: 'blue' },
        { name: 'timm \u00b7 ViT-B/16', detail: 'Patch tokens routed through global multi-head self-attention', col: 'green' },
        { name: 'timm \u00b7 ConvNeXt-B', detail: 'Modernized depthwise-separable convolutional blocks', col: 'orange' }
      ]
    },
    task: {
      hold: 'Sensory Diet (ImageNet / YFCC) + Architecture (ResNet-50 / ViT-B)',
      vary: 'Computational Objective (Learning signal)',
      question: 'Can label-free self-supervision or language alignment match explicit category labels?',
      arms: [
        { name: 'Torchvision \u00b7 Supervised', detail: '1,000 explicit human-assigned category labels', col: 'blue' },
        { name: 'VISSL \u00b7 SimCLR / BarlowTwins', detail: 'Unimodal instance-contrastive augmentation invariance', col: 'green' },
        { name: 'OpenCLIP / SLIP \u00b7 Language-aligned', detail: 'Multimodal image-caption cosine alignment on the sphere', col: 'orange' }
      ]
    }
  };

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, 'how the learning signal organizes the latent sphere');
  M.mini(split.b, 'opportunistic experiment designer (open-source zoos)');
  var hf = function (w) { return M.clamp(w * 0.80, 240, 325); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: '3D latent hypersphere showing how supervised, autoencoder, puzzle, contrastive, and CLIP tasks arrange representations' });
  var right = M.stage(split.b, { height: function () { return left.h || 260; }, ref: 400, label: 'Controlled model-zoo comparison holding two trifecta pressures constant while varying the third' });

  var tr = M.transport(root, {
    n: 61, fps: 30, aria: 'Training progression from random initialization to learned geometry',
    fmt: function (k) { return 'training ' + Math.round((k / 60) * 100) + '%'; },
    onFrame: function (k) { prog = M.ease(k / 60); drawAll(); }
  });
  tr.show(60);

  var row1 = M.row(root);
  var objT = M.toggle(row1, {
    label: 'Objective', value: taskMode,
    options: [
      ['supervised', 'Supervised'],
      ['autoencode', 'Autoencode'],
      ['puzzle', 'Rotation/Jigsaw'],
      ['contrastive', 'Contrastive'],
      ['clip', 'CLIP align']
    ],
    onChange: function (v) { taskMode = v; tr.show(60); drawAll(); }
  });

  var row2 = M.row(root);
  var varyT = M.toggle(row2, {
    label: 'Vary 1 factor', value: varyAxis,
    options: [
      ['input', 'Sensory Diet'],
      ['arch', 'Architecture'],
      ['task', 'Task Objective']
    ],
    onChange: function (v) { varyAxis = v; drawAll(); }
  });
  M.spacer(row2);
  M.resetButton(row2, function () {
    tr.pause();
    taskMode = 'contrastive'; varyAxis = 'task'; yaw = -0.42; pitch = 0.28;
    objT.set('contrastive', true); varyT.set('task', true);
    tr.show(60); drawAll();
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
    var cx = s.w * 0.5, cy = s.h * 0.53;
    M.drawSphere(ctx, cx, cy, R, yaw, pitch, s);

    var targ = targetPositions(taskMode);
    var famCols = [C.blue, C.orange, C.green];

    for (var i = 0; i < N; i++) {
      var curP = M.normVec([
        M.lerp(initPts[i][0], targ.pts[i][0], prog),
        M.lerp(initPts[i][1], targ.pts[i][1], prog),
        M.lerp(initPts[i][2], targ.pts[i][2], prog)
      ]);
      var curT = M.normVec([
        M.lerp(initTwins[i][0], targ.twins[i][0], prog),
        M.lerp(initTwins[i][1], targ.twins[i][1], prog),
        M.lerp(initTwins[i][2], targ.twins[i][2], prog)
      ]);

      var pp = M.sphereProject(curP, cx, cy, R, yaw, pitch);
      var col = taskMode === 'puzzle' ? M.catColor(i % 4) : famCols[fam[i]];
      var front = pp.z >= 0;

      if (taskMode === 'contrastive' || taskMode === 'clip') {
        var pt = M.sphereProject(curT, cx, cy, R, yaw, pitch);
        ctx.strokeStyle = M.rgba(col, front ? 0.55 : 0.2);
        ctx.lineWidth = 1.2 * s.sc;
        ctx.beginPath(); ctx.moveTo(pp.x, pp.y); ctx.lineTo(pt.x, pt.y); ctx.stroke();
        if (taskMode === 'clip') {
          /* Hollow square / ring for paired caption embedding */
          M.dot(ctx, pt.x, pt.y, 3.2 * s.sc, C.surface, M.rgba(col, front ? 0.9 : 0.35), 1.4 * s.sc);
        } else {
          M.dot(ctx, pt.x, pt.y, 2.7 * s.sc, M.rgba(col, front ? 0.75 : 0.25));
        }
      }
      M.dot(ctx, pp.x, pp.y, (front ? 3.8 : 2.7) * s.sc, M.rgba(col, front ? 0.92 : 0.30), front ? C.surface : null, 0.9);
    }

    var caption = {
      supervised: 'Supervised: explicit labels pull categories apart ("Gavagai" problem)',
      autoencode: 'Autoencoder: compression preserves smooth generative coordinates',
      puzzle: 'Pretext puzzle: 0\u00b0/90\u00b0/180\u00b0/270\u00b0 rotation or jigsaw sectors',
      contrastive: 'Contrastive: pull augmented views together, push instances apart',
      clip: 'CLIP: pull paired image (\u25cf) and caption (\u25cb) vectors together'
    }[taskMode];
    M.pill(ctx, s, caption, 10 * s.sc, 14 * s.sc, { size: 8.3 });
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);
    var exp = EXPERIMENTS[varyAxis];
    var padX = 12 * s.sc;

    /* Trifecta 3-pill header */
    var pillars = [
      ['input', 'Input (Diet)'],
      ['arch', 'Architecture'],
      ['task', 'Task (Signal)']
    ];
    var pw = (s.w - 2 * padX - 16 * s.sc) / 3;
    pillars.forEach(function (pl, idx) {
      var isVary = pl[0] === varyAxis;
      var bx = padX + idx * (pw + 8 * s.sc), by = 10 * s.sc, bh = 28 * s.sc;
      ctx.fillStyle = isVary ? M.rgba(C.orange, 0.16) : C.surface;
      ctx.strokeStyle = isVary ? C.orange : C.border;
      ctx.lineWidth = (isVary ? 1.7 : 1.0) * s.sc;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, by, pw, bh, 4 * s.sc); else ctx.rect(bx, by, pw, bh);
      ctx.fill(); ctx.stroke();

      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = isVary ? C.orange : C.muted;
      ctx.font = M.font(s, 8.2, 600, 'sans');
      ctx.fillText((isVary ? 'VARY: ' : 'HOLD: ') + pl[1], bx + pw / 2, by + bh / 2);
    });

    /* Scientific question banner */
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.8, 600, 'sans');
    var qTxt = exp.question;
    var maxCh = Math.max(36, Math.floor((s.w - 2 * padX) / (4.8 * s.sc)));
    if (qTxt.length > maxCh) qTxt = qTxt.slice(0, maxCh - 1) + '\u2026';
    ctx.fillText(qTxt, padX, 46 * s.sc);

    /* Three matched open-source model cards */
    var startY = 66 * s.sc, gap = 7 * s.sc;
    var cardH = (s.h - startY - 10 * s.sc - 2 * gap) / 3;
    exp.arms.forEach(function (arm, idx) {
      var y0 = startY + idx * (cardH + gap);
      var col = C[arm.col] || C.blue;
      ctx.fillStyle = C.surface;
      ctx.strokeStyle = M.rgba(col, 0.65);
      ctx.lineWidth = 1.2 * s.sc;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(padX, y0, s.w - 2 * padX, cardH, 4 * s.sc);
      else ctx.rect(padX, y0, s.w - 2 * padX, cardH);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = col;
      ctx.fillRect(padX, y0, 4 * s.sc, cardH);

      ctx.fillStyle = C.ink; ctx.font = M.font(s, 9.2, 600, 'sans');
      ctx.textBaseline = 'middle';
      ctx.fillText(arm.name, padX + 11 * s.sc, y0 + cardH * 0.36);

      ctx.fillStyle = C.muted; ctx.font = M.font(s, 8.3, 400, 'sans');
      var det = arm.detail;
      if (det.length > maxCh) det = det.slice(0, maxCh - 1) + '\u2026';
      ctx.fillText(det, padX + 11 * s.sc, y0 + cardH * 0.72);
    });
  }

  function drawAll() {
    if (!ready) return;
    drawLeft(); drawRight();
    var exp = EXPERIMENTS[varyAxis];
    read.innerHTML = '<span class="k">sphere objective</span> ' + taskMode +
      '<span class="sep">&middot;</span><span class="k">holding constant</span> ' + M.esc(exp.hold) +
      '<span class="sep">&middot;</span><span class="g">open-source model zoos turn input, architecture, and task into controlled experiments</span>';
  }

  left.draw = drawAll; right.draw = drawAll;
  ready = true;
  tr.show(60);
  left.request(); right.request();
  return { destroy: function () { tr.pause(); } };
});
