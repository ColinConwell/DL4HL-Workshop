/* ══════════════════════════════════════════════════════════════════
   Figure 4 — Inside a tabular MLP: forward activations & error backprop
   Structured predictors enter on the left, combine through weighted
   connections and ReLU hinges to produce a prediction on the right,
   and propagate prediction error backward so each weight updates in
   proportion to how strongly it contributed to the miss.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('mlp_backprop', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 4,
    title: 'Inside a tabular neural network: forward pass & error backpropagation',
    hint: 'Press play or step to run forward + backward updates &middot; modulate playback speed &middot; adjust predictor sliders to test live inputs'
  });

  var seed = cfg.seed || 19;
  var FEAT_LABELS = {
    signal: 'Signal Strength',
    noise: 'Noise Level',
    prior: 'Prior Cue'
  };
  var featKeys = ['signal', 'noise', 'prior'];
  var xCur = [0.65, -0.45, 0.8];
  var Xtr = [], ytr = [];
  var W1 = [], b1 = [], W2 = [], b2 = 0;
  var gW1 = [], gW2 = [], lastA1 = [0, 0, 0, 0], lastPred = 0, lastTarget = 0;
  var lossHist = [], stepCount = 0;
  var phase = 0;      // 0..0.5 forward pulse, 0.5..1.0 backward pulse
  var speedMult = 1;  // modulated via Speed control (default 1x is deliberately unhurried)

  function targetFn(x) {
    /* Nonlinear tabular interaction: schooling boosts expenditure more in urban settings,
       while household size has diminishing returns. */
    return 0.55 * x[0] + 0.35 * Math.max(0, x[1] + 0.4) + 0.45 * x[0] * x[2] - 0.2 * x[1] * x[1] + 0.25;
  }

  function initAll() {
    var rng = M.rng(seed);
    Xtr = []; ytr = [];
    for (var i = 0; i < 64; i++) {
      var row = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.next() < 0.5 ? -0.8 : 0.8];
      Xtr.push(row);
      ytr.push(targetFn(row) + rng.normal(0, 0.05));
    }
    W1 = []; b1 = []; W2 = []; b2 = 0;
    gW1 = []; gW2 = [0, 0, 0, 0];
    for (var j = 0; j < 4; j++) {
      W1.push([rng.normal(0, 0.55), rng.normal(0, 0.55), rng.normal(0, 0.55)]);
      gW1.push([0, 0, 0]);
      b1.push(rng.uniform(-0.15, 0.15));
      W2.push(rng.normal(0, 0.55));
    }
    stepCount = 0;
    phase = 0;
    lossHist = [evalLoss()];
    inspectCurrent();
  }

  function forwardOne(x) {
    var z1 = new Array(4), a1 = new Array(4), out = b2;
    for (var j = 0; j < 4; j++) {
      z1[j] = b1[j] + W1[j][0] * x[0] + W1[j][1] * x[1] + W1[j][2] * x[2];
      a1[j] = z1[j] > 0 ? z1[j] : 0.08 * z1[j];
      out += W2[j] * a1[j];
    }
    return { z1: z1, a1: a1, out: out };
  }

  function evalLoss() {
    var s = 0;
    for (var i = 0; i < Xtr.length; i++) {
      var e = forwardOne(Xtr[i]).out - ytr[i];
      s += e * e;
    }
    return s / Xtr.length;
  }

  function inspectCurrent() {
    var fw = forwardOne(xCur);
    lastA1 = fw.a1;
    lastPred = fw.out;
    lastTarget = targetFn(xCur);
    var err = 2 * (lastPred - lastTarget);
    for (var j = 0; j < 4; j++) {
      gW2[j] = err * fw.a1[j];
      var dz = err * W2[j] * (fw.z1[j] > 0 ? 1 : 0.08);
      for (var k = 0; k < 3; k++) gW1[j][k] = dz * xCur[k];
    }
  }

  function trainStep(iters) {
    var lr = lrS ? lrS.get() : 0.08, n = Xtr.length;
    for (var it = 0; it < iters; it++) {
      var dW1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
      var db1 = [0, 0, 0, 0], dW2 = [0, 0, 0, 0], db2 = 0;
      for (var i = 0; i < n; i++) {
        var fw = forwardOne(Xtr[i]);
        var err = 2 * (fw.out - ytr[i]) / n;
        db2 += err;
        for (var j = 0; j < 4; j++) {
          dW2[j] += err * fw.a1[j];
          var dz = err * W2[j] * (fw.z1[j] > 0 ? 1 : 0.08);
          db1[j] += dz;
          for (var k = 0; k < 3; k++) dW1[j][k] += dz * Xtr[i][k];
        }
      }
      b2 -= lr * db2;
      for (var j2 = 0; j2 < 4; j2++) {
        W2[j2] -= lr * dW2[j2];
        b1[j2] -= lr * db1[j2];
        for (var k2 = 0; k2 < 3; k2++) W1[j2][k2] -= lr * dW1[j2][k2];
      }
      stepCount++;
      lossHist.push(evalLoss());
      if (lossHist.length > 160) lossHist.shift();
    }
    inspectCurrent();
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.38);
  M.mini(split.a, 'network graph: weights (width) + gradient credit (color)');
  M.mini(split.b, 'training MSE across gradient steps');
  var hf = function (w) { return M.clamp(w * 0.68, 230, 310); };
  var left = M.stage(split.a, { height: hf, ref: 440, label: 'Tabular neural network diagram showing forward activations and backward gradients' });
  var right = M.stage(split.b, { height: function () { return left.h || 250; }, ref: 320, label: 'Mean squared error falling over training steps' });

  M.legend(root, [
    { color: C.blue, label: 'positive weight / forward signal', line: true },
    { color: C.orange, label: 'negative weight / error gradient', line: true },
    { color: C.green, label: 'Hit y vs. predicted \u0177' }
  ]);

  var row1 = M.row(root);
  var playBtn = M.button(row1, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Train tabular MLP continuously' });
  M.button(row1, 'step', function () { pause(); phase = 0.75; trainStep(3); drawAll(); }, { aria: 'Run 3 backprop updates' });
  var speedT = M.toggle(row1, {
    label: 'Speed', value: '1',
    options: [['0.5', '0.5\u00d7'], ['1', '1\u00d7'], ['2', '2\u00d7'], ['4', '4\u00d7']],
    onChange: function (v) { speedMult = +v; }
  });
  var viewT = M.toggle(row1, {
    label: 'Color edges by', value: 'grad',
    options: [['weight', 'Weight sign'], ['grad', 'Backprop gradient']],
    onChange: function () { drawAll(); }
  });

  var row2 = M.row(root);
  var lrS = M.slider(row2, {
    label: 'Learning rate', min: 0.02, max: 0.22, step: 0.01, value: 0.08, accent: 'blue',
    fmt: function (v) { return M.f(v, 2); },
    onInput: function () { drawAll(); }
  });
  var x0S = M.slider(row2, {
    label: 'Signal Strength', min: -1, max: 1, step: 0.05, value: xCur[0], accent: 'blue',
    fmt: function (v) { return M.f(v, 2); },
    onInput: function (v) { xCur[0] = v; inspectCurrent(); drawAll(); }
  });
  var x1S = M.slider(row2, {
    label: 'Noise Level', min: -1, max: 1, step: 0.05, value: xCur[1], accent: 'orange',
    fmt: function (v) { return M.f(v, 2); },
    onInput: function (v) { xCur[1] = v; inspectCurrent(); drawAll(); }
  });
  M.spacer(row2);
  M.resetButton(row2, function () {
    pause();
    xCur = [0.65, -0.45, 0.8];
    speedMult = 1;
    speedT.set('1', true);
    x0S.set(xCur[0], true); x1S.set(xCur[1], true); lrS.set(0.08, true);
    initAll(); drawAll();
  });
  var read = M.readout(root);

  /* Unhurried cycle: at 1x speed, one full forward->backward wave takes ~0.95s
     and applies 2 gradient updates at the end of each backward sweep. */
  var loop = M.loop(function (dt) {
    var prevPhase = phase;
    phase = phase + dt * 0.95 * speedMult;
    while (phase >= 1) {
      phase -= 1;
      trainStep(2);
    }
    if (prevPhase < 0.5 && phase >= 0.5) inspectCurrent();
    drawAll();
    if (stepCount >= 140 && lossHist[lossHist.length - 1] < 0.005) { pause(); return false; }
    return true;
  });
  function play() { if (stepCount >= 140) initAll(); loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  var vR = M.view(right, { x: [0, 60], y: [0, 0.5], pad: { l: 44, r: 14, t: 16, b: 36 } });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    M.clear(ctx, s);

    var lx = [s.w * 0.30, s.w * 0.59, s.w * 0.86];
    var inY = [0.24, 0.52, 0.80].map(function (f) { return s.h * f; });
    var hidY = [0.17, 0.40, 0.63, 0.86].map(function (f) { return s.h * f; });
    var outY = [s.h * 0.52];
    var byGrad = viewT.get() === 'grad';

    /* Draw edges Input -> Hidden */
    for (var j = 0; j < 4; j++) {
      for (var k = 0; k < 3; k++) {
        var w = W1[j][k], g = gW1[j][k];
        var val = byGrad ? -g * 3.0 : w;
        var col = val >= 0 ? C.blue : C.orange;
        ctx.strokeStyle = M.rgba(col, M.clamp(0.22 + Math.abs(val) * 0.75, 0.18, 0.92));
        ctx.lineWidth = M.clamp(0.9 + Math.abs(w) * 2.4, 0.9, 4.5) * s.sc;
        ctx.beginPath(); ctx.moveTo(lx[0], inY[k]); ctx.lineTo(lx[1], hidY[j]); ctx.stroke();
        if (loop.on) {
          /* Forward wave: 0..0.25 moves Input->Hidden; Backward wave: 0.75..1.0 moves Hidden->Input */
          if (phase < 0.26) {
            var tf = M.clamp(phase / 0.26, 0, 1);
            M.dot(ctx, M.lerp(lx[0], lx[1], tf), M.lerp(inY[k], hidY[j], tf), 2.5 * s.sc, C.blue);
          } else if (phase >= 0.74) {
            var tb = M.clamp((phase - 0.74) / 0.26, 0, 1);
            M.dot(ctx, M.lerp(lx[1], lx[0], tb), M.lerp(hidY[j], inY[k], tb), 2.5 * s.sc, C.orange);
          }
        }
      }
    }

    /* Draw edges Hidden -> Output */
    for (var j2 = 0; j2 < 4; j2++) {
      var w2 = W2[j2], g2 = gW2[j2];
      var val2 = byGrad ? -g2 * 2.5 : w2;
      var col2 = val2 >= 0 ? C.blue : C.orange;
      ctx.strokeStyle = M.rgba(col2, M.clamp(0.25 + Math.abs(val2) * 0.75, 0.2, 0.95));
      ctx.lineWidth = M.clamp(1.0 + Math.abs(w2) * 2.6, 1.0, 4.8) * s.sc;
      ctx.beginPath(); ctx.moveTo(lx[1], hidY[j2]); ctx.lineTo(lx[2], outY[0]); ctx.stroke();
      if (loop.on) {
        if (phase >= 0.24 && phase < 0.50) {
          var tf2 = M.clamp((phase - 0.24) / 0.26, 0, 1);
          M.dot(ctx, M.lerp(lx[1], lx[2], tf2), M.lerp(hidY[j2], outY[0], tf2), 2.7 * s.sc, C.blue);
        } else if (phase >= 0.50 && phase < 0.76) {
          var tb2 = M.clamp((phase - 0.50) / 0.26, 0, 1);
          M.dot(ctx, M.lerp(lx[2], lx[1], tb2), M.lerp(outY[0], hidY[j2], tb2), 2.7 * s.sc, C.orange);
        }
      }
    }

    /* Input nodes with mapped display names */
    for (var i = 0; i < 3; i++) {
      M.dot(ctx, lx[0], inY[i], 12 * s.sc, C.surface, C.blue, 1.8 * s.sc);
      ctx.fillStyle = C.ink; ctx.font = M.font(s, 9.2, 500);
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(M.f(xCur[i], 1), lx[0], inY[i]);
      ctx.fillStyle = C.muted; ctx.font = M.font(s, 9.0, 500, 'sans');
      ctx.textAlign = 'right';
      ctx.fillText(FEAT_LABELS[featKeys[i]], lx[0] - 15 * s.sc, inY[i]);
    }

    /* Hidden ReLU nodes */
    for (var h = 0; h < 4; h++) {
      var act = lastA1[h] > 0;
      M.dot(ctx, lx[1], hidY[h], 13 * s.sc, act ? C.blueSoft : C.surface, act ? C.blue : C.muted, 1.8 * s.sc);
      ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.font = M.font(s, 9);
      ctx.fillText(M.f(lastA1[h], 2), lx[1], hidY[h]);
    }

    /* Output node + Hit / Miss comparison */
    M.dot(ctx, lx[2], outY[0], 15 * s.sc, C.surface, C.green, 2.2 * s.sc);
    ctx.fillStyle = C.ink; ctx.font = M.font(s, 9.5, 600);
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(M.f(lastPred, 2), lx[2], outY[0]);
    ctx.fillStyle = C.muted; ctx.font = M.font(s, 9, 500, 'sans');
    ctx.fillText('Hit ' + M.f(lastTarget, 2), lx[2], outY[0] + 24 * s.sc);
    ctx.fillText('Miss ' + M.f(lastPred - lastTarget, 2), lx[2], outY[0] - 24 * s.sc);
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    var ymax = Math.max(0.08, Math.max.apply(null, lossHist) * 1.08);
    vR.setDomain([0, Math.max(30, lossHist.length - 1)], [0, ymax]);
    M.clear(ctx, s);
    M.axes(ctx, vR, s, { xlabel: 'gradient steps', ylabel: 'mean squared error', nx: 4, ny: 4 });

    ctx.save(); vR.clip(ctx);
    ctx.strokeStyle = C.blue;
    ctx.lineWidth = 2.2 * s.sc;
    ctx.beginPath();
    for (var i = 0; i < lossHist.length; i++) {
      var px = vR.px(i), py = vR.py(lossHist[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    var last = lossHist.length - 1;
    M.dot(ctx, vR.px(last), vR.py(lossHist[last]), 4.2 * s.sc, C.orange, C.surface, 1.2 * s.sc);
    ctx.restore();
  }

  function drawAll() {
    drawLeft(); drawRight();
    var mse = lossHist[lossHist.length - 1];
    var dirNote = loop.on
      ? (phase < 0.5 ? 'Forward pass: weighted inputs propagate left-to-right to form prediction \u0177.' : 'Backward pass: Miss propagates right-to-left to assign credit across weights.')
      : 'Backprop scales each weight update by input activity \u00d7 downstream sensitivity.';
    read.innerHTML = '<span class="k">step</span> ' + stepCount +
      '<span class="sep">&middot;</span><span class="k">batch MSE</span> ' + M.f(mse, 4) +
      '<span class="sep">&middot;</span><span class="k">probe \u0177</span> ' + M.f(lastPred, 2) +
      ' vs <span class="k">Hit</span> ' + M.f(lastTarget, 2) +
      ' (<span class="k">Miss</span> ' + M.f(lastPred - lastTarget, 2) + ')' +
      '<div class="wread-interp"><span class="b">' + dirNote + '</span></div>';
  }

  initAll();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: pause };
});
