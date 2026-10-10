/* Vẽ lớp phủ khuôn mặt và cập nhật các vùng thông tin của giao diện. */
(function (global) {
  'use strict';

  var LABEL_STYLE = {
    without_mask: { text: 'Không đeo', color: '#ef4444', soft: 'rgba(239,68,68,0.16)' },
    with_mask: { text: 'Đeo đúng', color: '#22c55e', soft: 'rgba(34,197,94,0.16)' },
    mask_weared_incorrect: { text: 'Đeo sai', color: '#f59e0b', soft: 'rgba(245,158,11,0.16)' }
  };

  var MANUAL_STYLE = { text: 'Thủ công', color: '#38bdf8', soft: 'rgba(56,189,248,0.16)' };
  var PENDING_STYLE = { text: 'Đang chờ kết quả', color: '#94a3b8', soft: 'rgba(148,163,184,0.12)' };

  function styleFor(item) {
    if (item.source === 'manual' && (!item.label || !LABEL_STYLE[item.label])) return MANUAL_STYLE;
    if (!item.label) return PENDING_STYLE;
    return LABEL_STYLE[item.label] || PENDING_STYLE;
  }

  function roundRect(ctx, x, y, w, h, r) {
    var radius = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  function formatMs(value) {
    var n = Number(value);
    if (!isFinite(n) || n < 0) return '—';
    if (n >= 1000) return (n / 1000).toFixed(2) + ' s';
    if (n >= 100) return Math.round(n).toString();
    return (Math.round(n * 10) / 10).toFixed(1);
  }

  function formatPercent(value) {
    var n = Number(value);
    if (!isFinite(n)) return '—';
    var pct = n <= 1 ? n * 100 : n;
    return pct.toFixed(1) + '%';
  }

  function formatBytes(bytes) {
    var n = Number(bytes);
    if (!isFinite(n) || n <= 0) return '—';
    if (n < 1024) return Math.round(n) + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1024 / 1024).toFixed(2) + ' MB';
  }

  function Overlay(canvas) {
    this.canvas = canvas;
    this.ctx = canvas && canvas.getContext ? canvas.getContext('2d') : null;
    this.scaleFactor = 1;
  }

  Overlay.prototype.syncSize = function (width, height, dpr) {
    if (!this.canvas) return;
    this.imageWidth = width;
    this.imageHeight = height;
    var ratio = dpr || global.devicePixelRatio || 1;
    var cssWidth = this.canvas.clientWidth || width;
    var cssHeight = this.canvas.clientHeight || height;
    var targetWidth = Math.max(1, Math.round(cssWidth * ratio));
    var targetHeight = Math.max(1, Math.round(cssHeight * ratio));
    if (this.canvas.width !== targetWidth || this.canvas.height !== targetHeight) {
      this.canvas.width = targetWidth;
      this.canvas.height = targetHeight;
    }
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    this.scaleFactor = cssWidth ? targetWidth / cssWidth : 1;
  };

  // Match the video's object-fit: contain, including letterboxing on resize.
  Overlay.prototype.imageTransform = function () {
    var scale = Math.min(this.cssWidth / this.imageWidth, this.cssHeight / this.imageHeight);
    return {
      scale: scale,
      x: (this.cssWidth - this.imageWidth * scale) / 2,
      y: (this.cssHeight - this.imageHeight * scale) / 2
    };
  };

  Overlay.prototype.clear = function () {
    if (this.ctx && this.canvas) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  };

  Overlay.prototype.draw = function (items, options) {
    if (!this.ctx || !this.canvas) return;
    var opts = options || {};
    var ratio = this.scaleFactor;
    var ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!items || !items.length) return;

    ctx.save();
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    var transform = this.imageTransform();
    var imageWidth = this.imageWidth;
    var width = this.cssWidth;
    var height = this.cssHeight;
    var fontSize = width < 420 ? 11 : 13;
    ctx.font = '600 ' + fontSize + 'px system-ui, "Segoe UI", sans-serif';
    ctx.textBaseline = 'top';

    items.forEach(function (item) {
      var style = styleFor(item);
      // Mirror only positions; flipping the canvas context also reverses text.
      var imageX = opts.mirrored ? imageWidth - item.x - item.width : item.x;
      var x = transform.x + imageX * transform.scale;
      var y = transform.y + item.y * transform.scale;
      var w = item.width * transform.scale;
      var h = item.height * transform.scale;

      ctx.fillStyle = style.soft;
      ctx.fillRect(x, y, w, h);
      ctx.lineWidth = Math.max(2, width / 320);
      ctx.strokeStyle = style.color;
      ctx.setLineDash(item.stale ? [6, 5] : []);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      var parts = [style.text];
      if (item.label && isFinite(item.confidence)) parts.push(formatPercent(item.confidence));
      if (item.stale) parts.push('mới');
      var text = parts.join(' · ');
      var padX = 6;
      var padY = 4;
      var metrics = ctx.measureText(text);
      var boxW = metrics.width + padX * 2;
      var boxH = fontSize + padY * 2;

      var labelX = x;
      var labelY = y - boxH - 3;
      if (labelY < 0) labelY = y + 3;
      labelX = Math.max(0, Math.min(labelX, width - boxW));
      if (labelX + boxW > width) labelX = Math.max(0, width - boxW);

      ctx.fillStyle = style.color;
      roundRect(ctx, labelX, labelY, boxW, boxH, 5);
      ctx.fill();
      ctx.fillStyle = '#0b1220';
      ctx.fillText(text, labelX + padX, labelY + padY);

      if (item.id && width > 360) {
        ctx.fillStyle = 'rgba(226,232,240,0.92)';
        ctx.font = '500 ' + (fontSize - 2) + 'px system-ui, "Segoe UI", sans-serif';
        ctx.fillText('#' + item.id.replace(/^[a-z]+-/, ''), labelX + 2, labelY + boxH + 2);
        ctx.font = '600 ' + fontSize + 'px system-ui, "Segoe UI", sans-serif';
      }
    });

    ctx.restore();
  };

  Overlay.prototype.pointToImage = function (clientX, clientY, imageWidth, imageHeight, mirrored) {
    if (!this.canvas) return null;
    var rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    var scale = Math.min(rect.width / imageWidth, rect.height / imageHeight);
    var offsetX = (rect.width - imageWidth * scale) / 2;
    var offsetY = (rect.height - imageHeight * scale) / 2;
    var x = (clientX - rect.left - offsetX) / scale;
    var y = (clientY - rect.top - offsetY) / scale;
    if (x < 0 || y < 0 || x > imageWidth || y > imageHeight) return null;
    if (mirrored) x = imageWidth - x;
    return { x: x, y: y };
  };

  var Dom = {
    els: {},

    init: function () {
      var ids = [
        'healthChip', 'healthText', 'btnHealth', 'secureBanner', 'stageWrap', 'stage', 'video',
        'overlay', 'stageEmpty', 'cameraSelect', 'btnCamera', 'btnOnce', 'btnClearManual',
        'swAuto', 'swBoxes', 'rangeInterval', 'valInterval', 'rangeQuality', 'valQuality',
        'statusBadge', 'statusText', 'spinner', 'statRtt', 'statServer', 'statTotal',
        'statFaces', 'statImage', 'statBytes', 'statFps', 'faceList', 'faceCountChip',
        'inputApi', 'selectDetector', 'inputMaxWidth', 'inputTimeout', 'btnSaveConfig',
        'btnResetConfig', 'swMirror', 'configHint', 'inputFile', 'btnAnalyzeFile',
        'staticStage', 'staticImage', 'staticOverlay', 'staticEmpty', 'logBody', 'btnClearLog',
        'toastStack', 'detectorInfo'
      ];
      ids.forEach(function (id) {
        Dom.els[id] = document.getElementById(id);
      });
      return Dom.els;
    },

    setStatus: function (state, text) {
      var badge = Dom.els.statusBadge;
      var labels = { idle: 'Chưa chạy', loading: 'Đang xử lý', live: 'Đang chạy', ok: 'Sẵn sàng', warn: 'Cảnh báo', error: 'Lỗi' };
      var payload = typeof text === 'string' ? { text: text } : (text || {});
      if (badge) {
        badge.dataset.state = state;
        badge.textContent = payload.badge || labels[state] || state;
      }
      if (Dom.els.statusText) Dom.els.statusText.textContent = payload.text || '';
    },

    setBusy: function (busy) {
      if (Dom.els.spinner) Dom.els.spinner.hidden = !busy;
    },

    setHealth: function (state, text) {
      var chip = Dom.els.healthChip;
      if (chip) chip.dataset.state = state === true ? 'ok' : (state === false ? 'error' : 'unknown');
      if (Dom.els.healthText) Dom.els.healthText.textContent = text;
    },

    setDetectorInfo: function (text, state) {
      if (!Dom.els.detectorInfo) return;
      Dom.els.detectorInfo.textContent = text;
      Dom.els.detectorInfo.dataset.state = state || '';
    },

    setStats: function (stats) {
      var s = stats || {};
      if (Dom.els.statRtt) Dom.els.statRtt.textContent = s.roundTripMs === undefined || s.roundTripMs === null ? '—' : formatMs(s.roundTripMs);
      if (Dom.els.statServer) Dom.els.statServer.textContent = s.serverMs === undefined || s.serverMs === null ? '—' : formatMs(s.serverMs);
      if (Dom.els.statTotal) Dom.els.statTotal.textContent = s.totalMs === undefined || s.totalMs === null ? '—' : formatMs(s.totalMs);
      if (Dom.els.statFaces) Dom.els.statFaces.textContent = (s.sentFaces || 0) + ' / ' + (s.receivedFaces || 0);
      if (Dom.els.statImage) Dom.els.statImage.textContent = s.width && s.height ? s.width + '×' + s.height : '—';
      if (Dom.els.statBytes) Dom.els.statBytes.textContent = s.bytes === undefined ? '—' : formatBytes(s.bytes);
      if (Dom.els.statFps) Dom.els.statFps.textContent = s.fps === undefined || s.fps === null ? '—' : s.fps.toFixed(2);
    },

    renderFaces: function (items) {
      var list = Dom.els.faceList;
      if (!list) return;
      if (Dom.els.faceCountChip) Dom.els.faceCountChip.textContent = (items || []).length + ' mặt';
      if (!items || !items.length) {
        list.innerHTML = '<li class="empty">Chưa có kết quả. Bật camera rồi đưa khuôn mặt vào khung hình.</li>';
        return;
      }
      list.innerHTML = items.map(function (item) {
        var style = styleFor(item);
        var conf = item.label && isFinite(item.confidence)
          ? '<span class="pill">' + formatPercent(item.confidence) + '</span>'
          : '<span class="pill pill-muted">—</span>';
        var inference = isFinite(item.inferenceMs) && item.inferenceMs > 0
          ? formatMs(item.inferenceMs) + ' ms'
          : '—';
        var source = item.source === 'manual' ? 'thủ công' : 'tự động';
        return '' +
          '<li class="face-item">' +
            '<span class="face-dot" style="background:' + style.color + '"></span>' +
            '<div class="face-main">' +
              '<div class="face-title"><b>' + escapeHtml(item.labelText || style.text) + '</b>' + conf + '</div>' +
              '<div class="face-meta">#' + escapeHtml(item.id) + ' · ' + escapeHtml(source) +
                ' · box (' + Math.round(item.x) + ', ' + Math.round(item.y) + ', ' +
                Math.round(item.width) + '×' + Math.round(item.height) + ')</div>' +
            '</div>' +
            '<div class="face-ms"><span>' + inference + '</span><small>suy luận</small></div>' +
          '</li>';
      }).join('');
    },

    renderLog: function (entries) {
      var body = Dom.els.logBody;
      if (!body) return;
      if (!entries || !entries.length) {
        body.innerHTML = '<tr class="empty-row"><td colspan="5">Chưa có lần gửi nào.</td></tr>';
        return;
      }
      body.innerHTML = entries.map(function (entry) {
        var ok = entry.ok !== false;
        return '<tr class="' + (ok ? '' : 'row-error') + '">' +
          '<td>' + escapeHtml(entry.time) + '</td>' +
          '<td>' + (entry.faces === undefined ? '—' : entry.faces) + '</td>' +
          '<td>' + (entry.rtt === null || entry.rtt === undefined ? '—' : formatMs(entry.rtt)) + '</td>' +
          '<td>' + (entry.server === null || entry.server === undefined ? '—' : formatMs(entry.server)) + '</td>' +
          '<td class="log-status">' + escapeHtml(entry.status) + '</td>' +
        '</tr>';
      }).join('');
    },

    toast: function (message, kind, timeout) {
      var stack = Dom.els.toastStack;
      if (!stack) return;
      var node = document.createElement('div');
      node.className = 'toast toast-' + (kind || 'info');
      node.textContent = message;
      stack.appendChild(node);
      global.setTimeout(function () {
        node.classList.add('toast-out');
        global.setTimeout(function () {
          if (node.parentNode) node.parentNode.removeChild(node);
        }, 300);
      }, timeout || 4200);
    },

    setCameraEnabled: function (on) {
      var btn = Dom.els.btnCamera;
      if (btn) {
        btn.textContent = on ? 'Tắt camera' : 'Bật camera';
        btn.classList.toggle('btn-danger', on);
        btn.classList.toggle('btn-primary', !on);
      }
      if (Dom.els.btnOnce) Dom.els.btnOnce.disabled = !on;
      if (Dom.els.btnClearManual) Dom.els.btnClearManual.disabled = !on;
      if (Dom.els.stageEmpty) Dom.els.stageEmpty.hidden = on;
      if (Dom.els.stage) Dom.els.stage.classList.toggle('is-live', !!on);
    }
  };

  function escapeHtml(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  global.LAB3Ui = {
    Overlay: Overlay,
    Dom: Dom,
    LABEL_STYLE: LABEL_STYLE,
    MANUAL_STYLE: MANUAL_STYLE,
    PENDING_STYLE: PENDING_STYLE,
    styleFor: styleFor,
    formatMs: formatMs,
    formatPercent: formatPercent,
    formatBytes: formatBytes,
    escapeHtml: escapeHtml
  };
})(window);
