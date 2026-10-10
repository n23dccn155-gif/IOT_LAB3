/* Điều phối: camera, vòng lặp phát hiện, gửi ảnh + tọa độ tới API, vẽ kết quả. */
(function (global) {
  'use strict';

  var Config = global.LAB3Config;
  var Api = global.LAB3Api;
  var Detectors = global.LAB3Detectors;
  var Ui = global.LAB3Ui;
  var Dom = Ui.Dom;
  var els = Dom.els;

  var state = {
    config: Config.load(),
    client: null,
    stream: null,
    detector: null,
    detectorKind: 'manual',
    detectorWanted: null,
    detectorInit: null,
    detectorGeneration: 0,
    detectorNote: '',
    tracker: null,
    overlay: null,
    staticOverlay: null,
    captureCanvas: null,
    captureCtx: null,
    detectCanvas: null,
    detectCtx: null,
    faces: [],
    predictions: new Map(),
    manualBoxes: [],
    dragging: null,
    rafId: null,
    loopTimer: null,
    detectTimer: null,
    detectBusy: false,
    requestBusy: false,
    abortController: null,
    stats: {},
    lastAnalyzeAt: 0,
    analyzeGaps: [],
    logEntries: [],
    failures: 0,
    healthTimer: null,
    resumeAutoAfterHealth: false,
    stopped: false,
    cameraLabel: ''
  };

  function now() {
    return (global.performance && global.performance.now) ? global.performance.now() : Date.now();
  }

  function pad(n) {
    return n < 10 ? '0' + n : String(n);
  }

  function clockNow() {
    var d = new Date();
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }

  function note(kind, text) {
    Dom.toast(text, kind, kind === 'error' ? 6500 : 4200);
  }

  /* ---------- cấu hình ---------- */

  function applyConfigToForm() {
    var c = state.config;
    els.inputApi.value = c.apiBaseUrl;
    els.selectDetector.value = c.detector;
    els.inputMaxWidth.value = c.maxSendWidth;
    els.inputTimeout.value = c.timeoutMs;
    els.rangeInterval.value = c.requestIntervalMs;
    els.valInterval.textContent = c.requestIntervalMs;
    els.rangeQuality.value = c.jpegQuality.toFixed(2);
    els.valQuality.textContent = c.jpegQuality.toFixed(2);
    els.swAuto.checked = true;
    els.swBoxes.checked = c.showBoxes;
    els.swMirror.checked = c.mirror;
    els.stage.classList.toggle('mirrored', c.mirror);
    state.client = new Api.ApiClient(c);
    if (!state.tracker) {
      state.tracker = new Detectors.Tracker({ iouThreshold: c.iouThreshold, maxFaces: c.maxFaces });
    }
  }

  function saveConfig(patch, silent) {
    var result = Config.save(patch || {});
    state.config = result.config;
    applyConfigToForm();
    ensureDetector().then(function () {
      if (state.stream) startDetectLoop();
    }).catch(function () { /* đã hiển thị */ });
    checkHealth(true);
    if (!silent) {
      note(result.ok ? 'ok' : 'warn', result.ok
        ? 'Đã lưu cấu hình.'
        : 'Không lưu được localStorage, cấu hình vẫn áp dụng cho phiên này.');
    }
    return result;
  }

  /* ---------- API ---------- */

  function checkHealth(silent) {
    if (!silent) Dom.setHealth(null, 'Đang kiểm tra máy chủ…');
    return state.client.health().then(function (res) {
      if (res.ok) {
        var detail = res.text ? String(res.text).slice(0, 36) : '';
        Dom.setHealth(true, 'API sẵn sàng' + (detail ? ' · ' + detail : '') + ' · ' + Ui.formatMs(res.timing.totalMs) + ' ms');
        if (state.resumeAutoAfterHealth && state.stream) {
          state.resumeAutoAfterHealth = false;
          state.failures = 0;
          els.swAuto.checked = true;
          note('ok', 'API đã kết nối lại, tiếp tục nhận diện tự động.');
        }
      } else {
        var msg = res.error && res.error.message ? res.error.message : 'Không kết nối được API.';
        Dom.setHealth(false, msg.length > 96 ? msg.slice(0, 96) + '…' : msg);
      }
      return res;
    });
  }

  /* ---------- phát hiện khuôn mặt ---------- */

  function updateDetectorInfo(tone) {
    var text = state.detectorNote;
    if (state.cameraLabel) text += ' · Camera: ' + state.cameraLabel;
    Dom.setDetectorInfo(text, tone || state.detectorTone || '');
  }

  function ensureDetector() {
    var wanted = state.config.detector;
    if (state.detector && state.detectorWanted === wanted) {
      return Promise.resolve(state.detector);
    }
    if (state.detectorInit && state.detectorWanted === wanted) {
      return state.detectorInit;
    }
    stopDetectLoop();
    if (state.detector && state.detector.close) {
      try { state.detector.close(); } catch (err) { /* bỏ qua */ }
    }
    state.detector = null;
    state.detectorWanted = wanted;
    var generation = ++state.detectorGeneration;

    var choose = wanted === 'auto' ? Detectors.resolveAuto() : Promise.resolve(wanted);

    var initialization = choose.then(function (kind) {
      if (generation !== state.detectorGeneration) return ensureDetector();
      if (kind === 'native' && !Detectors.isNativeSupported()) kind = 'manual';
      if (kind === 'mediapipe' && global.location && global.location.protocol === 'file:') {
        kind = 'manual';
        Dom.setDetectorInfo('MediaPipe cần chạy qua máy chủ tĩnh (http/https), đang dùng đánh dấu thủ công.', 'warn');
      }
      state.detectorKind = kind;
      var detector = Detectors.createDetector(kind, {
        iouThreshold: state.config.iouThreshold,
        maxFaces: state.config.maxFaces
      });
      return detector.init().then(function () {
        if (generation !== state.detectorGeneration) {
          if (detector.close) detector.close();
          return ensureDetector();
        }
        state.detector = detector;
        state.tracker = new Detectors.Tracker({
          iouThreshold: state.config.iouThreshold,
          maxFaces: state.config.maxFaces
        });
        describeDetector(kind);
        return detector;
      }).catch(function (err) {
        if (generation !== state.detectorGeneration) return ensureDetector();
        var message = err && err.message ? err.message : String(err);
        if (kind === 'mediapipe') {
          note('warn', 'Không dùng được MediaPipe (' + message + '). Chuyển sang đánh dấu thủ công.');
          state.detectorKind = 'manual';
          var manual = Detectors.createDetector('manual', {});
          state.detector = manual;
          describeDetector('manual');
          return manual;
        }
        state.detectorTone = 'error';
        updateDetectorInfo();
        throw err;
      });
    });
    state.detectorInit = initialization;
    return initialization.then(function (detector) {
      if (state.detectorInit === initialization) state.detectorInit = null;
      return detector;
    }, function (error) {
      if (state.detectorInit === initialization) state.detectorInit = null;
      throw error;
    });
  }

  function describeDetector(kind) {
    state.detectorTone = 'ok';
    if (kind === 'native') {
      state.detectorNote = 'Đang dùng FaceDetector API của trình duyệt: tự phát hiện khuôn mặt, không tải thư viện ngoài.';
    } else if (kind === 'mediapipe') {
      state.detectorNote = 'Đang dùng MediaPipe FaceDetector ' +
        (state.detector.delegate || '') + ' (tự động phát hiện khuôn mặt).';
    } else {
      state.detectorTone = 'warn';
      state.detectorNote = (Detectors.isNativeSupported()
        ? 'Đang dùng đánh dấu thủ công (trình duyệt có FaceDetector API nhưng bạn chọn thủ công).'
        : 'Trình duyệt không có FaceDetector API, dùng đánh dấu thủ công.') +
        ' Kéo chuột trên ảnh để bỏ khung khuôn mặt, Alt+nhấp để xóa khung.';
    }
    updateDetectorInfo();
  }

  function startDetectLoop() {
    stopDetectLoop();
    if (!state.detector || state.detectorKind === 'manual') return;
    state.detectTimer = global.setInterval(runDetectTick, state.config.detectIntervalMs);
  }

  function stopDetectLoop() {
    if (state.detectTimer) {
      global.clearInterval(state.detectTimer);
      state.detectTimer = null;
    }
    state.detectBusy = false;
  }

  function runDetectTick() {
    var video = els.video;
    if (!state.detector || state.detectBusy || !state.stream || state.dragging) return;
    if (!video.videoWidth || !video.videoHeight) return;

    state.detectBusy = true;
    var scale = Math.min(1, 480 / video.videoWidth);
    var w = Math.max(1, Math.round(video.videoWidth * scale));
    var h = Math.max(1, Math.round(video.videoHeight * scale));

    try {
      if (state.detectCanvas.width !== w || state.detectCanvas.height !== h) {
        state.detectCanvas.width = w;
        state.detectCanvas.height = h;
      }
      state.detectCtx.drawImage(video, 0, 0, w, h);
    } catch (err) {
      state.detectBusy = false;
      return;
    }

    // MediaPipe may throw synchronously before returning a detection result.
    Promise.resolve().then(function () {
      return state.detector.detect(state.detectCanvas, { x: video.videoWidth / w, y: video.videoHeight / h });
    })
      .then(function (boxes) {
        state.detectBusy = false;
        if (state.stopped || !state.stream) return;
        var limit = { width: video.videoWidth, height: video.videoHeight };
        var tracked = state.tracker.update((boxes || []).map(function (box) {
          return Detectors.clampBox(box, limit);
        }), Date.now());
        state.faces = mergeManual(tracked, limit);
        paint(video.videoWidth, video.videoHeight);
      })
      .catch(function (err) {
        state.detectBusy = false;
        var msg = err && err.message ? err.message : String(err);
        state.detectorTone = 'error';
        state.detectorNote = 'Phát hiện khuôn mặt đã dừng: ' + msg;
        updateDetectorInfo();
        Dom.setStatus('error', state.detectorNote);
        note('error', 'Bộ phát hiện lỗi, đã dừng vòng lặp phát hiện: ' + msg);
        stopDetectLoop();
      });
  }

  function mergeManual(tracked, limit) {
    var out = (tracked || []).slice();
    state.manualBoxes.forEach(function (box, index) {
      var clamped = Detectors.clampBox(box, limit);
      if (clamped.width > 4 && clamped.height > 4) {
        out.push({
          id: clamped.id || ('manual-' + (index + 1)),
          x: clamped.x,
          y: clamped.y,
          width: clamped.width,
          height: clamped.height,
          source: 'manual'
        });
      }
    });
    return out;
  }

  /* ---------- camera ---------- */

  function startCamera() {
    if (global.location && global.location.protocol === 'file:') {
      var msg = 'Trình duyệt chặn camera với giao thức file:. Hãy chạy "python serve.py" trong thư mục frontend rồi mở địa chỉ localhost.';
      Dom.setStatus('error', msg);
      note('error', msg);
      return Promise.reject(new Error('insecure-context'));
    }
    if (!global.navigator.mediaDevices || !global.navigator.mediaDevices.getUserMedia) {
      var unsupported = 'Trình duyệt này không hỗ trợ getUserMedia, không mở được camera.';
      Dom.setStatus('error', unsupported);
      return Promise.reject(new Error('no-getusermedia'));
    }

    return ensureDetector().then(function () {
      var deviceId = els.cameraSelect.value;
      var constraints = {
        audio: false,
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
      };
      Dom.setStatus('loading', 'Đang mở camera…');
      return global.navigator.mediaDevices.getUserMedia(constraints);
    }).then(function (stream) {
      stopCameraStreams();
      state.stream = stream;
      state.stopped = false;
      els.video.srcObject = stream;
      return els.video.play().catch(function () { /* một số trình duyệt chặn autoplay, vẫn hiển thị khi có tương tác */ });
    }).then(function () {
      return refreshCameraList();
    }).then(function () {
      var track = state.stream.getVideoTracks()[0];
      var settings = track && track.getSettings ? track.getSettings() : {};
      state.cameraLabel = settings.width && settings.height
        ? settings.width + '×' + settings.height
        : 'độ phân giải không báo';
      state.tracker.reset();
      state.predictions.clear();
      state.faces = [];
      state.manualBoxes = [];
      Dom.setCameraEnabled(true);
      syncStageAspect(els.video.videoWidth, els.video.videoHeight);
      startDetectLoop();
      startRequestLoop();
      startRenderLoop();
      updateDetectorInfo();
      var mode = state.detectorKind === 'manual'
        ? 'đánh dấu thủ công'
        : 'phát hiện tự động';
      Dom.setStatus('live', 'Camera đang chạy (' + mode + '). ' + (
        state.detectorKind === 'manual'
          ? 'Kéo chuột trên ảnh để tạo khung khuôn mặt.'
          : 'Đưa khuôn mặt vào khung hình để bắt đầu gửi ảnh.'
      ));
    }).catch(function (err) {
      var message = describeCameraError(err);
      if (err && err.message !== 'insecure-context' && err.message !== 'no-getusermedia') {
        Dom.setStatus('error', message);
        note('error', message);
      }
      throw err;
    });
  }

  function describeCameraError(err) {
    if (!err) return 'Không mở được camera.';
    var name = err.name || '';
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError' || name === 'SecurityError') {
      return 'Quyền camera bị chặn. Cho phép camera cho trang này (biểu tượng ổ khóa trên thanh địa chỉ) rồi thử lại.';
    }
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError' || name === 'OverconstrainedError') {
      return 'Không tìm thấy camera phù hợp. Thử đổi thiết bị trong danh sách camera.';
    }
    if (name === 'NotReadableError' || name === 'TrackStartError') {
      return 'Camera đang bị ứng dụng khác sử dụng. Đóng ứng dụng đó (Teams, Zoom, OBS…) rồi thử lại.';
    }
    if (name === 'OverconstrainedError') {
      return 'Thiết bị camera không đáp ứng được cấu hình yêu cầu.';
    }
    return 'Không mở được camera: ' + (err.message || String(err));
  }

  function stopCameraStreams() {
    if (state.stream) {
      state.stream.getTracks().forEach(function (track) { track.stop(); });
      state.stream = null;
    }
    if (els.video) {
      els.video.pause();
      els.video.srcObject = null;
    }
  }

  function stopCamera() {
    state.stopped = true;
    stopDetectLoop();
    stopRequestLoop();
    stopRenderLoop();
    stopCameraStreams();
    state.resumeAutoAfterHealth = false;
    state.faces = [];
    state.manualBoxes = [];
    state.predictions.clear();
    state.dragging = null;
    state.cameraLabel = '';
    if (state.tracker) state.tracker.reset();
    if (state.overlay) state.overlay.clear();
    Dom.renderFaces([]);
    Dom.setCameraEnabled(false);
    Dom.setStatus('idle', 'Camera đã tắt. Mọi luồng camera và bộ đếm đã được dọn dẹp.');
  }

  function refreshCameraList() {
    if (!global.navigator.mediaDevices || !global.navigator.mediaDevices.enumerateDevices) {
      return Promise.resolve();
    }
    return global.navigator.mediaDevices.enumerateDevices().then(function (devices) {
      var current = els.cameraSelect.value;
      var videos = devices.filter(function (d) { return d.kind === 'videoinput'; });
      els.cameraSelect.innerHTML = '';
      var fallback = document.createElement('option');
      fallback.value = '';
      fallback.textContent = videos.length > 1 ? 'Thiết bị mặc định' : 'Camera phía trước';
      els.cameraSelect.appendChild(fallback);
      videos.forEach(function (device, index) {
        var opt = document.createElement('option');
        opt.value = device.deviceId;
        opt.textContent = device.label || ('Camera ' + (index + 1));
        els.cameraSelect.appendChild(opt);
      });
      if (current && videos.some(function (d) { return d.deviceId === current; })) {
        els.cameraSelect.value = current;
      }
    }).catch(function () { /* không chặn luồng camera */ });
  }

  /* ---------- khung hình & vẽ ---------- */

  function syncStageAspect(width, height) {
    if (!width || !height || !els.stage) return;
    els.stage.style.aspectRatio = width + ' / ' + height;
  }

  function syncStaticAspect(width, height) {
    if (!width || !height || !els.staticStage) return;
    els.staticStage.style.aspectRatio = width + ' / ' + height;
  }

  function startRenderLoop() {
    stopRenderLoop();
    var tick = function () {
      if (!state.stream) return;
      paint(els.video.videoWidth, els.video.videoHeight);
      state.rafId = global.requestAnimationFrame(tick);
    };
    state.rafId = global.requestAnimationFrame(tick);
  }

  function stopRenderLoop() {
    if (state.rafId) {
      global.cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
  }

  function paint(videoW, videoH) {
    if (!state.overlay) return;
    syncStageAspect(videoW, videoH);
    state.overlay.syncSize(videoW, videoH);
    if (!state.config.showBoxes || !videoW || !videoH) {
      state.overlay.clear();
      return;
    }
    state.overlay.draw(state.faces.map(function (face) {
      return Object.assign({}, face, state.predictions.get(face.id) || {});
    }), { mirrored: state.config.mirror });
  }

  /* ---------- gửi ảnh + tọa độ ---------- */

  function startRequestLoop() {
    stopRequestLoop();
    var loop = function () {
      state.loopTimer = global.setTimeout(function () {
        if (els.swAuto.checked && state.stream && !state.requestBusy && state.faces.length) {
          analyzeOnce({ reason: 'auto' });
        }
        loop();
      }, Math.max(150, state.config.requestIntervalMs));
    };
    loop();
  }

  function stopRequestLoop() {
    if (state.loopTimer) {
      global.clearTimeout(state.loopTimer);
      state.loopTimer = null;
    }
    cancelRequest();
  }

  function cancelRequest() {
    if (state.abortController) {
      try { state.abortController.abort(); } catch (err) { /* bỏ qua */ }
      state.abortController = null;
    }
    state.requestBusy = false;
    Dom.setBusy(false);
  }

  function frameToBlob(canvas, type, quality) {
    return new Promise(function (resolve, reject) {
      if (typeof canvas.toBlob === 'function') {
        canvas.toBlob(function (blob) {
          if (blob) resolve(blob);
          else reject(new Error('Không mã hóa được ảnh JPEG trên trình duyệt này.'));
        }, type, quality);
        return;
      }
      try {
        var dataUrl = canvas.toDataURL(type, quality);
        var bytes = global.atob(dataUrl.split(',')[1]);
        var buffer = new global.Uint8Array(bytes.length);
        for (var i = 0; i < bytes.length; i += 1) buffer[i] = bytes.charCodeAt(i);
        resolve(new global.Blob([buffer], { type: type }));
      } catch (err) {
        reject(new Error('Trình duyệt không hỗ trợ xuất ảnh JPEG.'));
      }
    });
  }

  function sendFrame(source, boxes) {
    var srcW = source.naturalWidth || source.videoWidth || source.width;
    var srcH = source.naturalHeight || source.videoHeight || source.height;
    var maxW = Number(state.config.maxSendWidth) || 0;
    var scale = maxW > 0 && srcW > maxW ? maxW / srcW : 1;
    var outW = Math.max(1, Math.round(srcW * scale));
    var outH = Math.max(1, Math.round(srcH * scale));

    if (state.captureCanvas.width !== outW || state.captureCanvas.height !== outH) {
      state.captureCanvas.width = outW;
      state.captureCanvas.height = outH;
    }

    var drawStart = now();
    try {
      state.captureCtx.drawImage(source, 0, 0, outW, outH);
    } catch (err) {
      return Promise.reject(new Error('Không đọc được ảnh để gửi: ' + (err.message || String(err))));
    }
    var captureMs = now() - drawStart;

    var payload = boxes.map(function (face) {
      return {
        id: String(face.id),
        x: Math.round(face.x * scale),
        y: Math.round(face.y * scale),
        width: Math.round(face.width * scale),
        height: Math.round(face.height * scale)
      };
    }).filter(function (face) {
      return face.width > 1 && face.height > 1;
    }).map(function (face) {
      // API từ chối khung vượt ra ngoài ảnh, nên ép lại trong [0, outW/outH].
      var x = Math.max(0, Math.min(face.x, outW - 1));
      var y = Math.max(0, Math.min(face.y, outH - 1));
      return {
        id: face.id,
        x: x,
        y: y,
        width: Math.max(1, Math.min(face.width, outW - x)),
        height: Math.max(1, Math.min(face.height, outH - y))
      };
    });

    var controller = new global.AbortController();
    state.abortController = controller;
    state.requestBusy = true;
    Dom.setBusy(true);

    return frameToBlob(state.captureCanvas, 'image/jpeg', state.config.jpegQuality).then(function (blob) {
      return state.client.predict({
        blob: blob,
        faces: payload,
        width: outW,
        height: outH,
        captureMs: captureMs,
        signal: controller.signal
      }).then(function (res) {
        res.captureMs = captureMs;
        res.bytes = blob.size;
        res.sentFaces = payload.length;
        return res;
      });
    }).then(function (res) {
      if (state.abortController === controller) {
        state.abortController = null;
        state.requestBusy = false;
        Dom.setBusy(false);
      }
      return res;
    }, function (err) {
      if (state.abortController === controller) {
        state.abortController = null;
        state.requestBusy = false;
        Dom.setBusy(false);
      }
      throw err;
    });
  }

  function analyzeOnce(options) {
    var opts = options || {};
    var video = els.video;

    if (state.requestBusy) {
      return Promise.resolve(null);
    }
    if (!video.videoWidth || !video.videoHeight) {
      if (opts.reason !== 'auto') {
        Dom.setStatus('error', 'Chưa có hình ảnh nào từ camera. Hãy bật camera trước, hoặc dùng phần Phân tích ảnh tĩnh.');
      }
      return Promise.resolve(null);
    }
    if (!state.faces.length) {
      Dom.setStatus(opts.reason === 'auto' ? 'live' : 'warn', 'Chưa có khuôn mặt nào. ' + hintForDetector());
      return Promise.resolve(null);
    }

    Dom.setStatus('loading', 'Đang gửi ' + state.faces.length + ' khuôn mặt…');
    return sendFrame(video, state.faces).then(function (res) {
      applyResult(res);
      return res;
    }).catch(function (err) {
      handleError(err, state.faces.length);
      return null;
    });
  }

  function hintForDetector() {
    return state.detectorKind === 'manual'
      ? 'Kéo chuột trên khung hình để tạo khung khuôn mặt, rồi nhấn “Phân tích 1 lần”.'
      : 'Đưa khuôn mặt vào khung hình và chờ một nhịp phát hiện.';
  }

  function applyResult(res) {
    state.failures = 0;
    state.predictions = new Map();
    res.predictions.forEach(function (prediction) {
      state.predictions.set(prediction.id, prediction);
    });

    var gap = now() - state.lastAnalyzeAt;
    state.lastAnalyzeAt = now();
    if (gap > 0 && gap < 5000) {
      state.analyzeGaps.push(gap);
      if (state.analyzeGaps.length > 20) state.analyzeGaps.shift();
    }
    var avgGap = state.analyzeGaps.length
      ? state.analyzeGaps.reduce(function (a, b) { return a + b; }, 0) / state.analyzeGaps.length
      : 0;

    var totalMs = res.captureMs + res.timings.roundTripMs;
    state.stats = {
      roundTripMs: res.timings.roundTripMs,
      serverMs: res.totalInferenceMs,
      totalMs: totalMs,
      sentFaces: res.sentFaces,
      receivedFaces: res.predictions.length,
      width: res.image.width,
      height: res.image.height,
      bytes: res.bytes,
      fps: state.analyzeGaps.length >= 2 && avgGap > 0 ? 1000 / avgGap : null
    };
    Dom.setStats(state.stats);
    Dom.renderFaces(state.faces.map(function (face) {
      return Object.assign({}, face, state.predictions.get(face.id) || {
        labelText: 'Chưa có kết quả',
        confidence: NaN,
        inferenceMs: NaN
      });
    }));
    paint(els.video.videoWidth, els.video.videoHeight);

    pushLog({
      time: clockNow(),
      faces: res.sentFaces,
      rtt: res.timings.roundTripMs,
      server: res.totalInferenceMs,
      status: 'OK' + (res.requestId ? ' #' + res.requestId.slice(0, 8) : '')
    });

    var summary = res.predictions.slice(0, 3).map(function (p) {
      return p.labelText + ' ' + Ui.formatPercent(p.confidence);
    });
    Dom.setStatus('ok',
      'Nhận ' + res.predictions.length + '/' + res.sentFaces + ' kết quả · toàn trình ' + Ui.formatMs(totalMs) +
      ' · server ' + Ui.formatMs(res.totalInferenceMs) + ' ms' +
      (summary.length ? ' · ' + summary.join(', ') : ' · API không trả dự đoán nào'));

    if (res.requestId) {
      els.configHint.textContent = 'request_id gần nhất: ' + res.requestId;
    }
  }

  function handleError(err, faceCount) {
    if (err && (err.name === 'AbortError' || err.kind === 'aborted')) return;
    state.failures += 1;
    var message = err && err.message ? err.message : String(err);
    Dom.setStatus('error', message);
    pushLog({
      time: clockNow(),
      faces: faceCount,
      rtt: null,
      server: null,
      status: 'Lỗi: ' + message.slice(0, 44)
    });

    var kind = err && err.kind ? err.kind : 'unknown';
    var unavailable = kind === 'network' || kind === 'timeout' ||
      (kind === 'http' && err.status >= 500);
    if (unavailable && state.failures >= 3) {
      state.resumeAutoAfterHealth = state.resumeAutoAfterHealth || els.swAuto.checked;
      els.swAuto.checked = false;
      Dom.setHealth(false, message);
      note('error', 'Chưa kết nối được API. Tạm dừng gửi ảnh và tự thử lại khi API sẵn sàng.');
    } else {
      note('error', message);
    }
  }

  function pushLog(entry) {
    state.logEntries.unshift(entry);
    if (state.logEntries.length > 25) state.logEntries.length = 25;
    Dom.renderLog(state.logEntries);
  }

  /* ---------- đánh dấu thủ công ---------- */

  function setupManualDrawing() {
    var canvas = els.overlay;
    if (!canvas) return;

    canvas.addEventListener('pointerdown', function (event) {
      if (!state.stream || event.button === 2) return;
      if (state.detectorKind !== 'manual') {
        note('info', 'Đang dùng phát hiện tự động nên không vẽ khung thủ công. Đổi “Bộ phát hiện” sang Đánh dấu thủ công nếu muốn tự kéo khung.');
        return;
      }
      var point = state.overlay.pointToImage(event.clientX, event.clientY, els.video.videoWidth, els.video.videoHeight, state.config.mirror);
      if (!point) return;
      if (event.altKey) {
        state.manualBoxes = state.manualBoxes.filter(function (box) {
          var inside = point.x >= box.x && point.x <= box.x + box.width &&
            point.y >= box.y && point.y <= box.y + box.height;
          return !inside;
        });
        refreshManualFaces();
        return;
      }
      try { canvas.setPointerCapture(event.pointerId); } catch (err) { /* bỏ qua */ }
      state.dragging = { start: point, current: point };
    });

    canvas.addEventListener('pointermove', function (event) {
      if (!state.dragging) return;
      var point = state.overlay.pointToImage(event.clientX, event.clientY, els.video.videoWidth, els.video.videoHeight, state.config.mirror);
      if (!point) return;
      state.dragging.current = point;
      paint(els.video.videoWidth, els.video.videoHeight);
    });

    var finish = function (event) {
      if (!state.dragging) return;
      var drag = state.dragging;
      state.dragging = null;
      try { canvas.releasePointerCapture(event.pointerId); } catch (err) { /* bỏ qua */ }
      var x = Math.min(drag.start.x, drag.current.x);
      var y = Math.min(drag.start.y, drag.current.y);
      var w = Math.abs(drag.current.x - drag.start.x);
      var h = Math.abs(drag.current.y - drag.start.y);
      paint(els.video.videoWidth, els.video.videoHeight);
      if (w < 14 || h < 14) return;
      state.manualBoxes.push({ x: x, y: y, width: w, height: h });
      refreshManualFaces();
      Dom.setStatus('live', 'Đã thêm khung thủ công (' + state.manualBoxes.length + ' khung). Nhấn “Phân tích 1 lần” hoặc bật Tự động.');
    };

    canvas.addEventListener('pointerup', finish);
    canvas.addEventListener('pointercancel', function () {
      state.dragging = null;
      paint(els.video.videoWidth, els.video.videoHeight);
    });
    canvas.addEventListener('contextmenu', function (event) {
      if (state.detectorKind === 'manual') event.preventDefault();
    });
  }

  function refreshManualFaces() {
    var limit = { width: els.video.videoWidth, height: els.video.videoHeight };
    state.faces = mergeManual([], limit);
    Dom.renderFaces([]);
    paint(limit.width, limit.height);
    if (!state.manualBoxes.length) {
      Dom.setStatus('live', 'Chưa có khung nào. Kéo chuột trên hình để tạo khung khuôn mặt.');
    }
  }

  /* ---------- ảnh tĩnh ---------- */

  function analyzeFile(file) {
    if (!file) return Promise.resolve(null);
    if (state.requestBusy) {
      note('warn', 'Đang có request đang chạy, chờ xong rồi thử lại.');
      return Promise.resolve(null);
    }
    return ensureDetector().then(function () {
      return loadImageFile(file);
    }).then(function (image) {
      els.staticEmpty.hidden = true;
      els.staticImage.src = image.src;
      var w = image.naturalWidth;
      var h = image.naturalHeight;
      syncStaticAspect(w, h);
      state.staticOverlay.syncSize(w, h);
      return detectInImage(image, w, h).then(function (boxes) {
        state.staticOverlay.draw(boxes, { mirrored: false });
        Dom.setStatus('loading', 'Đang gửi ảnh tĩnh ' + w + '×' + h + ' với ' + boxes.length + ' khuôn mặt…');
        return sendFrame(image, boxes).then(function (res) {
          Dom.setStats({
            roundTripMs: res.timings.roundTripMs,
            serverMs: res.totalInferenceMs,
            totalMs: res.captureMs + res.timings.roundTripMs,
            sentFaces: res.sentFaces,
            receivedFaces: res.predictions.length,
            width: res.image.width,
            height: res.image.height,
            bytes: res.bytes,
            fps: state.stats.fps
          });
          var byId = new Map();
          res.predictions.forEach(function (p) { byId.set(p.id, p); });
          state.staticOverlay.draw(boxes.map(function (face) {
            return Object.assign({}, face, byId.get(face.id) || {});
          }), { mirrored: false });
          Dom.renderFaces(boxes.map(function (face) {
            return Object.assign({}, face, byId.get(face.id) || {
              labelText: 'Chưa có kết quả',
              confidence: NaN,
              inferenceMs: NaN
            });
          }));
          pushLog({
            time: clockNow(),
            faces: res.sentFaces,
            rtt: res.timings.roundTripMs,
            server: res.totalInferenceMs,
            status: 'Ảnh tĩnh' + (res.requestId ? ' #' + res.requestId.slice(0, 8) : '')
          });
          state.failures = 0;
          Dom.setStatus('ok', 'Ảnh tĩnh: nhận ' + res.predictions.length + '/' + res.sentFaces +
            ' kết quả · gửi–nhận ' + Ui.formatMs(res.timings.roundTripMs) +
            ' ms · server ' + Ui.formatMs(res.totalInferenceMs) + ' ms');
          return res;
        });
      });
    }).catch(function (err) {
      var message = err && err.message ? err.message : String(err);
      Dom.setStatus('error', 'Lỗi xử lý ảnh: ' + message);
      note('error', 'Lỗi xử lý ảnh: ' + message);
      return null;
    });
  }

  function loadImageFile(file) {
    var type = file.type || '';
    if (type && type !== 'image/jpeg' && type !== 'image/png' && type !== 'image/jpg') {
      return Promise.reject(new Error('Chỉ hỗ trợ ảnh JPEG hoặc PNG (tệp đang chọn: ' + type + ').'));
    }
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onerror = function () { reject(new Error('Không đọc được tệp ảnh.')); };
      reader.onload = function () {
        var img = new global.Image();
        img.onload = function () { resolve(img); };
        img.onerror = function () { reject(new Error('Tệp không phải ảnh hợp lệ.')); };
        img.src = reader.result;
      };
      reader.readAsDataURL(file);
    });
  }

  function detectInImage(image, w, h) {
    var limit = { width: w, height: h };
    if (state.detector && state.detectorKind !== 'manual') {
      var scale = Math.min(1, 640 / w);
      var dw = Math.max(1, Math.round(w * scale));
      var dh = Math.max(1, Math.round(h * scale));
      state.detectCanvas.width = dw;
      state.detectCanvas.height = dh;
      state.detectCtx.drawImage(image, 0, 0, dw, dh);
      return Promise.resolve().then(function () {
        return state.detector.detect(state.detectCanvas, { x: w / dw, y: h / dh });
      })
        .then(function (boxes) {
          var tracked = state.tracker.update((boxes || []).map(function (box) {
            return Detectors.clampBox(box, limit);
          }), Date.now());
          if (tracked.length) return tracked;
          note('warn', 'Không tự phát hiện được khuôn mặt trong ảnh, đã dùng khung vuông ở giữa ảnh. Kiểm tra lại tọa độ.');
          return centerBox(w, h, state.detectorKind);
        })
        .catch(function (err) {
          note('warn', 'Phát hiện khuôn mặt trong ảnh thất bại (' + (err.message || err) + '), dùng khung vuông ở giữa ảnh.');
          return centerBox(w, h, state.detectorKind);
        });
    }
    note('info', 'Chưa có bộ phát hiện tự động: đã dùng khung vuông ở giữa ảnh, hãy chỉnh nếu khuôn mặt lệch.');
    return Promise.resolve(centerBox(w, h, 'manual'));
  }

  function centerBox(w, h, source) {
    var size = Math.round(Math.min(w, h) * 0.6);
    return [{
      id: 'manual-1',
      x: Math.round((w - size) / 2),
      y: Math.round((h - size) / 2),
      width: size,
      height: size,
      source: source === 'manual' ? 'manual' : 'detector',
      fallback: true
    }];
  }

  /* ---------- sự kiện ---------- */

  function bindEvents() {
    els.btnCamera.addEventListener('click', function () {
      if (state.stream) {
        stopCamera();
        return;
      }
      startCamera().catch(function () { /* thông báo đã hiển thị */ });
    });

    els.btnOnce.addEventListener('click', function () {
      analyzeOnce({ reason: 'manual' });
    });

    els.btnClearManual.addEventListener('click', function () {
      state.manualBoxes = [];
      refreshManualFaces();
      note('info', 'Đã xóa các khung thủ công.');
    });

    els.btnHealth.addEventListener('click', function () {
      checkHealth();
    });

    els.btnClearLog.addEventListener('click', function () {
      state.logEntries = [];
      Dom.renderLog([]);
    });

    els.cameraSelect.addEventListener('change', function () {
      if (state.stream) {
        stopCamera();
        global.setTimeout(function () {
          startCamera().catch(function () { /* thông báo đã hiển thị */ });
        }, 60);
      }
    });

    els.swBoxes.addEventListener('change', function () {
      state.config.showBoxes = els.swBoxes.checked;
      paint(els.video.videoWidth, els.video.videoHeight);
    });

    els.swAuto.addEventListener('change', function () {
      state.resumeAutoAfterHealth = false;
    });

    els.swMirror.addEventListener('change', function () {
      state.config.mirror = els.swMirror.checked;
      els.stage.classList.toggle('mirrored', state.config.mirror);
      paint(els.video.videoWidth, els.video.videoHeight);
    });

    els.rangeInterval.addEventListener('input', function () {
      var value = Number(els.rangeInterval.value);
      els.valInterval.textContent = value;
      state.config.requestIntervalMs = value;
      if (state.stream) startRequestLoop();
    });

    els.rangeQuality.addEventListener('input', function () {
      var value = Number(els.rangeQuality.value);
      els.valQuality.textContent = value.toFixed(2);
      state.config.jpegQuality = value;
    });

    els.btnSaveConfig.addEventListener('click', function () {
      saveConfig({
        apiBaseUrl: els.inputApi.value.trim(),
        detector: els.selectDetector.value,
        maxSendWidth: Number(els.inputMaxWidth.value) || 0,
        timeoutMs: Number(els.inputTimeout.value) || 8000,
        mirror: els.swMirror.checked,
        showBoxes: els.swBoxes.checked
      });
    });

    els.btnResetConfig.addEventListener('click', function () {
      state.config = Config.reset();
      state.detectorWanted = null;
      state.detector = null;
      applyConfigToForm();
      ensureDetector().then(function () {
        if (state.stream) startDetectLoop();
      }).catch(function () { /* đã hiển thị */ });
      checkHealth();
      note('info', 'Đã đưa cấu hình về mặc định.');
    });

    els.selectDetector.addEventListener('change', function () {
      saveConfig({ detector: els.selectDetector.value }, true);
    });

    els.inputFile.addEventListener('change', function () {
      var file = els.inputFile.files && els.inputFile.files[0];
      els.btnAnalyzeFile.disabled = !file;
      if (file) analyzeFile(file);
    });

    els.btnAnalyzeFile.addEventListener('click', function () {
      var file = els.inputFile.files && els.inputFile.files[0];
      if (file) analyzeFile(file);
    });

    els.staticImage.addEventListener('load', function () {
      var w = els.staticImage.naturalWidth;
      var h = els.staticImage.naturalHeight;
      syncStaticAspect(w, h);
      if (state.staticOverlay) state.staticOverlay.syncSize(w, h);
    });

    global.addEventListener('beforeunload', function () {
      stopCamera();
    });

    global.addEventListener('pagehide', function () {
      stopCamera();
    });

    global.addEventListener('resize', function () {
      if (state.stream) paint(els.video.videoWidth, els.video.videoHeight);
    });

    if (global.document && global.document.addEventListener) {
      global.document.addEventListener('visibilitychange', function () {
        if (global.document.hidden && state.requestBusy) {
          cancelRequest();
          Dom.setStatus('live', 'Đã hủy request đang chạy vì chuyển sang tab khác.');
        }
      });
    }
  }

  function setupCanvases() {
    state.captureCanvas = document.createElement('canvas');
    state.captureCtx = state.captureCanvas.getContext('2d');
    state.detectCanvas = document.createElement('canvas');
    state.detectCtx = state.detectCanvas.getContext('2d');
    state.overlay = new Ui.Overlay(els.overlay);
    state.staticOverlay = new Ui.Overlay(els.staticOverlay);
  }

  function showSecureBanner() {
    var insecure = (global.location && global.location.protocol === 'file:') || !global.isSecureContext;
    if (els.secureBanner) els.secureBanner.hidden = !insecure;
  }

  function init() {
    Dom.init();
    els = Dom.els;
    applyConfigToForm();
    setupCanvases();
    setupManualDrawing();
    bindEvents();
    showSecureBanner();
    Dom.setCameraEnabled(false);
    Dom.setStats({});
    Dom.renderFaces([]);
    Dom.renderLog([]);
    Dom.setStatus('idle', 'Nhấn “Bật camera” để bắt đầu.');
    state.detectorNote = Detectors.isNativeSupported()
      ? 'Trình duyệt hỗ trợ FaceDetector API, sẽ dùng bộ phát hiện này khi bật camera.'
      : 'Trình duyệt không có FaceDetector API: có thể dùng MediaPipe từ CDN hoặc đánh dấu thủ công.';
    state.detectorTone = Detectors.isNativeSupported() ? 'ok' : 'warn';
    updateDetectorInfo();
    ensureDetector().catch(function () { /* đã hiển thị */ });
    checkHealth();
    if (state.healthTimer) global.clearInterval(state.healthTimer);
    state.healthTimer = global.setInterval(function () {
      if (state.stream) checkHealth(true);
    }, 5000);
    refreshCameraList();
  }

  if (global.document.readyState === 'loading') {
    global.document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  global.LAB3App = {
    state: state,
    init: init,
    analyzeOnce: analyzeOnce,
    startCamera: startCamera,
    stopCamera: stopCamera,
    checkHealth: checkHealth,
    saveConfig: saveConfig
  };
})(window);
