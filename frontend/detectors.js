/* Phát hiện khuôn mặt phía trình duyệt.
   Ưu tiên FaceDetector API gốc của trình duyệt (không tải thêm gì).
   Khi không có: MediaPipe qua CDN nếu người dùng chọn, hoặc đánh dấu thủ công. */
(function (global) {
  'use strict';

  var FACE_DETECTOR_GLOBAL = 'FaceDetector';

  function iou(a, b) {
    var ax2 = a.x + a.width;
    var ay2 = a.y + a.height;
    var bx2 = b.x + b.width;
    var by2 = b.y + b.height;
    var w = Math.min(ax2, bx2) - Math.max(a.x, b.x);
    var h = Math.min(ay2, by2) - Math.max(a.y, b.y);
    if (w <= 0 || h <= 0) return 0;
    var inter = w * h;
    var union = a.width * a.height + b.width * b.height - inter;
    return union > 0 ? inter / union : 0;
  }

  function area(box) {
    return Math.max(0, box.width) * Math.max(0, box.height);
  }

  function clampBox(box, limit) {
    var maxW = limit.width;
    var maxH = limit.height;
    var x = Math.max(0, Math.min(box.x, maxW - 1));
    var y = Math.max(0, Math.min(box.y, maxH - 1));
    var w = Math.max(1, Math.min(box.width, maxW - x));
    var h = Math.max(1, Math.min(box.height, maxH - y));
    return { x: x, y: y, width: w, height: h };
  }

  function Tracker(options) {
    var opts = options || {};
    this.iouThreshold = opts.iouThreshold === undefined ? 0.3 : opts.iouThreshold;
    this.maxFaces = opts.maxFaces || 20;
    this.smoothing = opts.smoothing === undefined ? 0.55 : opts.smoothing;
    this.staleMs = opts.staleMs || 700;
    this.entries = new Map();
    this.nextId = 1;
  }

  Tracker.prototype.reset = function () {
    this.entries.clear();
    this.nextId = 1;
  };

  Tracker.prototype._pickId = function (box, used) {
    var bestId = null;
    var bestScore = 0;
    var self = this;
    this.entries.forEach(function (entry, id) {
      if (used.has(id)) return;
      var score = iou(entry.box, box);
      if (score > bestScore) {
        bestScore = score;
        bestId = id;
      }
    });
    return bestScore >= self.iouThreshold ? bestId : null;
  };

  Tracker.prototype.update = function (rawBoxes, now) {
    var timestamp = now === undefined ? Date.now() : now;
    var boxes = [];
    var self = this;

    (rawBoxes || []).slice(0, this.maxFaces).forEach(function (raw) {
      var box = {
        x: Number(raw.x) || 0,
        y: Number(raw.y) || 0,
        width: Math.max(1, Number(raw.width) || 1),
        height: Math.max(1, Number(raw.height) || 1)
      };
      if (raw.confidence !== undefined && raw.confidence !== null) {
        box.detectorConfidence = Number(raw.confidence) || 0;
      }
      var used = new Set(boxes.map(function (b) { return b.id; }));
      var id = raw.id !== undefined && raw.id !== null
        ? String(raw.id)
        : self._pickId(box, used);
      if (id === null) {
        id = 'face-' + self.nextId;
        self.nextId += 1;
      }
      var previous = self.entries.get(id);
      if (previous) {
        var k = self.smoothing;
        box = {
          x: previous.box.x + (box.x - previous.box.x) * (1 - k),
          y: previous.box.y + (box.y - previous.box.y) * (1 - k),
          width: previous.box.width + (box.width - previous.box.width) * (1 - k),
          height: previous.box.height + (box.height - previous.box.height) * (1 - k)
        };
      }
      box.id = id;
      box.source = raw.source || (previous && previous.source) || 'detector';
      box.lastSeen = timestamp;
      box.confirmed = true;
      self.entries.set(id, { box: box, lastSeen: timestamp });
      boxes.push(box);
    });

    this.entries.forEach(function (entry, id) {
      if (timestamp - entry.lastSeen > self.staleMs) self.entries.delete(id);
    });
    return boxes;
  };

  Tracker.prototype.count = function () {
    return this.entries.size;
  };

  function NativeDetector(options) {
    var opts = options || {};
    this.name = 'native';
    this.displayName = 'FaceDetector API của trình duyệt';
    this.fastMode = opts.fastMode !== false;
    this.maxDetectedFaces = opts.maxDetectedFaces || 20;
    this._Ctor = global[FACE_DETECTOR_GLOBAL];
  }

  NativeDetector.isSupported = function () {
    return typeof global[FACE_DETECTOR_GLOBAL] === 'function';
  };

  NativeDetector.prototype.init = function () {
    if (!NativeDetector.isSupported()) {
      return Promise.reject(new Error('Trình duyệt không hỗ trợ FaceDetector API.'));
    }
    try {
      this.detector = new global[FACE_DETECTOR_GLOBAL]({ fastMode: this.fastMode, maxDetectedFaces: this.maxDetectedFaces });
    } catch (err) {
      try {
        this.detector = new global[FACE_DETECTOR_GLOBAL]();
      } catch (err2) {
        return Promise.reject(err2 instanceof Error ? err2 : new Error(String(err2)));
      }
    }
    return Promise.resolve(this);
  };

  NativeDetector.prototype.detect = function (source, scale) {
    if (!this.detector) return Promise.resolve([]);
    var factor = scale && scale.x ? scale : { x: 1, y: 1 };
    return this.detector.detect(source).then(function (faces) {
      return (faces || []).map(function (face) {
        var box = face.boundingBox;
        return {
          x: box.x * factor.x,
          y: box.y * factor.y,
          width: box.width * factor.x,
          height: box.height * factor.y,
          source: 'detector',
          detectorConfidence: 1
        };
      });
    });
  };

  NativeDetector.prototype.close = function () {
    this.detector = null;
  };

  var MEDIAPIPE_MODULE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
  var MEDIAPIPE_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
  var MEDIAPIPE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

  function MediaPipeDetector(options) {
    var opts = options || {};
    this.name = 'mediapipe';
    this.displayName = 'MediaPipe FaceDetector (tải từ CDN)';
    this.moduleUrl = opts.moduleUrl || MEDIAPIPE_MODULE;
    this.wasmUrl = opts.wasmUrl || MEDIAPIPE_WASM;
    this.modelUrl = opts.modelUrl || MEDIAPIPE_MODEL;
    this.minConfidence = opts.minConfidence === undefined ? 0.5 : opts.minConfidence;
    this.detector = null;
    this.delegate = null;
  }

  MediaPipeDetector.prototype.init = function () {
    var self = this;
    if (typeof global.importScripts === 'function') {
      return Promise.reject(new Error('MediaPipe cần được nạp bằng dynamic import, hãy dùng máy chủ tĩnh (http).'));
    }
    return import(/* webpackIgnore: true */ this.moduleUrl).then(function (vision) {
      return vision.FilesetResolver.forVisionTasks(self.wasmUrl).then(function (fileset) {
        function create(delegate) {
          return Promise.resolve().then(function () {
            return vision.FaceDetector.createFromOptions(fileset, {
              baseOptions: { modelAssetPath: self.modelUrl, delegate: delegate },
              runningMode: 'IMAGE',
              minDetectionConfidence: self.minConfidence
            });
          }).then(function (detector) {
            self.delegate = delegate;
            return detector;
          });
        }
        return create('GPU').catch(function () {
          // Keep automatic detection available when the GPU delegate fails.
          return create('CPU');
        });
      });
    }).then(function (detector) {
      self.detector = detector;
      return self;
    }).catch(function (err) {
      throw new Error('Không nạp được MediaPipe từ CDN: ' + (err && err.message ? err.message : String(err)));
    });
  };

  MediaPipeDetector.prototype.detect = function (source, scale) {
    if (!this.detector) return Promise.resolve([]);
    var factor = scale && scale.x ? scale : { x: 1, y: 1 };
    var result = this.detector.detect(source);
    return Promise.resolve(result).then(function (out) {
      var detections = (out && out.detections) || [];
      return detections.map(function (item) {
        var box = item.boundingBox || {};
        var originX = Number(box.originX) || 0;
        var originY = Number(box.originY) || 0;
        var conf = item.categories && item.categories.length
          ? Number(item.categories[0].score) || 0
          : null;
        return {
          x: originX * factor.x,
          y: originY * factor.y,
          width: (Number(box.width) || 0) * factor.x,
          height: (Number(box.height) || 0) * factor.y,
          source: 'detector',
          detectorConfidence: conf
        };
      }).filter(function (box) {
        return box.width > 2 && box.height > 2;
      });
    });
  };

  MediaPipeDetector.prototype.close = function () {
    if (this.detector && typeof this.detector.close === 'function') {
      try { this.detector.close(); } catch (err) { /* bỏ qua */ }
    }
    this.detector = null;
  };

  function ManualDetector() {
    this.name = 'manual';
    this.displayName = 'Đánh dấu thủ công trên khung hình';
    this.boxes = [];
  }

  ManualDetector.prototype.init = function () {
    return Promise.resolve(this);
  };

  ManualDetector.prototype.detect = function () {
    return Promise.resolve(this.boxes.map(function (box) {
      return Object.assign({}, box, { source: 'manual' });
    }));
  };

  ManualDetector.prototype.add = function (box) {
    var next = clampBox(box, { width: this.limitWidth, height: this.limitHeight });
    if (next.width < 8 || next.height < 8) return null;
    var id = 'manual-' + (this.boxes.length + 1) + '-' + Math.round(next.x) + 'x' + Math.round(next.y);
    next.id = id;
    next.source = 'manual';
    this.boxes.push(next);
    return id;
  };

  ManualDetector.prototype.removeAt = function (x, y) {
    for (var i = this.boxes.length - 1; i >= 0; i -= 1) {
      var box = this.boxes[i];
      if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) {
        this.boxes.splice(i, 1);
        return true;
      }
    }
    return false;
  };

  ManualDetector.prototype.clear = function () {
    this.boxes.length = 0;
  };

  ManualDetector.prototype.close = function () {
    this.clear();
  };

  function createDetector(kind, options) {
    var opts = options || {};
    var normalized = (kind || 'auto').toLowerCase();
    var detector;
    if (normalized === 'native') {
      detector = new NativeDetector(opts);
    } else if (normalized === 'mediapipe') {
      detector = new MediaPipeDetector(opts);
    } else {
      detector = new ManualDetector(opts);
    }
    detector.kind = normalized;
    return detector;
  }

  function resolveAuto(options) {
    if (NativeDetector.isSupported()) return Promise.resolve('native');
    // Most browsers do not expose the experimental FaceDetector API. In that
    // case prefer the existing MediaPipe implementation so the default mode
    // still detects multiple faces automatically. Its init failure already
    // falls back to manual marking in app.js.
    if (global.navigator && global.navigator.onLine !== false) return Promise.resolve('mediapipe');
    return Promise.resolve('manual');
  }

  global.LAB3Detectors = {
    Tracker: Tracker,
    NativeDetector: NativeDetector,
    MediaPipeDetector: MediaPipeDetector,
    ManualDetector: ManualDetector,
    createDetector: createDetector,
    resolveAuto: resolveAuto,
    isNativeSupported: NativeDetector.isSupported,
    iou: iou,
    clampBox: clampBox,
    area: area
  };
})(window);
