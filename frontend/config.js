/* Cấu hình mặc định của giao diện LAB3.
   Thứ tự ưu tiên: tham số trên URL -> localStorage -> window.LAB3_CONFIG -> DEFAULT_CONFIG. */
(function (global) {
  'use strict';

  var STORAGE_KEY = 'lab3.frontend.config.v1';

  var DEFAULT_CONFIG = {
    // The default frontend server proxies API calls on the same origin.
    // This avoids cross-origin camera-page requests being blocked by CORS.
    apiBaseUrl: '',
    healthPath: '/health',
    predictPath: '/api/v1/predict',
    requestIntervalMs: 400,
    detectIntervalMs: 160,
    jpegQuality: 0.85,
    maxSendWidth: 0,
    detector: 'auto',
    timeoutMs: 8000,
    maxFaces: 20,
    iouThreshold: 0.3,
    mirror: true,
    showBoxes: true,
    autoStart: true
  };

  var NUMERIC_KEYS = {
    requestIntervalMs: [150, 3000],
    detectIntervalMs: [60, 2000],
    jpegQuality: [0.3, 0.98],
    maxSendWidth: [0, 4096],
    timeoutMs: [500, 60000],
    maxFaces: [1, 64],
    iouThreshold: [0.05, 0.95]
  };

  var ENUM_KEYS = {
    detector: ['auto', 'native', 'mediapipe', 'manual']
  };

  function clampNumber(value, key, fallback) {
    if (value === null || value === undefined || value === '') return fallback;
    var n = Number(value);
    if (!isFinite(n)) return fallback;
    var range = NUMERIC_KEYS[key];
    if (range) n = Math.min(range[1], Math.max(range[0], n));
    return n;
  }

  function isPlainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function sanitize(raw, base) {
    var out = {};
    Object.keys(base).forEach(function (key) {
      var value = raw ? raw[key] : undefined;
      if (value === undefined || value === null) {
        out[key] = base[key];
        return;
      }
      if (typeof base[key] === 'number') {
        out[key] = clampNumber(value, key, base[key]);
      } else if (typeof base[key] === 'boolean') {
        out[key] = typeof value === 'string' ? value === 'true' || value === '1' : Boolean(value);
      } else if (ENUM_KEYS[key]) {
        var lowered = String(value).toLowerCase();
        out[key] = ENUM_KEYS[key].indexOf(lowered) >= 0 ? lowered : base[key];
      } else {
        out[key] = String(value).trim();
      }
    });
    return out;
  }

  function readStorage() {
    try {
      var raw = global.localStorage && global.localStorage.getItem(STORAGE_KEY);
      if (!raw) return {};
      var parsed = JSON.parse(raw);
      return isPlainObject(parsed) ? parsed : {};
    } catch (err) {
      return {};
    }
  }

  function readQuery() {
    var out = {};
    try {
      var params = new URLSearchParams(global.location && global.location.search ? global.location.search : '');
      var map = {
        api: 'apiBaseUrl',
        detector: 'detector',
        interval: 'requestIntervalMs',
        quality: 'jpegQuality',
        maxWidth: 'maxSendWidth'
      };
      Object.keys(map).forEach(function (queryKey) {
        if (params.has(queryKey)) out[map[queryKey]] = params.get(queryKey);
      });
      if (params.has('mirror')) out.mirror = params.get('mirror') !== '0';
    } catch (err) {
      return {};
    }
    return out;
  }

  function resolve() {
    var base = Object.assign({}, DEFAULT_CONFIG, isPlainObject(global.LAB3_CONFIG) ? global.LAB3_CONFIG : {});
    var merged = Object.assign({}, base, readStorage(), readQuery());
    return sanitize(merged, base);
  }

  var Config = {
    DEFAULTS: DEFAULT_CONFIG,
    STORAGE_KEY: STORAGE_KEY,

    resolve: resolve,

    load: function () {
      return resolve();
    },

    save: function (patch) {
      var current = readStorage();
      var next = Object.assign({}, current, patch || {});
      var base = Object.assign({}, DEFAULT_CONFIG, isPlainObject(global.LAB3_CONFIG) ? global.LAB3_CONFIG : {});
      var clean = sanitize(next, base);
      var persisted = {};
      Object.keys(base).forEach(function (key) {
        persisted[key] = clean[key];
      });
      try {
        if (global.localStorage) global.localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
      } catch (err) {
        return { ok: false, config: clean, error: err };
      }
      return { ok: true, config: clean };
    },

    reset: function () {
      try {
        if (global.localStorage) global.localStorage.removeItem(STORAGE_KEY);
      } catch (err) {
        /* localStorage bị chặn, không sao */
      }
      return resolve();
    },

    joinUrl: function (baseUrl, path) {
      var base = String(baseUrl || '').trim();
      if (!base) return path;
      return base.replace(/\/+$/, '') + path;
    },

    predictUrl: function (config) {
      return Config.joinUrl(config.apiBaseUrl, config.predictPath);
    },

    healthUrl: function (config) {
      return Config.joinUrl(config.apiBaseUrl, config.healthPath);
    }
  };

  global.LAB3Config = Config;
})(window);
