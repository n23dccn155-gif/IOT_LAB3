/* Lớp giao tiếp với API FastAPI của LAB3.
   POST /api/v1/predict  multipart/form-data: image (JPEG/PNG), faces (JSON array)
   GET  /health */
(function (global) {
  'use strict';

  var LABELS = ['without_mask', 'with_mask', 'mask_weared_incorrect'];

  var LABEL_TEXT = {
    without_mask: 'Không đeo khẩu trang',
    with_mask: 'Đeo khẩu trang đúng',
    mask_weared_incorrect: 'Đeo khẩu trang sai'
  };

  function ApiError(message, kind, extra) {
    var err = Error.call(this, message);
    this.name = 'ApiError';
    this.message = message;
    this.kind = kind || 'unknown';
    if (extra) Object.keys(extra).forEach(function (k) { this[k] = extra[k]; }, this);
    if (Error.captureStackTrace) Error.captureStackTrace(this, ApiError);
    else this.stack = new Error(message).stack;
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  function now() {
    return (global.performance && global.performance.now) ? global.performance.now() : Date.now();
  }

  function toNumber(value, fallback) {
    var n = Number(value);
    return isFinite(n) ? n : (fallback === undefined ? null : fallback);
  }

  function extractDetail(payload, status) {
    if (payload && typeof payload === 'object') {
      var detail = payload.detail;
      if (typeof detail === 'string' && detail) return detail;
      if (Array.isArray(detail) && detail.length) {
        return detail.map(function (item) {
          if (typeof item === 'string') return item;
          var loc = Array.isArray(item && item.loc) ? item.loc.join('.') : '';
          var msg = (item && (item.msg || item.message)) || 'Giá trị không hợp lệ';
          return loc ? loc + ': ' + msg : msg;
        }).join('; ');
      }
      if (typeof payload.message === 'string' && payload.message) return payload.message;
    }
    return 'Máy chủ trả về lỗi HTTP ' + status + '.';
  }

  function normalizePrediction(raw, index) {
    var box = (raw && raw.box) || {};
    return {
      id: String((raw && raw.id) !== undefined && raw.id !== null ? raw.id : 'face-' + index),
      label: String(raw && raw.label),
      labelText: LABEL_TEXT[raw && raw.label] || String(raw && raw.label || 'Không rõ'),
      knownLabel: LABELS.indexOf(String(raw && raw.label)) >= 0,
      confidence: toNumber(raw && raw.confidence, 0),
      box: {
        x: toNumber(box.x, 0) || 0,
        y: toNumber(box.y, 0) || 0,
        width: toNumber(box.width, 0) || 0,
        height: toNumber(box.height, 0) || 0
      },
      inferenceMs: toNumber(raw && raw.inference_ms, 0) || 0
    };
  }

function ApiClient(config) {
    this.config = config;
  }

  ApiClient.prototype._request = function (url, options) {
    var opts = options || {};
    var controller = new (global.AbortController || function () { this.abort = function () {}; })();
    var external = opts.signal;
    var timedOut = false;
    var aborted = false;
    var limit = Number(opts.timeoutMs || this.config.timeoutMs || 8000);

    if (external && external.aborted) aborted = true;
    var onExternalAbort = function () {
      aborted = true;
      controller.abort();
    };
    if (external && external.addEventListener) external.addEventListener('abort', onExternalAbort);

    var timer = null;
    if (limit > 0) {
      timer = global.setTimeout(function () {
        timedOut = true;
        controller.abort();
      }, limit);
    }

    var headers = { Accept: 'application/json' };
    if (opts.headers) Object.keys(opts.headers).forEach(function (k) { headers[k] = opts.headers[k]; });

    var started = now();

    var body = opts.body;
    if (body !== undefined && body !== null && !(body instanceof global.FormData) && !(body instanceof global.Blob)) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(body);
    }

    var fetchPromise;
    try {
      fetchPromise = global.fetch(url, {
        method: opts.method || 'GET',
        headers: headers,
        body: body,
        signal: controller.signal,
        mode: 'cors',
        cache: 'no-store',
        credentials: 'omit'
      });
    } catch (err) {
      if (timer) global.clearTimeout(timer);
      return Promise.reject(new ApiError(
        'Không gửi được yêu cầu tới ' + url + ' (' + (err && err.message ? err.message : String(err)) + '). Kiểm tra lại địa chỉ API.',
        'network',
        { url: url, cause: err && err.message ? err.message : String(err) }
      ));
    }

    return fetchPromise.then(function (response) {
      var headerMs = now() - started;
      return response.text().then(function (text) {
        var parsed = null;
        if (text) {
          try { parsed = JSON.parse(text); } catch (err) { parsed = null; }
        }
        if (!response.ok) {
          throw new ApiError(extractDetail(parsed, response.status), 'http', {
            status: response.status,
            statusText: response.statusText,
            url: url,
            payload: parsed
          });
        }
        if (parsed === null && text) {
          throw new ApiError('Máy chủ trả về nội dung không phải JSON.', 'parse', { url: url });
        }
        return {
          status: response.status,
          data: parsed,
          rawText: text,
          url: url,
          timing: { sentMs: 0, receiveMs: headerMs, totalMs: now() - started }
        };
      });
    }).catch(function (err) {
      if (err instanceof ApiError) throw err;
      if (aborted) {
        throw new ApiError('Yêu cầu đã bị hủy.', 'aborted', { url: url, name: 'AbortError' });
      }
      if (timedOut || (err && err.name === 'AbortError')) {
        throw new ApiError('Yêu cầu vượt quá ' + limit + ' ms nên đã hủy.', 'timeout', { url: url, timeoutMs: limit });
      }
      throw new ApiError('Không kết nối được máy chủ API. Kiểm tra máy chủ đang chạy, địa chỉ API và cấu hình CORS.', 'network', {
        url: url,
        cause: err && err.message ? err.message : String(err)
      });
    }).finally(function () {
      if (timer) global.clearTimeout(timer);
      if (external && external.removeEventListener) external.removeEventListener('abort', onExternalAbort);
    });
  };

  ApiClient.prototype.health = function (signal) {
    var url = global.LAB3Config.healthUrl(this.config);
    return this._request(url, { method: 'GET', signal: signal }).then(function (res) {
      var data = res.data && typeof res.data === 'object' ? res.data : {};
      var text = '';
      if (typeof data.status === 'string') text = data.status;
      else if (typeof data.message === 'string') text = data.message;
      else if (typeof data.detail === 'string') text = data.detail;
      else if (data.ok === true) text = 'ok';
      else text = (res.rawText || '').replace(/\s+/g, ' ').slice(0, 60);
      if (data.ready === false) {
        return {
          ok: false, url: url,
          error: new ApiError('API đang chạy nhưng model chưa sẵn sàng. Đang chờ máy chủ nạp model.', 'server'),
          detail: data, timing: res.timing
        };
      }
      return {
        ok: true,
        status: res.status,
        url: url,
        detail: data,
        text: text,
        timing: res.timing
      };
    }).catch(function (err) {
      return { ok: false, url: url, error: err };
    });
  };

  ApiClient.prototype.predict = function (options) {
    var opts = options || {};
    var blob = opts.blob;
    if (!(blob instanceof global.Blob)) {
      return Promise.reject(new ApiError('Ảnh gửi đi không hợp lệ (thiếu Blob).', 'client'));
    }
    var faces = Array.isArray(opts.faces) ? opts.faces : [];
    if (faces.length === 0) {
      return Promise.reject(new ApiError('Chưa có khuôn mặt nào để gửi.', 'client'));
    }

    var width = toNumber(opts.width, 0) || 0;
    var height = toNumber(opts.height, 0) || 0;

    var form = new global.FormData();
    var filename = (blob.type || '').indexOf('png') >= 0 ? 'frame.png' : 'frame.jpg';
    form.append('image', blob, filename);
    form.append('faces', JSON.stringify(faces.map(function (face) {
      return {
        id: String(face.id),
        x: Math.round(Number(face.x) || 0),
        y: Math.round(Number(face.y) || 0),
        width: Math.round(Number(face.width) || 0),
        height: Math.round(Number(face.height) || 0)
      };
    })));

    var url = global.LAB3Config.predictUrl(this.config);
    var started = now();

    return this._request(url, {
      method: 'POST',
      body: form,
      signal: opts.signal,
      timeoutMs: opts.timeoutMs || this.config.timeoutMs
    }).then(function (res) {
      var data = res.data && typeof res.data === 'object' ? res.data : {};
      var predictions = Array.isArray(data.predictions) ? data.predictions : [];
      var receivedAt = now();

      return {
        ok: true,
        requestId: data.request_id === undefined || data.request_id === null ? null : String(data.request_id),
        image: {
          width: toNumber(data.image && data.image.width, width) || width,
          height: toNumber(data.image && data.image.height, height) || height
        },
        predictions: predictions.map(normalizePrediction),
        totalInferenceMs: toNumber(data.total_inference_ms, 0) || 0,
        timings: {
          captureMs: opts.captureMs === undefined ? null : opts.captureMs,
          sendMs: res.timing.sentMs,
          receiveMs: res.timing.receiveMs,
          roundTripMs: receivedAt - started,
          serverMs: toNumber(data.total_inference_ms, 0) || 0
        },
        raw: data
      };
    }).catch(function (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(String(err && err.message ? err.message : err), 'unknown');
    });
  };

  global.LAB3Api = {
    ApiClient: ApiClient,
    ApiError: ApiError,
    LABELS: LABELS,
    LABEL_TEXT: LABEL_TEXT,
    toNumber: toNumber
  };
})(window);
