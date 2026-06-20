/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// Proxy is the router-proxy HTTP application. Construct via NewProxy and
// register handlers with Mount; the proxy is safe for concurrent use.
type Proxy struct {
	cfg     *Config
	matcher *Matcher
	disp    *Dispatcher
	logger  *slog.Logger
}

// ProxyOption customizes a Proxy at construction time. The proxy
// owns the Dispatcher, so options that affect dispatching (eg
// quarantine duration) are forwarded through.
type ProxyOption func(*Proxy)

// WithDispatcherOptions threads DispatcherOption values down to the
// proxy's owned Dispatcher. Used to set --quarantine-duration from
// the CLI without leaking Dispatcher construction up to callers.
func WithDispatcherOptions(opts ...DispatcherOption) ProxyOption {
	return func(p *Proxy) {
		// Rebuild the dispatcher with the requested options. NewProxy
		// runs WithDispatcherOptions *after* NewDispatcher already
		// constructed a default-options dispatcher; rebuilding here
		// keeps the option API uniform without making callers care
		// about construction order.
		p.disp = NewDispatcher(p.cfg, opts...)
	}
}

// NewProxy constructs a Proxy from a loaded Config.
func NewProxy(cfg *Config, logger *slog.Logger, opts ...ProxyOption) *Proxy {
	if logger == nil {
		logger = slog.Default()
	}
	p := &Proxy{
		cfg:     cfg,
		matcher: NewMatcher(cfg),
		disp:    NewDispatcher(cfg),
		logger:  logger,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// Mount wires up the OpenAI-compatible endpoints plus /health on the
// given mux. Callers attach the mux to an http.Server.
func (p *Proxy) Mount(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/chat/completions", p.handleChatCompletions)
	mux.HandleFunc("GET /v1/models", p.handleModels)
	mux.HandleFunc("GET /health", p.handleHealth)
	mux.HandleFunc("GET /healthz", p.handleHealth)
}

// handleHealth always returns 200. The Kubernetes liveness probe uses
// this; the proxy is "alive" as long as its goroutine runs. Readiness
// gating on backend health lands with #432 / #428.
func (p *Proxy) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

// handleModels returns an OpenAI-compatible /v1/models payload listing
// every backend in the config. Cloud backends report their upstream
// Model field; local backends report their backend name.
func (p *Proxy) handleModels(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	now := time.Now().Unix()
	models := make([]model, 0, len(p.cfg.Backends))
	for _, b := range p.cfg.Backends {
		id := b.Name
		if b.Model != "" {
			id = b.Model
		}
		owned := "llmkube"
		if b.Provider != "" {
			owned = b.Provider
		}
		models = append(models, model{ID: id, Object: "model", Created: now, OwnedBy: owned})
	}
	body, _ := json.Marshal(map[string]any{"object": "list", "data": models})
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(body)
}

// handleChatCompletions is the primary routing endpoint. It buffers the
// inbound request body (needs the "model" field for matching), evaluates
// the rule set, dispatches to the chosen backend, and streams the
// response back. SSE / chunked passthrough is automatic.
func (p *Proxy) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodyBytes))
	if err != nil {
		writeError(w, http.StatusBadRequest, "request body: "+err.Error())
		return
	}

	features, isStream := extractFeatures(body, r, p.cfg.ClassificationHeader())

	decision := p.matcher.Match(&features)
	if len(decision.Backends) == 0 {
		writeError(w, http.StatusServiceUnavailable, "no rule matched and no defaultRoute configured")
		p.audit(features, decision, nil, http.StatusServiceUnavailable, "no_route", 0)
		return
	}

	if err := p.enforceFailClosed(&features, &decision); err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		p.audit(features, decision, nil, http.StatusServiceUnavailable, "fail_closed", 0)
		return
	}

	start := time.Now()
	chosen, resp, err := p.dispatchWithFallback(r.Context(), &decision, r.Header, body, "/v1/chat/completions")
	elapsed := time.Since(start)
	if err != nil {
		// Runtime fail-closed: when every backend in a fail-closed
		// rule's pool is unreachable, return 503 with a clear reason
		// rather than 502. This is the runtime counterpart to the
		// controller's static fail-closed validation: we refuse the
		// request instead of letting it spill onto an unmatched
		// backend (which dispatchWithFallback never does anyway, but
		// the status code communicates intent — sensitive data did
		// not egress because policy said so, not because of a generic
		// upstream outage).
		if decision.FailClosed {
			writeError(w, http.StatusServiceUnavailable,
				"fail-closed: all rule backends unhealthy: "+err.Error())
			p.audit(features, decision, nil, http.StatusServiceUnavailable,
				"fail_closed_runtime", elapsed)
			return
		}
		writeError(w, http.StatusBadGateway, "all backends failed: "+err.Error())
		p.audit(features, decision, nil, http.StatusBadGateway, "all_backends_failed", elapsed)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	streamed := streamResponse(w, resp, isStream)
	p.audit(features, decision, chosen, resp.StatusCode, streamedReason(streamed), elapsed)
}

const maxRequestBodyBytes = 32 << 20 // 32 MiB, generous for long prompts

// extractFeatures pulls the model name, stream flag, classification, and
// task complexity out of the inbound request. The model name comes from
// the JSON body; the rest come from headers per the MVP header-only
// classification mode.
func extractFeatures(body []byte, r *http.Request, classHeader string) (RequestFeatures, bool) {
	headers := make(map[string]string, len(r.Header))
	for k, vals := range r.Header {
		if len(vals) > 0 {
			headers[strings.ToLower(k)] = vals[0]
		}
	}

	// Body may not be valid JSON yet at this point (the upstream may be
	// more permissive). We extract what we can and proceed.
	model, isStream := parseModelStream(body)

	return RequestFeatures{
		Model:          model,
		Classification: strings.ToLower(headers[strings.ToLower(classHeader)]),
		TaskComplexity: strings.ToLower(headers["x-llmkube-task-complexity"]),
		Headers:        headers,
	}, isStream
}

// parseModelStream extracts the top-level "model" (string) and "stream"
// (bool) fields from an OpenAI-style chat-completion body WITHOUT parsing the
// (often multi-hundred-KB) messages/tools payload. A chat body's expensive
// content is the messages array; model and stream are small top-level fields,
// so a full json.Unmarshal spends ~all its time scanning bytes it discards.
//
// This walks the top-level object structurally and early-exits the moment both
// fields are seen — so when model/stream precede the large arrays, those arrays
// are never scanned at all. Values that are not model/stream are skipped with a
// non-allocating boundary scan (no copy, unlike a json.RawMessage decode), so
// the worst case (both fields after the array) is still a single O(n) pass with
// no per-element allocation, never worse than the previous full Unmarshal.
//
// Value DECODING is delegated to encoding/json over the located byte span, so
// string-escape and type semantics match json.Unmarshal exactly (including
// case-insensitive key matching and type-mismatch → zero value). Only the
// structural boundary scan is hand-rolled.
//
// Behaviour is identical to the previous json.Unmarshal(body, &{model,stream})
// for every well-formed JSON object. A non-object / non-JSON body yields the
// zero values. A truncated/malformed object is best-effort: it may surface
// fields read cleanly before the corruption (the old code discarded them) — a
// malformed body is not a supported input.
func parseModelStream(body []byte) (model string, stream bool) {
	i := skipWS(body, 0)
	if i >= len(body) || body[i] != '{' {
		return "", false // not a JSON object
	}
	i++

	haveModel, haveStream := false, false
	for {
		i = skipWS(body, i)
		if i >= len(body) {
			return model, stream
		}
		switch body[i] {
		case '}':
			return model, stream
		case ',':
			i++
			continue
		case '"':
			// fall through to key handling below
		default:
			return model, stream // malformed; best-effort with what we have
		}

		keyStart := i
		keyEnd, ok := scanStringEnd(body, i)
		if !ok {
			return model, stream
		}
		i = skipWS(body, keyEnd)
		if i >= len(body) || body[i] != ':' {
			return model, stream
		}
		i = skipWS(body, i+1)
		valStart := i
		valEnd, ok := skipValue(body, i)
		if !ok {
			return model, stream
		}
		i = valEnd

		// Key bytes include the surrounding quotes. json.Unmarshal matches
		// struct field names case-insensitively, so we do too. Last write
		// wins (no early-exit while only one field has been seen), matching
		// encoding/json for duplicate keys.
		key := body[keyStart:keyEnd]
		switch {
		case equalFoldKey(key, "model"):
			var s string
			if json.Unmarshal(body[valStart:valEnd], &s) == nil {
				model = s
			}
			haveModel = true
		case equalFoldKey(key, "stream"):
			var b bool
			if json.Unmarshal(body[valStart:valEnd], &b) == nil {
				stream = b
			}
			haveStream = true
		}
		if haveModel && haveStream {
			return model, stream // both captured — skip the rest of the body
		}
	}
}

func skipWS(b []byte, i int) int {
	for i < len(b) {
		switch b[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// scanStringEnd returns the index just past the closing quote of the JSON
// string beginning at b[i] (which must be '"'). ok is false if unterminated.
func scanStringEnd(b []byte, i int) (int, bool) {
	i++ // past opening quote
	for i < len(b) {
		switch b[i] {
		case '\\':
			i += 2 // skip the escape and its escaped char (\" \\ \uXXXX all safe)
		case '"':
			return i + 1, true
		default:
			i++
		}
	}
	return i, false
}

// skipValue returns the index just past the complete JSON value starting at
// b[i] (leading whitespace already consumed). ok is false on a malformed value.
func skipValue(b []byte, i int) (int, bool) {
	if i >= len(b) {
		return i, false
	}
	switch b[i] {
	case '"':
		return scanStringEnd(b, i)
	case '{', '[':
		return skipContainer(b, i)
	default:
		// number / true / false / null: run to the next structural delimiter.
		for i < len(b) {
			switch b[i] {
			case ',', '}', ']', ' ', '\t', '\n', '\r':
				return i, true
			default:
				i++
			}
		}
		return i, true // value runs to EOF (e.g. a bare top-level scalar)
	}
}

// skipContainer skips a balanced object or array starting at b[i] ('{' or '['),
// honoring strings so brackets inside string literals are not counted.
func skipContainer(b []byte, i int) (int, bool) {
	depth := 0
	for i < len(b) {
		switch b[i] {
		case '"':
			ni, ok := scanStringEnd(b, i)
			if !ok {
				return ni, false
			}
			i = ni
		case '{', '[':
			depth++
			i++
		case '}', ']':
			depth--
			i++
			if depth == 0 {
				return i, true
			}
		default:
			i++
		}
	}
	return i, false // unbalanced
}

// equalFoldKey reports whether the quoted JSON key (bytes including the
// surrounding double quotes) equals target case-insensitively. target must be
// lowercase ASCII; this avoids allocating a string per key. Keys containing
// JSON escapes (e.g. "model") are not folded — they never occur for the
// fixed field names we look for.
func equalFoldKey(quoted []byte, target string) bool {
	if len(quoted) != len(target)+2 || quoted[0] != '"' || quoted[len(quoted)-1] != '"' {
		return false
	}
	inner := quoted[1 : len(quoted)-1]
	for j := 0; j < len(target); j++ {
		c := inner[j]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != target[j] {
			return false
		}
	}
	return true
}

// enforceFailClosed implements the runtime half of the fail-closed gate.
// For sensitive-data requests routing to a fail-closed rule, every
// backend in the route must be local-tier; otherwise we refuse. The
// static half of this check runs in the controller at apply time, but
// repeating it here defends against drift between controller and proxy
// config.
func (p *Proxy) enforceFailClosed(f *RequestFeatures, dec *MatchResult) error {
	if !dec.FailClosed {
		return nil
	}
	sensitive := p.cfg.SensitiveSet()
	if !sensitive[f.Classification] {
		return nil
	}
	for _, name := range dec.Backends {
		b := p.matcher.BackendByName(name)
		if b == nil {
			return fmt.Errorf("fail-closed: backend %q not configured", name)
		}
		if b.Tier != "local" {
			return fmt.Errorf("fail-closed: sensitive classification %q cannot route to %s-tier backend %q",
				f.Classification, b.Tier, name)
		}
	}
	return nil
}

// dispatchWithFallback walks the backend list in declared order and
// returns the first successful (non-5xx, non-error) response. On
// fail-closed routes with all backends unhealthy, the last error
// propagates back to the handler as HTTP 502 or 503 (the caller in
// handleChatCompletions decides the surface based on dec.FailClosed).
//
// Per-attempt deadline: resolveDispatchTimeout produces the cap for
// each backend attempt, with resolution order rule -> backend ->
// proxy default. The deadline is applied per attempt, not once for
// the whole loop, so a slow primary that timed out does NOT eat the
// fallback's budget.
func (p *Proxy) dispatchWithFallback(
	ctx context.Context,
	dec *MatchResult,
	headers http.Header,
	body []byte,
	path string,
) (*Backend, *http.Response, error) {
	var lastErr error
	for _, name := range dec.Backends {
		b := p.matcher.BackendByName(name)
		if b == nil {
			lastErr = fmt.Errorf("backend %q not configured", name)
			continue
		}
		if !p.disp.IsHealthy(name) {
			lastErr = fmt.Errorf("backend %q marked unhealthy", name)
			continue
		}
		attemptCtx, cancel := context.WithTimeout(ctx,
			resolveDispatchTimeout(dec, b, p.disp.ResponseHeaderTimeout()))
		resp, err := p.disp.Dispatch(attemptCtx, b, http.MethodPost, path, headers, body)
		if err != nil {
			cancel()
			lastErr = err
			continue
		}
		if resp.StatusCode >= 500 {
			// Drain and close so the connection is reusable.
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			cancel()
			lastErr = fmt.Errorf("%s returned %d", name, resp.StatusCode)
			continue
		}
		// Successful response: wrap the body so its Close also
		// cancels the per-attempt context. The caller's existing
		// `defer resp.Body.Close()` is enough — no separate cancel
		// plumbing needed, and streaming dispatches keep the
		// deadline alive until the client finishes reading.
		resp.Body = &cancelOnClose{ReadCloser: resp.Body, cancel: cancel}
		return b, resp, nil
	}
	if lastErr == nil {
		lastErr = errors.New("no backends attempted")
	}
	return nil, nil, lastErr
}

// cancelOnClose pairs a response body with the cancel func of the
// per-attempt context. The proxy's caller already does
// `defer resp.Body.Close()`, so wrapping the body ensures the
// per-attempt context cancel fires no later than the response is
// fully consumed (and no sooner — streaming responses need the
// deadline to outlive the chat completion).
type cancelOnClose struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (c *cancelOnClose) Close() error {
	err := c.ReadCloser.Close()
	c.cancel()
	return err
}

// resolveDispatchTimeout computes the per-attempt context deadline
// using the resolution order: rule.Timeout || backend.Timeout ||
// proxy default. Zero values fall through to the next level. This
// is the runtime half of #458; the controller already validated
// bounds at apply time so any non-zero value reaching here is sane.
func resolveDispatchTimeout(dec *MatchResult, backend *Backend, proxyDefault time.Duration) time.Duration {
	if dec != nil && dec.Rule != nil && dec.Rule.Timeout > 0 {
		return dec.Rule.Timeout
	}
	if backend != nil && backend.Timeout > 0 {
		return backend.Timeout
	}
	return proxyDefault
}

// streamResponse copies the upstream response to the client. Returns
// true if we treated the response as a stream (set SSE-friendly headers
// and flushed after every chunk). The decision is driven by the
// request's "stream": true flag and the upstream Content-Type.
func streamResponse(w http.ResponseWriter, resp *http.Response, requestedStream bool) bool {
	upstreamCT := resp.Header.Get("Content-Type")
	isSSE := strings.HasPrefix(upstreamCT, "text/event-stream")
	isStream := requestedStream || isSSE

	// Forward upstream headers (except hop-by-hop).
	for k, vals := range resp.Header {
		if hopByHop[strings.ToLower(k)] {
			continue
		}
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	if isStream && w.Header().Get("Cache-Control") == "" {
		w.Header().Set("Cache-Control", "no-cache")
	}
	w.WriteHeader(resp.StatusCode)

	if !isStream {
		_, _ = io.Copy(w, resp.Body)
		return false
	}

	var flush func()
	if f, ok := w.(http.Flusher); ok {
		flush = f.Flush
	}
	_, _ = PipeBody(w, resp.Body, flush)
	return true
}

// audit emits one structured log line per request. Sink configuration
// (file, OTLP) lands with #434; for the MVP we always log to the proxy
// stdout via slog.
func (p *Proxy) audit(
	f RequestFeatures,
	dec MatchResult,
	chosen *Backend,
	statusCode int,
	outcome string,
	elapsed time.Duration,
) {
	attrs := []any{
		"model", f.Model,
		"classification", f.Classification,
		"taskComplexity", f.TaskComplexity,
		"status", statusCode,
		"outcome", outcome,
		"latencyMs", elapsed.Milliseconds(),
	}
	if dec.Rule != nil {
		attrs = append(attrs, "rule", dec.Rule.Name)
	}
	if chosen != nil {
		attrs = append(attrs, "backend", chosen.Name, "backendTier", chosen.Tier)
	}
	// The resolved per-request deadline is informative regardless of
	// outcome: on success it shows the budget that DID land, on
	// timeout it shows the budget that DIDN'T suffice. Operators
	// debugging "why did this rule 504?" can grep audit logs for
	// `timeoutMs=<expected>` and reconcile vs the CRD spec.
	attrs = append(attrs, "timeoutMs",
		resolveDispatchTimeout(&dec, chosen, p.disp.ResponseHeaderTimeout()).Milliseconds())
	p.logger.Info("router.dispatch", attrs...)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{"code": code, "message": msg},
	})
	_, _ = w.Write(body)
}

func streamedReason(streamed bool) string {
	if streamed {
		return "ok_stream"
	}
	return "ok"
}
