/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package router

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// These tests pin the behaviour of extractFeatures BEFORE the PERF-01
// optimization (lazy body parse) so the refactor is provably
// behaviour-preserving for every well-formed body and for the malformed
// inputs that share semantics across implementations. extractFeaturesNaive
// below is a verbatim copy of the original full-json.Unmarshal
// implementation, kept as the benchmark baseline.

const testClassHeader = "x-llmkube-classification"

// --- behaviour: body field extraction (model + stream) ---

func TestExtractFeaturesBodyParsing(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantModel  string
		wantStream bool
	}{
		{"model and stream", `{"model":"gpt-4","stream":true}`, "gpt-4", true},
		{"stream before model (order independent)", `{"stream":true,"model":"gpt-4"}`, "gpt-4", true},
		{"stream false explicit", `{"model":"m","stream":false}`, "m", false},
		{"model only", `{"model":"only"}`, "only", false},
		{"stream only", `{"stream":true}`, "", true},
		// The model key inside a nested messages object must NOT be picked
		// up; only the top-level model counts.
		{"nested model ignored, top-level wins", `{"messages":[{"role":"user","model":"nested"}],"model":"real","stream":true}`, "real", true},
		{"model after large-ish prefix", `{"temperature":0.7,"max_tokens":256,"model":"after","stream":true}`, "after", true},
		// json.Unmarshal matches struct field names case-insensitively;
		// preserve that.
		{"case-insensitive model key", `{"Model":"CapKey"}`, "CapKey", false},
		{"upper-case model key", `{"MODEL":"shout","STREAM":true}`, "shout", true},
		// Duplicate top-level key: encoding/json takes the last; with no
		// stream present the lazy parser reads through to the last too.
		{"duplicate model last wins", `{"model":"a","model":"b"}`, "b", false},
		{"leading/trailing whitespace", "   {\"model\":\"sp\"}  ", "sp", false},
		// Type mismatches and null leave the zero value (no error surfaced).
		{"model wrong type", `{"model":123}`, "", false},
		{"stream wrong type string", `{"stream":"true"}`, "", false},
		{"stream wrong type number", `{"stream":1}`, "", false},
		{"null values", `{"model":null,"stream":null}`, "", false},
		// Non-object / non-JSON bodies → empty defaults (best-effort).
		{"empty object", `{}`, "", false},
		{"not json", `not json`, "", false},
		{"top-level array", `[1,2,3]`, "", false},
		{"top-level string", `"juststring"`, "", false},
		{"empty body", ``, "", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(tc.body))
			f, isStream := extractFeatures([]byte(tc.body), r, testClassHeader)
			if f.Model != tc.wantModel {
				t.Errorf("Model = %q, want %q", f.Model, tc.wantModel)
			}
			if isStream != tc.wantStream {
				t.Errorf("stream = %v, want %v", isStream, tc.wantStream)
			}
		})
	}
}

// TestExtractFeaturesMatchesNaiveOnWellFormed asserts the optimized parser
// is byte-for-byte equivalent to the original full-Unmarshal implementation
// for every well-formed body — the contract that actually matters.
func TestExtractFeaturesMatchesNaiveOnWellFormed(t *testing.T) {
	bodies := []string{
		`{"model":"gpt-4","stream":true}`,
		`{"stream":false,"model":"x"}`,
		`{"model":"only"}`,
		`{"stream":true}`,
		`{}`,
		`{"messages":[{"role":"user","content":"hi","model":"nested"}],"model":"real","stream":true}`,
		`{"temperature":0.2,"model":"after-fields","top_p":0.9,"stream":true}`,
		`{"Model":"CapKey","Stream":true}`,
		`{"model":123,"stream":"nope"}`,
		string(buildChatBody(20, false)),
		string(buildChatBody(20, true)),
		string(buildChatBody(100, true)),
	}
	r := httptest.NewRequest(http.MethodPost, "/", nil)
	for i, b := range bodies {
		gotM, gotS := extractFeatures([]byte(b), r, testClassHeader)
		wantM, wantS := extractFeaturesNaive([]byte(b), r, testClassHeader)
		if gotM.Model != wantM.Model || gotS != wantS {
			t.Errorf("case %d: lazy=(%q,%v) naive=(%q,%v) diverged\nbody=%.120s",
				i, gotM.Model, gotS, wantM.Model, wantS, b)
		}
	}
}

// --- behaviour: header extraction (unchanged by PERF-01) ---

func TestExtractFeaturesHeaders(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"model":"m"}`))
	r.Header.Set("X-LLMKube-Classification", "PII")
	r.Header.Set("X-LLMKube-Task-Complexity", "HARD")
	r.Header.Set("Content-Type", "application/json")

	f, _ := extractFeatures([]byte(`{"model":"m"}`), r, testClassHeader)

	if f.Classification != "pii" {
		t.Errorf("Classification = %q, want %q (lower-cased value)", f.Classification, "pii")
	}
	if f.TaskComplexity != "hard" {
		t.Errorf("TaskComplexity = %q, want %q", f.TaskComplexity, "hard")
	}
	// Header map keys are canonicalised to lower-case.
	if got := f.Headers["content-type"]; got != "application/json" {
		t.Errorf("Headers[content-type] = %q, want application/json", got)
	}
	if _, ok := f.Headers["Content-Type"]; ok {
		t.Errorf("Headers should be keyed by lower-case, found mixed-case key")
	}
}

func TestExtractFeaturesCustomClassHeader(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{}`))
	r.Header.Set("X-Tenant-Class", "Confidential")
	f, _ := extractFeatures([]byte(`{}`), r, "x-tenant-class")
	if f.Classification != "confidential" {
		t.Errorf("Classification = %q, want confidential", f.Classification)
	}
}

// TestExtractFeaturesTruncatedBodyIsLenient documents the ONE intentional
// behavioural relaxation from PERF-01: the original full-Unmarshal discarded
// every field when the JSON object was truncated, whereas the early-exit
// lazy parser surfaces top-level fields it has already read cleanly. Both
// are valid under the "best-effort, never panic" contract; a truncated body
// is not a supported input. We assert only that it does not panic and never
// invents a value the bytes do not contain.
func TestExtractFeaturesTruncatedBodyIsLenient(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/", nil)
	for _, body := range []string{`{"model":"x"`, `{"model":"x",`, `{"stream":true`} {
		f, isStream := extractFeatures([]byte(body), r, testClassHeader)
		// Never fabricate a model the bytes don't contain.
		if f.Model != "" && f.Model != "x" {
			t.Errorf("body %q: unexpected Model %q", body, f.Model)
		}
		_ = isStream // value is best-effort for truncated input
	}
}

// ---------------------------------------------------------------------------
// Benchmark baseline: verbatim copy of the ORIGINAL extractFeatures (full
// json.Unmarshal of the whole body). Used by BenchmarkExtractFeaturesNaiveJSON
// for an apples-to-apples side-by-side against the optimized production path.
// ---------------------------------------------------------------------------

func extractFeaturesNaive(body []byte, r *http.Request, classHeader string) (RequestFeatures, bool) {
	headers := make(map[string]string, len(r.Header))
	for k, vals := range r.Header {
		if len(vals) > 0 {
			headers[strings.ToLower(k)] = vals[0]
		}
	}

	var partial struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	_ = json.Unmarshal(body, &partial)

	return RequestFeatures{
		Model:          partial.Model,
		Classification: strings.ToLower(headers[strings.ToLower(classHeader)]),
		TaskComplexity: strings.ToLower(headers["x-llmkube-task-complexity"]),
		Headers:        headers,
	}, partial.Stream
}

// buildChatBody produces a realistic OpenAI chat-completion body with the
// given number of messages. When streamLast is true the "stream" field is
// placed AFTER the (large) messages array — the worst case for an early-exit
// parser, which must still skip the array to reach it. Otherwise model+stream
// lead the object (best case: the messages array is never scanned).
func buildChatBody(numMessages int, streamLast bool) []byte {
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	msgs := make([]msg, numMessages)
	for i := range msgs {
		msgs[i] = msg{
			Role: "user",
			Content: fmt.Sprintf("message %d: %s", i,
				strings.Repeat("the quick brown fox jumps over the lazy dog. ", 18)),
		}
	}
	msgsJSON, _ := json.Marshal(msgs)

	var b strings.Builder
	b.WriteString(`{"model":"meta-llama/Llama-3.1-8B-Instruct",`)
	if !streamLast {
		b.WriteString(`"stream":true,`)
	}
	b.WriteString(`"temperature":0.7,"max_tokens":512,"messages":`)
	b.Write(msgsJSON)
	if streamLast {
		b.WriteString(`,"stream":true`)
	}
	b.WriteString(`}`)
	return []byte(b.String())
}

func benchRequest() *http.Request {
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	r.Header.Set("X-LLMKube-Classification", "public")
	r.Header.Set("User-Agent", "bench/1.0")
	return r
}

// benchmark payloads: name -> body. Sizes span a tiny request, a typical
// multi-turn chat, and a large context window, in both stream-early and
// stream-last orderings.
func benchPayloads() []struct {
	name string
	body []byte
} {
	return []struct {
		name string
		body []byte
	}{
		{"tiny_no_messages", []byte(`{"model":"meta-llama/Llama-3.1-8B-Instruct","stream":true}`)},
		{"chat_20msgs_stream_early", buildChatBody(20, false)},
		{"chat_100msgs_stream_early", buildChatBody(100, false)},
		{"chat_100msgs_stream_last", buildChatBody(100, true)},
	}
}

// sink prevents the compiler from optimizing the benchmarked calls away.
var (
	sinkFeatures RequestFeatures
	sinkStream   bool
)

// BenchmarkExtractFeatures measures the PRODUCTION extractFeatures. Run it on
// the pre-change commit for the "before" baseline and on the post-change
// commit for "after"; compare with benchstat.
func BenchmarkExtractFeatures(b *testing.B) {
	r := benchRequest()
	for _, p := range benchPayloads() {
		b.Run(p.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(p.body)))
			for i := 0; i < b.N; i++ {
				sinkFeatures, sinkStream = extractFeatures(p.body, r, testClassHeader)
			}
		})
	}
}

// BenchmarkExtractFeaturesNaiveJSON measures the original full-Unmarshal
// implementation on the identical payloads, so a single `go test -bench`
// run shows the optimized-vs-original delta directly (durable, commit-agnostic).
func BenchmarkExtractFeaturesNaiveJSON(b *testing.B) {
	r := benchRequest()
	for _, p := range benchPayloads() {
		b.Run(p.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(p.body)))
			for i := 0; i < b.N; i++ {
				sinkFeatures, sinkStream = extractFeaturesNaive(p.body, r, testClassHeader)
			}
		})
	}
}
