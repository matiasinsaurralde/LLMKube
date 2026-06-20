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
	"strings"
	"testing"
)

// FuzzParseModelStream asserts the optimized scanner is byte-for-byte
// equivalent to the original full-Unmarshal extraction for every well-formed
// JSON object (the contract that matters), and never panics on arbitrary
// input. Duplicate top-level model/stream keys are excluded from the
// equivalence assertion: encoding/json takes the last, whereas the early-exit
// scanner stops at the first complete pair — a documented, pathological-only
// difference.
func FuzzParseModelStream(f *testing.F) {
	seeds := []string{
		`{"model":"gpt-4","stream":true}`,
		`{"stream":true,"model":"x","messages":[{"role":"user","content":"hi"}]}`,
		`{"messages":[{"model":"nested"}],"model":"real"}`,
		`{"model":123,"stream":"no"}`,
		`{"Model":"caps","STREAM":true}`,
		`{"temperature":0.7,"model":"mé","stream":false}`,
		`{}`, `not json`, `[1,2,3]`, `"s"`, ``, `{"model":"x"`,
		`{"a":{"b":[1,"}{",2]},"model":"deep","stream":true}`,
	}
	for _, s := range seeds {
		f.Add([]byte(s))
	}

	f.Fuzz(func(t *testing.T, body []byte) {
		// Must never panic on any input.
		gotModel, gotStream := parseModelStream(body)

		if !json.Valid(body) {
			return
		}
		var probe any
		if err := json.Unmarshal(body, &probe); err != nil {
			return
		}
		if _, isObject := probe.(map[string]any); !isObject {
			// Valid non-object JSON must yield zero values.
			if gotModel != "" || gotStream {
				t.Fatalf("non-object %q: got (%q,%v), want empty", body, gotModel, gotStream)
			}
			return
		}
		if topLevelKeyCount(t, body, "model") > 1 || topLevelKeyCount(t, body, "stream") > 1 {
			return // duplicate keys: see doc comment
		}

		var p struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		_ = json.Unmarshal(body, &p)
		if gotModel != p.Model || gotStream != p.Stream {
			t.Fatalf("diverged for %q: scanner=(%q,%v) unmarshal=(%q,%v)",
				body, gotModel, gotStream, p.Model, p.Stream)
		}
	})
}

// topLevelKeyCount counts case-insensitive occurrences of a top-level object
// key (test helper; correctness over speed).
func topLevelKeyCount(t *testing.T, body []byte, key string) int {
	t.Helper()
	dec := json.NewDecoder(strings.NewReader(string(body)))
	tok, err := dec.Token()
	if err != nil {
		return 0
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return 0
	}
	count := 0
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return count
		}
		if k, ok := kt.(string); ok && strings.EqualFold(k, key) {
			count++
		}
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return count
		}
	}
	return count
}
