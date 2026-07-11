# LLMKube Security Audit Findings

**Date:** 2026-07-11  
**Scope:** Static code review of the LLMKube repository (inference operator, ModelRouter, metal-agent, Foreman agentic stack, Helm charts, CLI)  
**Method:** Source review, pattern search, architectural analysis. No live cluster penetration testing.

## Executive Summary

LLMKube spans three trust boundaries:

1. **Inference operator** (`inference.llmkube.dev`) — Models, InferenceServices, ModelRouters
2. **Foreman** (`foreman.llmkube.dev`) — agentic workloads, validating webhooks, bash/git/job tools
3. **Metal agent** — host-native inference with Kubernetes EndpointSlice registration

The codebase has solid defaults in several areas (restricted pod security contexts for controller/router-proxy, Foreman workspace path containment, metrics RBAC auth, weekly `govulncheck`). The most serious gaps are **unauthenticated inference endpoints**, **client-spoofable ModelRouter classification** (undermining fail-closed data-sovereignty claims), **SSRF via model download URLs**, and **CRD fields that allow privileged pod escape hatches**.

**Assumed threat model for severity ratings:** a multi-tenant Kubernetes cluster where namespace-scoped users can create `Model` and `InferenceService` CRs (e.g. via `inferenceservice-editor-role`). Single-admin clusters with strict Pod Security Admission reduce several High findings to configuration risk.

---

## Findings (Most Severe First)

### CRITICAL

#### F-001: ModelRouter fail-closed policy bypass via client-controlled classification header

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Exploitability** | High — any client with network access to the router Service |
| **Locations** | `internal/router/proxy.go:177-225`, `internal/router/config.go:157-164` |

**Description:** Fail-closed routing for PII/PHI depends on the `x-llmkube-classification` header (or a configured equivalent). The proxy reads this header directly from the inbound HTTP request and trusts it. Server-side classification (`detector` / `hybrid` modes) is documented as future work; only `header-only` mode is implemented.

**Exploit scenario:** An attacker (or careless client) sends sensitive prompts to `/v1/chat/completions` with `x-llmkube-classification: public` (or omits the header entirely if it is not in the sensitive set). Requests route to cloud-tier backends despite CRD rules configured with `failClosed: true` and local-only backends for sensitive data.

**Impact:** Data-sovereignty and compliance controls are ineffective. Sensitive prompts, PII, and PHI can egress to third-party LLM providers.

**Remediation:**
- Treat header-only classification as advisory until server-side detection ships; update documentation accordingly.
- Default-deny cloud backends unless authenticated service identity is present.
- Implement `detector`/`hybrid` classification before marketing fail-closed as a compliance control.
- Optionally reject requests missing required classification headers on fail-closed routes.

---

#### F-002: Inference and router-proxy APIs expose unauthenticated HTTP by default

| Field | Value |
|-------|-------|
| **Severity** | Critical (in multi-tenant or internet-exposed deployments) |
| **Exploitability** | High — requires only network reachability |
| **Locations** | `internal/controller/service_builder.go:33-74`, `internal/controller/router_service_builder.go:63-76`, `cmd/router-proxy/main.go:39-40,82-88` |

**Description:** OpenAI-compatible `/v1/chat/completions` is served over plain HTTP with no API key, mTLS, or OAuth middleware. `spec.endpoint.type: LoadBalancer` or `NodePort` can expose inference directly to external networks.

**Exploit scenario:** Any principal with pod-to-Service network access (or internet access via LoadBalancer) invokes models freely: GPU exhaustion, prompt injection against internal systems, data exfiltration via model responses, and unbounded token spend on cloud backends.

**Impact:** Complete loss of confidentiality, availability, and cost control on the inference data plane.

**Remediation:**
- Document mandatory NetworkPolicy + ingress authentication (OAuth2 proxy, Envoy ext_authz, service mesh).
- Add optional API-key or mTLS support to router-proxy.
- Emit warnings or require explicit opt-in for `LoadBalancer`/`NodePort` endpoint types.
- Ship a `values-secure.yaml` profile with NetworkPolicies enabled and auth guidance.

---

### HIGH

#### F-003: SSRF via user-controlled model download URLs

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Exploitability** | Medium–High — requires `Model` create RBAC |
| **Locations** | `internal/controller/model_controller.go:636-680`, `internal/controller/model_revalidate.go:158-172`, `internal/controller/model_storage.go:55,85`, `pkg/agent/executor.go:270-299`, `pkg/agent/memory.go` |

**Description:** Model `spec.source` HTTP/HTTPS URLs are fetched via `http.Get` / `http.DefaultClient` with no timeout, redirect limit, response size cap, or blocklist for private/link-local/metadata addresses (`169.254.169.254`, RFC1918, loopback). Init containers run `curl -f -L` against the same URLs.

**Exploit scenario:**
- **Controller pod:** Attacker creates `Model` with `source: http://169.254.169.254/latest/meta-data/iam/security-credentials/`. Hourly HEAD revalidation issues requests from the operator's network position and service account.
- **Tenant pod:** Init container curls internal ClusterIP services or cloud metadata endpoints reachable from the pod network.
- **Metal-agent (Mac):** Downloads arbitrary URLs to `--model-store`, reaching localhost services on the host.

**Impact:** Cloud credential theft, internal service enumeration, lateral movement, denial of service via large responses.

**Remediation:**
- Centralize URL validation: HTTPS-only (configurable), block private/link-local/metadata IPs after DNS resolution, cap redirects, set timeouts and max download size.
- Use `http.Client` with `context` deadlines everywhere (replace bare `http.Get`).
- Apply egress NetworkPolicies denying metadata and internal CIDRs from init containers.
- Consider disabling controller-side HTTP fetch entirely; rely on init-container downloads only.

---

#### F-004: HostPath mount from user-controlled local model paths

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Exploitability** | Medium–High — requires Model/InferenceService create + node scheduling |
| **Locations** | `internal/controller/model_storage.go:174-189`, `internal/controller/source.go:92-103` |

**Description:** Local model sources (`file://` or absolute paths) cause a `HostPath` volume mount with `Type: File` at the user-supplied path. No allowlist or path normalization is applied.

**Exploit scenario:** Attacker creates `Model` with `source: /etc/shadow` (or any readable host file on a target node). The init container mounts and copies the file into the shared namespace cache PVC; inference workloads can then read it.

**Impact:** Host filesystem disclosure; potential credential theft depending on node contents.

**Remediation:**
- Disable `file://` and absolute-path sources by default; gate behind an operator flag.
- Require `pvc://` for pre-staged models in multi-tenant environments.
- If HostPath must remain, enforce an allowlist prefix (e.g. `/mnt/models/` only) via admission webhook.

---

#### F-005: InferenceService CRD allows full security-context override (including privileged)

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Exploitability** | High — if InferenceService create is delegated to untrusted users |
| **Locations** | `internal/controller/deployment_builder.go:70-96,186-195,216-218`, `api/v1alpha1/inferenceservice_types.go` |

**Description:** Defaults drop capabilities and set `allowPrivilegeEscalation: false`, but user-supplied `spec.securityContext` and `spec.podSecurityContext` replace defaults verbatim. The CRD exposes full Kubernetes `SecurityContext` fields including `privileged`, `hostNetwork`, `hostPID`, and custom capabilities. `spec.command` overrides the container entrypoint for all runtimes; `spec.probeOverrides` accepts full `exec` probes.

**Exploit scenario:**

```yaml
spec:
  image: attacker/evil:latest
  command: ["/bin/sh", "-c", "sleep infinity"]
  securityContext:
    privileged: true
  probeOverrides:
    liveness:
      exec:
        command: ["cat", "/var/run/secrets/kubernetes.io/serviceaccount/token"]
```

**Impact:** Container escape and node compromise when Pod Security Admission is weak or absent.

**Remediation:**
- Add validating admission webhooks denying `privileged`, `hostNetwork`, `hostPID`, `hostIPC`, and dangerous capabilities.
- Restrict `spec.command` to `runtime: generic` only.
- Document that InferenceService is an admin-only escape hatch, not a tenant-safe abstraction.

---

#### F-006: Foreman bash tool exposes GITHUB_TOKEN to LLM-driven shell

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Exploitability** | High — when bash tool is enabled and git auth is configured |
| **Locations** | `pkg/foreman/agent/tools/bash.go:227-240,251-277` |

**Description:** `GITHUB_TOKEN` is in `defaultBashEnvAllowlist` and passed into `sh -c` commands executed by the agent. Workspace and `cd` guards do not prevent outbound network exfiltration.

**Exploit scenario:** A compromised or jailbroken model runs `curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user` or exfiltrates the token to an external endpoint. Token may grant repo write, org read, or broader scopes.

**Impact:** GitHub account compromise, supply-chain attacks via malicious commits.

**Remediation:**
- Remove `GITHUB_TOKEN` from the bash allowlist; inject credentials only into dedicated `git` helpers (`pkg/foreman/agent/repo`).
- Use short-lived, repo-scoped tokens with minimal permissions.
- Disable bash tool by default; require explicit Agent opt-in.

---

#### F-007: Metal-agent binds inference to all interfaces; lacks shipped least-privilege RBAC

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Exploitability** | Medium–High — in LAN/Tailscale-exposed Mac deployments |
| **Locations** | `pkg/agent/executor.go:338-341`, `cmd/metal-agent/main.go:270-274,429-446`, `pkg/agent/agent.go` |

**Description:** `llama-server` is started with `--host 0.0.0.0`. The agent registers the host IP in EndpointSlices, exposing inference on the LAN. Default `--inference-service-allowlist` is empty, so the agent claims all metal-accelerator InferenceServices in the namespace. No dedicated ClusterRole/RoleBinding ships for metal-agent (unlike Foreman chart RBAC); install docs rely on the operator's kubeconfig, often over-scoped.

**Exploit scenario:** Any host on the same network segment hits the OpenAI-compatible API with no authentication. A compromised Mac host yields kubeconfig credentials with broad cluster access.

**Impact:** Unauthenticated inference access; cluster compromise via stolen kubeconfig.

**Remediation:**
- Default bind to `127.0.0.1`; expose via the existing client-proxy (`127.0.0.1:9999`) or explicit `--host-ip`.
- Ship a dedicated namespace-scoped Role/RoleBinding with minimal verbs.
- Document and encourage `--inference-service-allowlist` on multi-Mac fleets.
- Add auth/TLS in front of metal inference endpoints.

---

### MEDIUM

#### F-008: No validating admission webhooks for core inference CRDs

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — delays detection; amplifies F-003 through F-005 |
| **Locations** | `config/default/kustomization.yaml:21-23` (webhook sections commented out), `cmd/main.go` (no inference validators registered) |

**Description:** `Model`, `InferenceService`, and `ModelRouter` rely on OpenAPI schema and reconciler-time validation only. Foreman CRDs have validating webhooks; core inference CRDs do not. Malicious specs are accepted by the API server until the controller reacts.

**Remediation:** Implement validating webhooks with `failurePolicy: Fail` for dangerous field combinations (HostPath sources, privileged contexts, internal URLs).

---

#### F-009: NetworkPolicy disabled by default; inference pods not covered

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — on flat cluster networks |
| **Locations** | `charts/llmkube/values.yaml:256-259`, `charts/llmkube/templates/network-policy.yaml`, `charts/foreman/values.yaml` |

**Description:** Optional NetworkPolicies exist but default to `enabled: false`. The LLMKube chart policy covers only the controller manager pod — not inference pods, init containers, or router-proxy.

**Remediation:** Ship NetworkPolicy templates for inference and router-proxy pods. Enable by default in a secure values profile. Deny egress to cloud metadata and restrict ingress to authorized clients.

---

#### F-010: ModelRouter forwards most client headers to cloud backends

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — passive data leakage |
| **Locations** | `internal/router/backend.go:285-298,344-356` |

**Description:** `copyForwardedHeaders` forwards all non-hop-by-hop headers except `authorization` (replaced by backend credentials). Internal headers (`x-llmkube-*`, trace IDs, mesh metadata) may leak to external LLM providers.

**Remediation:** Use an allowlist of forwardable headers; strip internal classification headers and mesh metadata on cloud-tier backends.

---

#### F-011: Gate Job template allows unvalidated `make` targets

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — requires Foreman agent + LLM-influenced gate tool args |
| **Locations** | `pkg/foreman/agent/tools/gate_job_template.yaml:64-66`, `pkg/foreman/agent/tools/run_gate_job.go` |

**Description:** `checks` from JSON args are interpolated as `make {{ . }}` without strict validation. Values containing shell metacharacters could inject commands in the Job container.

**Remediation:** Validate checks against `^[a-zA-Z0-9_-]+$`; reject all other input.

---

#### F-012: PVC source path traversal (`..` not rejected)

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — requires PVC mount access |
| **Locations** | `internal/controller/source.go:62-89`, `internal/controller/model_storage.go:121-123` |

**Description:** `parsePVCSource` does not normalize or reject `..` segments. `modelPath` is built as `/model-source/{path}` without `subPath` enforcement.

**Exploit scenario:** `source: pvc://shared-pvc/../../sensitive/file.gguf` may escape the intended mount root inside the pod filesystem.

**Remediation:** Reject paths containing `..`; use `filepath.Clean` and enforce prefix under mount root; prefer `volumeMount.subPath`.

---

#### F-013: Shared per-namespace model cache PVC

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Low–Medium — requires namespace co-tenancy |
| **Locations** | `internal/controller/model_storage.go:40,276-338` |

**Description:** All InferenceService workloads in a namespace share one `llmkube-model-cache` PVC. Cache keys are the first 16 hex chars of `SHA256(source)` — predictable if the source URL is known.

**Exploit scenario:** A malicious tenant overwrites another team's cached model artifact by creating a Model with a known source, then modifying the cached file via init container writes.

**Remediation:** Per-team PVC naming; optional separate cache PVC per Model; integrity verification on cache reads.

---

#### F-014: Metal-agent accepts arbitrary host paths for MLX/vLLM runtimes

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — requires metal InferenceService create in watched namespace |
| **Locations** | `pkg/agent/executor_vllm_swift.go:194-205`, `pkg/agent/executor_omlx.go:77` |

**Description:** Absolute `ModelSource` paths are used directly on the Mac host (with `EvalSymlinks` in some paths). No requirement that models live under `--model-store`.

**Remediation:** Webhook or agent-side allowlist; reject paths outside `--model-store`; reject symlinks escaping the store root.

---

#### F-015: Arbitrary container image on all runtimes

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — by design for power users |
| **Locations** | `internal/controller/deployment_builder.go:161-164` |

**Description:** `spec.image` overrides the backend default image for any runtime. Cluster pulls and runs attacker-controlled images with default (non-privileged) security context.

**Remediation:** Operator-level image allowlist via admission policy; require `runtime: generic` for custom images.

---

#### F-016: OpenTelemetry exporter uses insecure gRPC

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Low–Medium — on untrusted networks |
| **Locations** | `cmd/main.go:71-74` |

**Description:** When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, traces are exported with `otlptracegrpc.WithInsecure()`.

**Remediation:** Support TLS credentials and `OTEL_EXPORTER_OTLP_INSECURE=false`; document secure collector setup.

---

#### F-017: AgenticTask webhook validates only on CREATE

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Low–Medium |
| **Locations** | `internal/foreman/webhook/agentictask_webhook.go:64-101` |

**Description:** `ValidateUpdate` is a no-op. `agentRef` integrity is checked only at creation; updates are not re-validated.

**Remediation:** Mark `agentRef` immutable in the CRD schema or re-validate on update.

---

#### F-018: Model source allows cleartext HTTP

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — on untrusted networks |
| **Locations** | `api/v1alpha1/model_types.go:46-47` |

**Description:** CRD pattern allows `http://` sources. Model weights can be fetched over unencrypted HTTP, enabling MITM substitution of GGUF files.

**Remediation:** Optional `https-only` operator flag; warn in CRD documentation and CLI.

---

#### F-019: CLI `cache inspect` creates unhardened inspector pods

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Low–Medium — bounded by caller's kubeconfig RBAC |
| **Locations** | `pkg/cli/cache_inspect.go:162-207` |

**Description:** Creates a fixed-name `llmkube-cache-inspector` busybox pod with no `securityContext`, no resource limits, and requires `pods/create` + `pods/exec` permissions.

**Remediation:** Add restricted securityContext, generate unique pod names, document minimum RBAC Role.

---

#### F-020: Self-update downloads from configurable URL (SHA-256 only)

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Low–Medium — requires control-plane write on release objects |
| **Locations** | `pkg/selfupdate/updater.go:141-186` |

**Description:** Binary updates download from `Target.URL` with SHA-256 verification. No hostname allowlist (e.g. GitHub releases only). Compromise of `FleetNode`/`AgentRelease` status could redirect downloads.

**Remediation:** Allowlist release artifact hostnames; add cosign/sigstore verification in addition to SHA-256.

---

#### F-021: Gate/coder Jobs clone arbitrary Git URLs with cluster egress

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Exploitability** | Medium — bounded by Job SA and NetworkPolicy |
| **Locations** | `pkg/foreman/agent/tools/gate_job_template.yaml:59-61`, `run_gate_job.go`, `run_coder_job.go` |

**Description:** Jobs clone user/LLM-influenced repositories and run `make` targets with network egress. Compromised inputs could target internal Git endpoints or supply-chain-poisoned repos.

**Remediation:** Allowlist clone hosts; run Jobs in an isolated namespace with deny-all egress except approved Git hosts; pin commit SHAs.

---

### LOW

#### F-022: GitHub token file may be world-readable

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Locations** | `pkg/foreman/agent/repo/auth.go:67-68` |

**Description:** Reads `~/.config/foreman/github-token` without checking file mode. If created as `0644`, other local users can read it.

**Remediation:** Require `0600` on the token file; warn if permissions are loose.

---

#### F-023: Helm ClusterRole missing `events` rules present in kustomize RBAC

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Locations** | `config/rbac/role.yaml` vs `charts/llmkube/templates/clusterrole.yaml` |

**Description:** RBAC drift between install paths. Events are low sensitivity but indicates inconsistent hardening and may cause permission surprises.

**Remediation:** Sync Helm ClusterRole with kustomize manifest.

---

#### F-024: HTTP health checks without explicit timeouts (some paths)

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Locations** | `pkg/agent/executor.go`, `executor_mlx_server.go`, `executor_vllm_swift.go` |

**Description:** Startup health polling uses `http.Get` to localhost without timeouts. Lower risk than controller SSRF; could hang goroutines.

**Remediation:** Use `http.Client{Timeout: ...}` consistently.

---

#### F-025: Controller ClusterRole grants cluster-wide Secret read

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Locations** | `config/rbac/role.yaml:32-36`, `charts/llmkube/templates/clusterrole.yaml:27-31` |

**Description:** Operator can list/get/watch Secrets cluster-wide (needed for router/cloud backend credentials). Cannot mutate secrets, but a compromised controller exposes all watched credentials.

**Remediation:** Scope secret reads to namespaces where ModelRouters run; use per-namespace RoleBindings in multi-tenant mode.

---

#### F-026: Foreman webhook `failurePolicy: Fail` creates admission availability dependency

| Field | Value |
|-------|-------|
| **Severity** | Low (availability; security-positive) |
| **Locations** | `charts/foreman/values.yaml` |

**Description:** Operator rollout blocks Agent/AgenticTask admission when webhooks are unavailable. Fail-closed is correct; single-replica deployments create a brief admission window.

**Remediation:** Run ≥2 operator replicas in production.

---

### INFORMATIONAL

#### F-027: Router-proxy image tag defaults to `dev`

| Field | Value |
|-------|-------|
| **Severity** | Info (supply chain hygiene) |
| **Locations** | `charts/llmkube/values.yaml` |

**Description:** Default `routerProxy.tag: "dev"` is not a pinned release artifact.

**Remediation:** Pin to release tags or image digests in production values.

---

#### F-028: gosec intentionally excludes SSRF/subprocess/path rules

| Field | Value |
|-------|-------|
| **Severity** | Info |
| **Locations** | `.golangci.yml:53-66` |

**Description:** G107 (variable URL in HTTP), G204 (subprocess), G304 (file inclusion) are excluded as "design intent." This is documented but means static analysis will not flag new instances of these patterns.

**Remediation:** Periodic manual review of HTTP fetch and subprocess call sites; consider re-enabling with targeted `#nosec` annotations only where justified.

---

#### F-029: No container image scanning in CI

| Field | Value |
|-------|-------|
| **Severity** | Info |
| **Locations** | `.github/workflows/security.yml` |

**Description:** CI runs `govulncheck` on Go dependencies but does not scan container images (Trivy/Grype) or generate SBOMs.

**Remediation:** Add image vulnerability scanning to release pipeline.

---

## Positive Security Controls

| Control | Location |
|---------|----------|
| Metrics HTTPS + Kubernetes RBAC authn/authz | `cmd/main.go:215-220`, `charts/llmkube/values.yaml` |
| Router-proxy restricted pod security context | `internal/controller/router_deployment_builder.go:221-247` |
| Default inference container drops ALL caps, no priv esc | `internal/controller/deployment_builder.go:86-95` |
| ModelRouter static fail-closed validation at reconcile time | `internal/controller/modelrouter_validation.go` |
| Foreman Agent/AgenticTask validating webhooks | `internal/foreman/webhook/` |
| Workspace path traversal protection | `pkg/foreman/agent/tools/workspace.go` |
| Bash sandbox: env allowlist, cd-guard, process groups, timeouts | `pkg/foreman/agent/tools/bash.go` |
| Metal health/client proxy bound to 127.0.0.1 | `pkg/agent/health.go:204`, `clientproxy.go:95` |
| HTTP/2 disabled by default on controller TLS | `cmd/main.go:173-186` |
| Request body cap on router-proxy (32 MiB) | `internal/router/proxy.go:171` |
| Weekly govulncheck CI | `.github/workflows/security.yml` |
| Router config JSON validated at load | `internal/router/config.go:193-235` |
| SHA-256 verification on self-update binaries | `pkg/selfupdate/updater.go` |

---

## Remediation Priority

| Priority | Actions |
|----------|---------|
| **Immediate** | Treat router classification as untrusted; add network-level auth in front of inference/router Services; enable NetworkPolicies in production. |
| **Short term** | Harden model URL fetching (SSRF guards, timeouts); restrict `file://` and privileged InferenceService fields via admission policy. |
| **Foreman** | Remove `GITHUB_TOKEN` from bash allowlist; validate gate `checks` strings. |
| **Metal** | Ship least-privilege RBAC; default inference bind to localhost. |
| **Supply chain** | Pin image digests; add container scanning to CI. |

---

## Scope Limitations

- No dynamic penetration testing or live cluster validation was performed.
- Terraform provisioning scripts (`terraform/`) were not deeply reviewed.
- Full red-team analysis of Foreman LLM tool abuse (read/write/grep orchestration) is out of scope; workspace sandboxing mitigates but does not eliminate agentic risk.
- Severity assumes multi-tenant RBAC delegation; single-admin clusters with strict PSA reduce several findings to configuration guidance.
