---
sidebar_position: 7.65
title: Deployment Behind a Reverse Proxy
description: Give CodefyUI its own hostname, bind it to loopback, and put nginx in front for TLS and single sign-on -- with a tested systemd unit and nginx config.
---

# Deployment Behind a Reverse Proxy

CodefyUI has no user accounts. There is no login, no per-person ownership, and
nothing in the backend that could answer "who ran this?" -- see
[Shared Instances](./shared-instances) for what that means in practice. So when
a team wants real authentication in front of a shared instance, the answer is
not a CodefyUI feature. It is a reverse proxy that already speaks your identity
system, with CodefyUI bound to loopback behind it.

This page is how to do that. Everything below was tested against a real nginx
and a real systemd; the caveats are called out where they exist.

```text
                     TLS + your SSO                plain HTTP, loopback only
  browser  ────────────────────────────►  nginx  ─────────────────────────►  cdui start
           https://codefyui.example.com    :443                              127.0.0.1:8000
```

Three things have to be true, and the second one is the one everybody misses.

1. CodefyUI owns a **hostname**, not a subpath.
2. That hostname is on CodefyUI's **Host whitelist**, or every request fails.
3. CodefyUI binds **loopback**, so the proxy is the only way in.

## 1. Give it a hostname, not a subpath

`https://codefyui.example.com/` works. `https://tools.example.com/codefyui/`
does not.

The built frontend emits root-absolute asset URLs -- `index.html` links
`/assets/index-*.js`, the backend mounts static files at the literal `/assets`,
and the API client's base path is the literal `/api`. There is no `base` option
and no build-time environment variable to change any of it, so a subpath would
require rebuilding the frontend from a fork.

:::note Why this is not on the roadmap
Owning a vhost is the normal deployment shape for an application like this, and
it costs a DNS record. Subpath support would mean a build knob threaded through
the frontend, the API client and the backend's static mounts, for a deployment
most teams do not need. If you genuinely cannot allocate a hostname, open an
issue and say why -- that is the input that would change the decision.
:::

## 2. Whitelist the public hostname

**This is the step that turns a working install into a blank browser page.**

CodefyUI checks the `Host` header on every request and answers anything it does
not recognise with `421 Misdirected Request`:

```json
{"detail": "Misdirected Request (Host not allowed)"}
```

That check is deliberate -- it is what stops DNS rebinding -- but it runs
outermost, so it applies to **the page itself**, not just the API. Behind a
proxy the `Host` is your public name, which is not on the default whitelist, so
the browser gets JSON instead of `index.html` and renders nothing. There is no
error in the UI, because there is no UI.

Fix it with one environment variable:

```bash
export CODEFYUI_EXTRA_ALLOWED_HOSTS="codefyui.example.com"
```

| Rule | Detail |
| --- | --- |
| Separator | Comma. Surrounding spaces are trimmed: `a, b` is fine. |
| Matching | Exact string, case-insensitive. **No wildcards** -- `*.example.com` never matches. |
| Port | Part of the string. Whitelist exactly what the browser puts in the address bar. |
| Lifetime | Read at startup only. It is not persisted -- set it every time the server starts. |

:::warning The port rule bites in testing
On `:443` or `:80` the browser sends `Host: codefyui.example.com` with no port,
so whitelist the bare name. On any other port it sends
`Host: codefyui.example.com:8443`, and you must whitelist **that**, port
included. Listing `codefyui.example.com` alone will not match, and
`codefyui.example.com:443` will not match either. Whitelisting both forms is
harmless and saves an afternoon:

```bash
export CODEFYUI_EXTRA_ALLOWED_HOSTS="codefyui.example.com,codefyui.example.com:8443"
```
:::

The WebSocket handshake performs the same `Host` check independently, and then
compares the browser's `Origin` against that same `Host`. Both are satisfied by
the one variable **as long as the proxy forwards `Host` unchanged** -- which is
why the nginx config below uses `$http_host` and not `$host`. If you rewrite
`Host` at the proxy, the canvas will load and then silently never connect.

## 3. Bind loopback

```bash
cdui start --host 127.0.0.1 --port 8000
```

`127.0.0.1` is already the default, so the interesting instruction is the
negative one: **do not use `--host 0.0.0.0`** once a proxy is in front. A LAN
bind hands full control of the instance to anyone who can reach the port, with
no credentials at all -- that trade is explained in
[Publish](./publish#6-serving-on-your-lan), and it is the thing the proxy exists
to take away.

`cdui start` is a real daemon: it detaches from the terminal, writes a pidfile,
and `cdui stop` terminates its whole process group with `SIGTERM` before
escalating to `SIGKILL`. Nothing ever opens a browser, so it is safe on a
headless server. What it does not have is supervision -- that is systemd's job,
below.

## 4. Passing uvicorn flags: `cdui start -- ...`

Everything after a bare `--` is forwarded to uvicorn verbatim:

```bash
cdui start --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1
```

This is how `--proxy-headers`, `--root-path`, `--forwarded-allow-ips` and
`--timeout-keep-alive` become reachable without invoking uvicorn by hand -- and
invoking it by hand costs you the pidfile, `cdui status` and `cdui stop`.

The separator cuts in both directions. `cdui start` reads its own flags only
from the part **before** `--`, so a forwarded `-f` or `--project` can never be
mistaken for one of CodefyUI's, and a future `cdui start` flag can never shadow
a uvicorn one.

:::note --host and --port are refused after the separator
`cdui start` records the bind address in its state file and in the child's
environment, and the Host whitelist is derived from it. A second copy arriving
through the passthrough would win inside uvicorn and desync all of that, so it
exits with code 2 and tells you to use `cdui start --host` instead.
:::

:::tip --proxy-headers is what makes the OpenAPI document say `https`
Set it. Behind TLS it is not merely tidy, it is load-bearing.

The one place CodefyUI bakes a scheme into a response is the OpenAPI document
for a published app, and its `servers[].url` (plus the two copy-paste `curl`
snippets) is built from the scheme of the incoming request. uvicorn rewrites
that scheme from `X-Forwarded-Proto` **only** when started with
`--proxy-headers`, and **only** for a peer inside `--forwarded-allow-ips`.

So: with both flags set, a document fetched over HTTPS advertises
`https://your-host/api/apps/<slug>` and Swagger UI's "Try it out" works.
Without them, it advertises `http://` even when reached over HTTPS -- the
browser then blocks the call as mixed content, and a generated client gets the
wrong base URL.

CodefyUI deliberately never reads `X-Forwarded-Proto` itself. If it did, any
client could forge the header and dictate the URL your published app advertises
to every integrator who fetches the document; leaving it to uvicorn keeps the
"is this hop trusted?" decision in the one place that has been told the answer.

**Mind `--forwarded-allow-ips` if your proxy is not on this machine.** It
defaults to `127.0.0.1`, so a proxy in another container or on another host is
*not* trusted, its `X-Forwarded-Proto` is ignored, and the document quietly goes
back to advertising `http://` -- with no error anywhere. Set it to the proxy's
address (the `nginx` example below terminates on the same host, which is why
`127.0.0.1` is correct there).
:::

:::note WebSocket message size
`cdui start` and `cdui dev` pass uvicorn `--ws-max-size`, derived from
`CODEFYUI_WS_MAX_MESSAGE_BYTES` (default: whatever `CODEFYUI_MAX_RUN_BODY_BYTES`
is, i.e. 64 MB). This is the largest graph the canvas may send over
`/ws/execution`; uvicorn's own default is 16 MB, which is *stricter* than the
HTTP body ceiling and would refuse graphs the REST API accepts.

Because `cdui` derives the flag, `--ws-max-size` is refused after `--` -- set
the environment variable instead. If you invoke uvicorn by hand you get its
16 MB back unless you pass the flag yourself.

Note your proxy has a say too: nginx's `client_max_body_size` bounds HTTP
bodies, and a WebSocket frame that large needs `proxy_read_timeout` headroom to
finish arriving.
:::

## A systemd unit

Tested: installed, `systemctl enable`d, started and stopped under a real
systemd, with `systemd-analyze verify` clean.

```ini
[Unit]
Description=CodefyUI
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=codefyui
Group=codefyui
WorkingDirectory=/opt/codefyui

# The public hostname, exactly as the browser will send it in Host. Without
# this, every request gets 421 -- including the page -- and the browser shows
# a blank screen.
Environment=CODEFYUI_EXTRA_ALLOWED_HOSTS=codefyui.example.com

# Anything secret belongs here rather than in Environment= above, which is
# readable by anyone who can run `systemctl show`. The leading '-' makes the
# file optional. Suggested mode 0640, owned by root:codefyui.
EnvironmentFile=-/etc/codefyui/codefyui.env

# --foreground on purpose: systemd is the supervisor, so cdui must not
# daemonize away from it. Bind loopback -- nginx is the only way in.
ExecStart=/opt/codefyui/cdui start --foreground --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1

Restart=on-failure
RestartSec=5s
# Draining an in-flight run takes longer than the 90s default allows.
TimeoutStopSec=120s

# Conservative hardening. Deliberately NOT PrivateDevices (breaks GPU access),
# ProtectHome or ProtectSystem=strict (both break writes to the install tree
# and the user data dir).
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo install -m 644 codefyui.service /etc/systemd/system/codefyui.service
sudo systemctl daemon-reload
sudo systemctl enable --now codefyui
systemctl status codefyui
journalctl -u codefyui -f
```

`--foreground` is the load-bearing flag. `cdui start`'s default daemon mode
double-forks, which is exactly what systemd does not want from a `Type=exec`
service. Note also that CodefyUI's own stdout goes to the journal, so
`journalctl -u codefyui` is where startup errors and the effective Host
whitelist appear -- but per-request lines do not, for the reason in the next
section but one.

## An nginx site

Tested: `nginx -t` clean, and every request below was actually made through it
to a running CodefyUI over TLS.

```nginx
# /etc/nginx/sites-available/codefyui  (symlink it into sites-enabled/)

# Both of these belong at http{} scope, which is where Debian/Ubuntu's
# sites-enabled/ include already puts you.

# Maps the Upgrade header onto Connection for the WebSocket handshake.
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

# Same as `combined`, but logs $uri (path only) instead of $request (which
# includes the query string). The WebSocket URL carries the session token as
# ?token=..., and the default format would write that credential into a file
# that usually gets shipped to a central log collector.
log_format codefyui_noquery
    '$remote_addr - $remote_user [$time_local] '
    '"$request_method $uri $server_protocol" '
    '$status $body_bytes_sent "$http_referer" "$http_user_agent"';

server {
    listen 80;
    listen [::]:80;
    server_name codefyui.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    http2 on;
    server_name codefyui.example.com;

    ssl_certificate     /etc/letsencrypt/live/codefyui.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/codefyui.example.com/privkey.pem;

    # Datasets and model checkpoints move through here. nginx's default
    # client_max_body_size is 1 MB and rejects the rest with a bare 413.
    client_max_body_size 2g;

    # This is your access log. CodefyUI does not write one.
    access_log /var/log/nginx/codefyui.access.log codefyui_noquery;
    error_log  /var/log/nginx/codefyui.error.log;

    # $http_host, NOT $host: it forwards the Host header byte-for-byte,
    # including any non-default port. The backend compares Host against its
    # whitelist by exact string, and the WebSocket handshake compares Origin
    # against that same Host. $host strips the port, which breaks both on any
    # listen port other than 443.
    proxy_set_header Host              $http_host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # /ws/execution carries every run's progress. Without these three lines
    # the canvas loads and then never reports anything.
    proxy_http_version 1.1;
    proxy_set_header Upgrade    $http_upgrade;
    proxy_set_header Connection $connection_upgrade;

    # A training run holds its socket open for as long as it trains and
    # streams as it goes. The 60s default would cut it off mid-epoch.
    proxy_read_timeout 24h;
    proxy_send_timeout 24h;
    proxy_buffering    off;

    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

Add your SSO in front of `location /` in whatever form your organisation uses --
`auth_request` against an OIDC helper, a vouch-style forward-auth, or your
vendor's nginx module. CodefyUI does not need to know it is there; it has no
identity model to integrate with, and that is the whole point.

## TLS terminates at the proxy

Nothing has to be rebuilt or reconfigured in the frontend for HTTPS. The
WebSocket URL is derived from `window.location` at runtime, so `wss://` follows
`https://` automatically and the port comes along with it. There is no
hardcoded host or scheme in the shipped bundle -- the only `ws://localhost:8000`
in the repository belongs to the Vite dev proxy, which never ships.

CodefyUI itself does not terminate TLS and has no certificate options. Keep
plain HTTP on the loopback hop.

## The proxy is also your access log

**CodefyUI does not write an HTTP access log.** Uvicorn's `uvicorn.access`
logger is raised to `WARNING` at startup, and access records are emitted at
`INFO`, so every per-request line is discarded. There is no request-logging
middleware anywhere in the backend either. Verified end to end: a batch of
requests through the proxy produced one nginx log line each and nothing at all
from CodefyUI.

So the `access_log` line in the nginx config above is not decoration. It is the
only record that a request happened, and without it a shared instance has no
audit trail at all.

:::warning Do not log the query string
The WebSocket URL carries the session token as `?token=...`. With nginx's
default `combined` format that credential lands in the access log in plaintext,
and the session token is enough to take over the instance. The
`codefyui_noquery` format above logs `$uri` instead of `$request` to prevent it.
If you use a different proxy, do the equivalent there.
:::

What CodefyUI *does* log -- startup, the effective Host whitelist, rejected
`Host` values, warnings and errors -- goes to stderr, which means the journal
under systemd. Set `CODEFYUI_LOG_DIR` to also get a rotating file (10 MB, five
generations).

## Authentication is the proxy's job, and it has limits

The proxy can decide **whether** someone reaches CodefyUI. It cannot make
CodefyUI treat two authenticated people differently, because there is no user
concept for it to hang off.

Concretely, once someone is through your SSO:

- Every saved graph, model, dataset and run record is visible and editable by
  everyone.
- Ambient credentials are instance-wide. A ChatGPT sign-in, `OPENAI_API_KEY` or
  `ANTHROPIC_API_KEY` in the environment, and Kaggle credentials all belong to
  the instance, not to a person -- one person signs in and everybody's graphs
  bill to them, with nothing recording who spent what.
- The package-install log is an open read. `GET /api/packs/jobs/{id}/events`
  takes no session token, exactly like `GET /api/runs/{id}/events` -- both are
  reads, and the Package Center polls them to draw its progress bar. STARTING
  an install is still guarded (the session token, plus a loopback bind unless
  `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` opts back in), but the log the install
  leaves behind names the interpreter it ran against -- the venv path in `uv`'s
  argv -- and carries `uv`'s own output verbatim. On a LAN bind, anyone who can
  reach the port can read it.

[Shared Instances](./shared-instances) covers those credentials in detail,
including the fallback order and where each is stored. Read it before you give a
team the URL. Its answer to "we need per-person attribution" still holds here:
run one instance per person, with separate environment files, separate
`CODEFYUI_USER_DATA_DIR` values and separate ports -- with the proxy in front of
each.

## What CodefyUI sends out, and to whom

Worth stating plainly, because it usually comes up in the same review as
"can we put this on a shared server":

**There is no telemetry, no analytics, no phone-home and no startup update
check.** The server makes no outbound request on startup or in the background,
and the browser UI makes no request to any origin except the one it was served
from. There is no analytics SDK in either dependency tree.

Traffic leaves the machine only when a person asks for it:

| What | When |
| --- | --- |
| LLM providers (OpenAI, Anthropic, OpenRouter, ChatGPT, or a URL you supply) | Running an `LLMChat` node, or the model list in settings. Off unless a key or sign-in is configured. |
| Dataset and model downloads (Kaggle, Hugging Face, torchvision) | Running a graph that contains one of those nodes. |
| `github.com` | `cdui install`, `cdui update`, `cdui plugin install`. |
| `astral.sh` | Only if `uv` is missing from `PATH`, which a normal install rules out. A one-time toolchain download, not a report. |

An air-gapped install is therefore a matter of not using those nodes, not of
disabling a reporting channel.

## Checklist

- [ ] A DNS name pointing at the proxy, not a subpath.
- [ ] `CODEFYUI_EXTRA_ALLOWED_HOSTS` set to that name, with the port if the
      proxy does not listen on 443.
- [ ] `cdui start --host 127.0.0.1`, never `0.0.0.0`.
- [ ] systemd owns the process, with `--foreground`.
- [ ] `proxy_set_header Host $http_host` and the three WebSocket lines.
- [ ] `client_max_body_size` raised past 1 MB.
- [ ] Proxy read timeout longer than your longest training run.
- [ ] Access log on, query string off.
- [ ] Everyone behind the SSO has been told they share one identity --
      [Shared Instances](./shared-instances).
