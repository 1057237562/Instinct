# Online relay and monitor integration

Deployment target: `81.68.133.63`, existing monitor `/opt/ssh-log-monitor`, port 8080.

- Monitor's **数据与训练** view embeds `/relay/?managed=1`. It is available even if
  the monitored SSH target is offline, and uses the monitor's existing admin login.
- Node proxies `/relay/` to `127.0.0.1:7800`, before body parsers. Upload bodies and
  Range responses stream unchanged; the backend API token stays on the server.
- `/relay/pull/...` GET/HEAD requests use the relay's per-task pull query token.
  They never receive a service API token from the proxy.
- Relay service: `instinct-relay.service`, executable `/opt/instinct-relay/bin/data_relay`,
  config `/opt/instinct-relay/config.toml`, runtime `/var/lib/instinct-relay`.
  `/etc/instinct-relay/proxy.json` contains the internal upstream token (0600).
- Code sync source and Config WebUI cache: `/opt/instinct-relay/workspace`.
  This is a deployment snapshot; updating local training code requires republishing
  the workspace before syncing it onward to AutoDL.
- Dataset files, cached upload chunks, model weights and checkpoints are never
  included by `prepare_online.py`. Existing AutoDL datasets are accessed in place.

Prepare an isolated checkout of the existing monitor without its credentials:

```powershell
python data_relay/deploy/integrate_monitor.py <monitor-checkout>
node data_relay/tests/test_monitor_proxy.cjs <monitor-checkout>
node data_relay/tests/test_monitor_integration.cjs <monitor-checkout>
cargo zigbuild --target x86_64-unknown-linux-gnu.2.35 --release --manifest-path data_relay/Cargo.toml
python data_relay/deploy/prepare_online.py <private-staging-directory>
```

The release archive contains runtime credentials: keep it in a private directory,
transfer only via SSH, and install configuration files with mode 0600. Preserve
the monitor's existing encrypted credentials and administrator password hash.
Back up its source and unit before running its existing deployment command.

The live monitor distribution excludes its `test/` folder. The integration tests
above verify real admin sessions, CSRF handling, iframe/assets/API prefixes, hash
workers, streaming uploads and Range responses independently of that absent suite.

Check after deployment:

```bash
systemctl status instinct-relay ssh-log-monitor
ss -lntp '( sport = :7800 or sport = :8080 )'
```

Port 7800 must remain loopback-only. Do not open a new public relay port.

The monitor's “切换服务器” picker and the relay instance selector now share a
serialized backend operation. It resolves the selected instance without SSH or
booting it, replaces the monitor's encrypted credentials, persists the relay
selection, drops old file/log sessions and restarts the Streamlit tunnel.
An instance without an SSH address clears the old target and remains selected;
refresh/reselect it once its connection details become available. Manual SSH
settings explicitly detach the browser from the AutoDL selection.

`/api/integration/ssh-target` contains credentials for backend use only and is
blocked by the public proxy, including for logged-in administrators. Never expose
it through another proxy. Existing task/queue instance snapshots are unchanged.
Verify selection changes with `test_server_selection.cjs` and
`test_server_selection_ui.cjs <monitor-checkout>`; these use only fake instances.
