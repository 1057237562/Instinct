"""Self-signed HTTPS certificate generator for the Instinct WebUIs.

Streamlit serves TLS through ``--server.sslCertFile`` / ``--server.sslKeyFile``.
This script ensures a certificate pair exists — generating a self-signed one on
first use — so the launchers can enable HTTPS with zero manual steps.

Generation backends, in order:
  1. ``cryptography`` (used when the package is installed)
  2. ``openssl req -x509`` (Git-for-Windows / POSIX fallback)

If both fail, remediation steps are printed and the exit code is non-zero.
The certificate covers localhost, the machine hostname, 127.0.0.1 and ::1,
which is enough for local and trusted-intranet access over a browser that has
accepted the self-signed warning.
"""

import argparse
import ipaddress
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ascii_hostname(hostname):
    """SAN strings must be ASCII; a localized hostname is simply dropped."""
    try:
        hostname.encode("ascii")
        return hostname
    except UnicodeEncodeError:
        return ""


def _san_hosts(hostname):
    hosts = ["localhost"]
    host = _ascii_hostname(hostname)
    if host and host.lower() != "localhost":
        hosts.append(host)
    return hosts


def _generate_with_cryptography(cert_path, key_path, hostname, days):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    names = [x509.DNSName(host) for host in _san_hosts(hostname)]
    names.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
    names.append(x509.IPAddress(ipaddress.ip_address("::1")))

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # Backdate 5 minutes so a slightly skewed client clock still validates.
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _openssl_candidates():
    found = shutil.which("openssl")
    if found:
        yield found
    # Git for Windows ships openssl.exe under <Git>\usr\bin, but only <Git>\cmd
    # is usually on PATH — derive the install root from the git executable.
    git = shutil.which("git")
    if git:
        git_root = Path(git).resolve().parent.parent
        for rel in ("usr/bin/openssl.exe", "mingw64/bin/openssl.exe", "bin/openssl.exe"):
            candidate = git_root / rel
            if candidate.is_file():
                yield str(candidate)
    for base in ("C:/Program Files/Git", "C:/Program Files (x86)/Git"):
        for rel in ("usr/bin/openssl.exe", "mingw64/bin/openssl.exe"):
            candidate = Path(base) / rel
            if candidate.is_file():
                yield str(candidate)


def _generate_with_openssl(cert_path, key_path, hostname, days):
    san = ",".join(f"DNS:{host}" for host in _san_hosts(hostname))
    san += ",IP:127.0.0.1,IP:::1"
    last_error = None
    for openssl in _openssl_candidates():
        base_cmd = [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-sha256",
            "-nodes", "-days", str(days),
            "-keyout", str(key_path), "-out", str(cert_path),
            "-subj", "/CN=localhost",
        ]
        # -addext needs openssl >= 1.1.1; older builds still make a cert
        # without the SAN, so retry without it before giving up.
        for cmd in (base_cmd + ["-addext", f"subjectAltName={san}"], base_cmd):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
            except OSError as exc:
                last_error = exc
                break
            if proc.returncode == 0:
                return openssl
            last_error = RuntimeError(
                (proc.stderr or "").strip() or f"openssl exited {proc.returncode}")
    raise RuntimeError(f"no usable openssl: {last_error}")


def _print_manual_instructions(cert_path, key_path):
    print("Enable HTTPS by providing a certificate pair yourself at:", file=sys.stderr)
    print(f"  cert: {cert_path}", file=sys.stderr)
    print(f"  key : {key_path}", file=sys.stderr)
    print("Or install the optional dependency and rerun the launcher:",
          file=sys.stderr)
    print("  pip install cryptography", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate or reuse a self-signed HTTPS certificate for "
                    "the Instinct WebUIs (relative --out-dir resolves against "
                    "the repo root).")
    parser.add_argument("--out-dir", default="certs",
                        help="directory for the pair (default: certs/)")
    parser.add_argument("--name", default="chat_webui",
                        help="file stem, becomes <name>.crt / <name>.key")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate even if the pair already exists")
    parser.add_argument("--days", type=int, default=3650,
                        help="validity in days (default: 3650)")
    parser.add_argument("--hostname", default=socket.gethostname(),
                        help="extra DNS SAN besides localhost/127.0.0.1/::1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    cert_path = out_dir / f"{args.name}.crt"
    key_path = out_dir / f"{args.name}.key"

    if cert_path.is_file() and key_path.is_file() and not args.overwrite:
        print("Reusing existing HTTPS certificate:")
        print(f"  cert: {cert_path}")
        print(f"  key : {key_path}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    backend = None
    try:
        _generate_with_cryptography(cert_path, key_path, args.hostname, args.days)
        backend = "cryptography"
    except ImportError:
        pass
    except Exception as exc:
        print(f"cryptography backend failed ({exc}); trying openssl ...",
              file=sys.stderr)
    if backend is None:
        try:
            used = _generate_with_openssl(cert_path, key_path, args.hostname,
                                          args.days)
            backend = f"openssl ({used})"
        except Exception as exc:
            print(f"error: certificate generation failed: {exc}", file=sys.stderr)
            _print_manual_instructions(cert_path, key_path)
            return 1

    if not (cert_path.is_file() and key_path.is_file()):
        print("error: backend reported success but files are missing",
              file=sys.stderr)
        return 1
    if os.name == "posix":
        os.chmod(key_path, 0o600)

    print(f"Generated self-signed HTTPS certificate (backend: {backend}):")
    print(f"  cert: {cert_path}")
    print(f"  key : {key_path}")
    print(f"  SANs: {', '.join(f'DNS:{h}' for h in _san_hosts(args.hostname))}, "
          "IP:127.0.0.1, IP:::1")
    print(f"  validity: {args.days} days")
    print("Browsers will warn about the self-signed certificate; accept it to "
          "continue, or replace this pair with your own cert/key.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
