#!/usr/bin/env bash
# =============================================================================
# Egress lockdown for the agent container ("scope the container to the MCP
# port"). Run as container-root with CAP_NET_ADMIN by the entrypoint BEFORE
# privileges are dropped to the 'agent' user; once the drop happens the rules
# are immutable from inside (no capabilities left, no-new-privileges set).
#
# Policy — the agent needs the public internet for its own model API, so the
# default stays ACCEPT and we carve out what must NOT be reachable:
#   * the host (MCP_HOST, normally host.docker.internal): ONLY tcp:$MCP_PORT
#   * everything else private / link-local / CGNAT (the LAN): rejected
# Limitations: LAN hosts numbered with public IPv4 or global IPv6 are
# indistinguishable from the internet and stay reachable (documented in
# AUDIT.md).
#
# Only filter/OUTPUT is touched. The nat table must stay untouched — Docker's
# embedded-DNS DNAT (127.0.0.11) lives there.
#
# Environment (set by DockerManager / the entrypoint):
#   MCP_PORT – TCP port of the MCP server on the host (required).
#   MCP_HOST – Hostname/IP of the MCP server (defaults to host.docker.internal).
#
# Exits non-zero on any failure; the entrypoint treats that as fatal
# (fail-closed) and tells the user about --no-egress-lockdown.
# =============================================================================
set -euo pipefail

if [ -z "${MCP_PORT:-}" ]; then
    echo "firewall: MCP_PORT is not set" >&2
    exit 1
fi

MCP_HOST="${MCP_HOST:-host.docker.internal}"

# ---- Pick a working iptables backend ----
# Debian bookworm defaults to the nft backend. On hosts that run legacy
# iptables the nf_tables kernel modules may be absent and cannot be loaded
# from inside the container, so probe nft first and fall back to legacy.
IPT=""
IPT6=""
for candidate in iptables-nft iptables-legacy iptables; do
    if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" -L OUTPUT -n >/dev/null 2>&1; then
        IPT="$candidate"
        IPT6="${candidate/iptables/ip6tables}"
        break
    fi
done
if [ -z "$IPT" ]; then
    echo "firewall: no working iptables backend (tried nft and legacy)" >&2
    exit 1
fi

# ---- Resolve the MCP host ----
# MCP_HOST may be an IP literal (Podman-on-Windows passes the WSL gateway IP)
# or a name (host.docker.internal via extra_hosts); getent handles both, and a
# name may map to several addresses — allow the MCP port on all of them.
MCP_IPS=$(getent ahostsv4 "$MCP_HOST" 2>/dev/null | awk '{print $1}' | sort -u)
if [ -z "$MCP_IPS" ]; then
    echo "firewall: could not resolve MCP host '$MCP_HOST'" >&2
    exit 1
fi

# ---- IPv4 rules (order is load-bearing: allows before rejects) ----
# Docker Desktop's host address (192.168.65.254), WSL gateways (172.16/12) and
# pasta's DNS forward (169.254.x) all live inside ranges rejected below, so
# every allow must be appended first.
$IPT -A OUTPUT -o lo -j ACCEPT
$IPT -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

for ip in $MCP_IPS; do
    $IPT -A OUTPUT -p tcp -d "$ip" --dport "$MCP_PORT" -j ACCEPT
done

# DNS to the configured resolvers (may sit in a private range, e.g. Docker
# Desktop's 192.168.65.7 or a LAN resolver copied from the host's resolv.conf).
NAMESERVERS=$(awk '/^nameserver[ \t]/ {print $2}' /etc/resolv.conf 2>/dev/null || true)
for ns in $NAMESERVERS; do
    case "$ns" in
        *:*) # IPv6 resolver — handled by the v6 rules below.
            ;;
        *)
            $IPT -A OUTPUT -p udp -d "$ns" --dport 53 -j ACCEPT
            $IPT -A OUTPUT -p tcp -d "$ns" --dport 53 -j ACCEPT
            ;;
    esac
done

# The MCP host itself: nothing but the MCP port (even if it has a public IP).
for ip in $MCP_IPS; do
    $IPT -A OUTPUT -d "$ip" -j REJECT
done

# The private / link-local / CGNAT ranges: the host's other addresses and the
# whole (privately-numbered) LAN.
for net in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 169.254.0.0/16 100.64.0.0/10; do
    $IPT -A OUTPUT -d "$net" -j REJECT
done

# ---- IPv6 rules ----
# Mirror the shape: loopback + established + DNS allowed, ULA and link-local
# rejected. Global IPv6 stays open (indistinguishable from the internet).
# ip6tables can genuinely be unsupported (kernel without IPv6); only tolerate
# that exact case — if IPv6 works, the rules must apply (fail-closed).
if $IPT6 -L OUTPUT -n >/dev/null 2>&1; then
    $IPT6 -A OUTPUT -o lo -j ACCEPT
    $IPT6 -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    for ns in $NAMESERVERS; do
        case "$ns" in
            *:*)
                $IPT6 -A OUTPUT -p udp -d "$ns" --dport 53 -j ACCEPT
                $IPT6 -A OUTPUT -p tcp -d "$ns" --dport 53 -j ACCEPT
                ;;
        esac
    done
    for net in fc00::/7 fe80::/10; do
        $IPT6 -A OUTPUT -d "$net" -j REJECT
    done
elif [ -e /proc/net/if_inet6 ]; then
    echo "firewall: IPv6 is enabled but ip6tables is not functional" >&2
    exit 1
fi

echo "firewall: egress locked down (host reachable only on tcp:${MCP_PORT})"
