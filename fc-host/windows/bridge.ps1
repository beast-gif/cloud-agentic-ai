<#
Agentic Cloud - Phase 5, Step 5.2
Let microVMs inside the fc-host VM reach the control plane (:8000) and Postgres (:5432)
that run on THIS Windows machine - and nothing else.

How:
  * Both services stay bound to 127.0.0.1 (never exposed on Wi-Fi/LAN).
  * A Windows "portproxy" listens ONLY on the Hyper-V Default Switch address
    (e.g. 172.20.176.1) and forwards to 127.0.0.1.
  * A firewall rule allows those two ports ONLY from the Default Switch subnet
    (e.g. 172.20.176.0/20) and ONLY on that local address.

The Default Switch address can change after a Windows reboot -> just run this script again.

Usage (PowerShell as Administrator, from the project folder):
  .\fc-host\windows\bridge.ps1            # set up / refresh, prints the address to use
  .\fc-host\windows\bridge.ps1 -Remove    # undo everything
#>
param([switch]$Remove, [int[]]$Ports = @(8000, 5432))

$ErrorActionPreference = "Stop"
$RulePrefix = "AgenticCloud fc-host"

function Remove-Bridge {
    foreach ($line in (netsh interface portproxy show v4tov4) ) {
        # data lines look like:  172.20.176.1    8000        127.0.0.1       8000
        $parts = ($line -split '\s+') | Where-Object { $_ -ne '' }
        if ($parts.Count -eq 4 -and $Ports -contains [int]($parts[1] -as [int])) {
            netsh interface portproxy delete v4tov4 listenaddress=$($parts[0]) listenport=$($parts[1]) | Out-Null
            Write-Host "removed portproxy $($parts[0]):$($parts[1])"
        }
    }
    Get-NetFirewallRule -DisplayName "$RulePrefix*" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
}

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this in PowerShell as Administrator."
}

Remove-Bridge
if ($Remove) { Write-Host "bridge removed"; return }

# ---- the Default Switch address of Windows
$addr = Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "vEthernet (Default Switch)" -ErrorAction Stop |
        Select-Object -First 1
$ip = $addr.IPAddress
$prefix = $addr.PrefixLength

# network address = ip AND mask   (e.g. 172.20.189.1/20 -> 172.20.176.0/20)
$ipBytes = ([System.Net.IPAddress]::Parse($ip)).GetAddressBytes()
[Array]::Reverse($ipBytes)
$ipInt = [BitConverter]::ToUInt32($ipBytes, 0)
$maskInt = [uint32]([math]::Pow(2, 32) - [math]::Pow(2, 32 - $prefix))
$netBytes = [BitConverter]::GetBytes([uint32]($ipInt -band $maskInt))
[Array]::Reverse($netBytes)
$subnet = "$(([System.Net.IPAddress]::new($netBytes)).ToString())/$prefix"

# ---- portproxy needs the IP Helper service
Set-Service iphlpsvc -StartupType Automatic
Start-Service iphlpsvc

foreach ($p in $Ports) {
    netsh interface portproxy add v4tov4 listenaddress=$ip listenport=$p connectaddress=127.0.0.1 connectport=$p | Out-Null
    New-NetFirewallRule -DisplayName "$RulePrefix $p" -Direction Inbound -Action Allow -Protocol TCP `
        -LocalAddress $ip -LocalPort $p -RemoteAddress $subnet -Profile Any | Out-Null
    Write-Host "forwarding $($ip):$p -> 127.0.0.1:$p   (allowed only from $subnet)"
}

Write-Host ""
Write-Host "Windows address for fc-host / microVMs:  $ip"
Write-Host "use it in fc-host:   export WIN_HOST=$ip"