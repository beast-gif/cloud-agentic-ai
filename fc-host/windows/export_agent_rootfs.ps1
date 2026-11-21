<#
Agentic Cloud - Phase 5, Step 5.2
Export the agent's Docker image as a plain filesystem tarball and copy it into the fc-host VM.
fc-host turns it into an ext4 disk image for Firecracker (fc-host/build_agent_rootfs.sh).

  docker image  --docker create-->  stopped container  --docker export-->  agent-rootfs.tar
  (layers)                          (merged filesystem)                     (one flat tree)

Usage (normal PowerShell, project folder; Docker Desktop running):
  .\fc-host\windows\export_agent_rootfs.ps1 -FcHost amrit@172.20.189.74
Rebuild the image first if agent_runtime/ changed:
  docker build -t agentic-cloud/agent:phase1 ./agent_runtime
#>
param(
    [Parameter(Mandatory = $true)][string]$FcHost,          # e.g. amrit@172.20.189.74
    [string]$Image = "agentic-cloud/agent:phase1",
    [string]$Out = "$env:TEMP\agent-rootfs.tar"
)
$ErrorActionPreference = "Stop"

$name = "fc-export-$(Get-Random)"
Write-Host "creating container from $Image ..."
docker create --name $name $Image | Out-Null
try {
    Write-Host "exporting filesystem to $Out ..."
    docker export $name -o $Out
} finally {
    docker rm $name | Out-Null
}
$sizeMb = [math]::Round((Get-Item $Out).Length / 1MB)
Write-Host "tarball: $sizeMb MB"

Write-Host "copying to $($FcHost):~/fc/images/ ..."
ssh $FcHost "mkdir -p ~/fc/images"
scp $Out "$($FcHost):~/fc/images/agent-rootfs.tar"
Remove-Item $Out
Write-Host "done. In fc-host run:  ~/fc-host/build_agent_rootfs.sh"