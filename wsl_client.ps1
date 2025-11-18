# Pandora AIOS WSL Access Client for Windows PowerShell
# Usage: .\wsl_client.ps1 -Command "ls -la" -Token "your-token-here"

param(
    [string]$Host = "localhost",
    [int]$Port = 9000,
    [string]$Token = "",
    [string]$Command = "",
    [switch]$Status,
    [switch]$Interactive
)

# ANSI Color codes for PowerShell
$Colors = @{
    Reset = "`e[0m"
    Red = "`e[91m"
    Green = "`e[92m"
    Yellow = "`e[93m"
    Blue = "`e[94m"
    Cyan = "`e[96m"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "Reset"
    )
    Write-Host "$($Colors[$Color])$Message$($Colors.Reset)"
}

function Connect-WSLTerminal {
    param(
        [string]$ServerHost,
        [int]$ServerPort
    )
    
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect($ServerHost, $ServerPort)
        
        Write-ColorOutput "Connected to WSL Terminal at ${ServerHost}:${ServerPort}" "Green"
        return $client
    }
    catch {
        Write-ColorOutput "Failed to connect: $_" "Red"
        return $null
    }
}

function Send-JSON {
    param(
        [System.Net.Sockets.TcpClient]$Client,
        [hashtable]$Data
    )
    
    $stream = $Client.GetStream()
    $json = $Data | ConvertTo-Json -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    
    # Send length (4 bytes, big-endian)
    $length = $bytes.Length
    $lengthBytes = [BitConverter]::GetBytes($length)
    [Array]::Reverse($lengthBytes)
    $stream.Write($lengthBytes, 0, 4)
    
    # Send message
    $stream.Write($bytes, 0, $bytes.Length)
}

function Receive-JSON {
    param(
        [System.Net.Sockets.TcpClient]$Client
    )
    
    $stream = $Client.GetStream()
    
    # Read length (4 bytes)
    $lengthBytes = New-Object byte[] 4
    $stream.Read($lengthBytes, 0, 4) | Out-Null
    [Array]::Reverse($lengthBytes)
    $length = [BitConverter]::ToInt32($lengthBytes, 0)
    
    # Read message
    $messageBytes = New-Object byte[] $length
    $bytesRead = 0
    while ($bytesRead -lt $length) {
        $chunk = $stream.Read($messageBytes, $bytesRead, $length - $bytesRead)
        $bytesRead += $chunk
    }
    
    $json = [System.Text.Encoding]::UTF8.GetString($messageBytes)
    return $json | ConvertFrom-Json
}

function Send-Authentication {
    param(
        [System.Net.Sockets.TcpClient]$Client,
        [string]$AuthToken
    )
    
    Write-ColorOutput "Authenticating..." "Yellow"
    
    $authRequest = @{
        type = "auth"
        token = $AuthToken
    }
    
    Send-JSON -Client $Client -Data $authRequest
    $response = Receive-JSON -Client $Client
    
    if ($response.success) {
        Write-ColorOutput "✓ Authentication successful" "Green"
        return $true
    }
    else {
        Write-ColorOutput "✗ Authentication failed: $($response.message)" "Red"
        return $false
    }
}

function Invoke-RemoteCommand {
    param(
        [System.Net.Sockets.TcpClient]$Client,
        [string]$Cmd
    )
    
    Write-ColorOutput "`nExecuting: $Cmd" "Cyan"
    
    $execRequest = @{
        type = "execute"
        command = $Cmd
    }
    
    Send-JSON -Client $Client -Data $execRequest
    $response = Receive-JSON -Client $Client
    
    if ($response.success) {
        Write-ColorOutput "`n--- Output ---" "Blue"
        if ($response.stdout) {
            Write-Host $response.stdout
        }
        
        if ($response.stderr) {
            Write-ColorOutput "`n--- Errors ---" "Yellow"
            Write-Host $response.stderr
        }
        
        Write-ColorOutput "`nReturn Code: $($response.returncode)" "Cyan"
    }
    else {
        Write-ColorOutput "Command failed: $($response.error)" "Red"
    }
}

function Get-ServerStatus {
    param(
        [System.Net.Sockets.TcpClient]$Client
    )
    
    $statusRequest = @{
        type = "status"
    }
    
    Send-JSON -Client $Client -Data $statusRequest
    $response = Receive-JSON -Client $Client
    
    Write-ColorOutput "`n=== Server Status ===" "Cyan"
    Write-Host "Session ID: $($response.session_id)"
    Write-Host "Connected Clients: $($response.connected_clients)"
    Write-Host "Commands Executed: $($response.command_count)"
    Write-Host "Uptime: $([math]::Round($response.uptime, 2)) seconds"
    
    if ($response.clients -and $response.clients.Count -gt 0) {
        Write-ColorOutput "`nActive Clients:" "Blue"
        foreach ($client in $response.clients) {
            $authStatus = if ($client.authenticated) { "✓" } else { "✗" }
            Write-Host "  [$authStatus] $($client.id) - Connected: $($client.connected_at)"
        }
    }
}

function Start-InteractiveSession {
    param(
        [System.Net.Sockets.TcpClient]$Client
    )
    
    Write-ColorOutput "`n=== Interactive Mode ===" "Cyan"
    Write-ColorOutput "Type 'exit' to quit, 'help' for commands`n" "Yellow"
    
    while ($true) {
        Write-Host "wsl> " -NoNewline -ForegroundColor Green
        $input = Read-Host
        
        if ($input -eq "exit" -or $input -eq "quit") {
            Write-ColorOutput "Goodbye!" "Cyan"
            break
        }
        
        if ($input -eq "help") {
            Write-ColorOutput "`nAvailable commands:" "Blue"
            Write-Host "  status  - Show server status"
            Write-Host "  help    - Show this help"
            Write-Host "  exit    - Exit interactive mode"
            Write-Host "  <cmd>   - Execute any Linux command on WSL`n"
            continue
        }
        
        if ($input -eq "status") {
            Get-ServerStatus -Client $Client
            continue
        }
        
        if ($input.Trim() -ne "") {
            Invoke-RemoteCommand -Client $Client -Cmd $input
        }
    }
}

# Main script execution
Write-ColorOutput "`n╔══════════════════════════════════════════════════════════╗" "Cyan"
Write-ColorOutput "║   Pandora AIOS WSL Access Terminal - Windows Client    ║" "Cyan"
Write-ColorOutput "╚══════════════════════════════════════════════════════════╝`n" "Cyan"

# Connect to server
$client = Connect-WSLTerminal -ServerHost $Host -ServerPort $Port

if ($client -eq $null) {
    exit 1
}

# Receive welcome message
$welcome = Receive-JSON -Client $client
Write-ColorOutput "Server: $($welcome.message)" "Blue"
Write-ColorOutput "Session: $($welcome.session_id)" "Blue"

if ($welcome.auth_required) {
    Write-ColorOutput "`n⚠ Authentication required" "Yellow"
    
    if ($Token -eq "") {
        Write-ColorOutput "Please provide authentication token with -Token parameter" "Red"
        $client.Close()
        exit 1
    }
    
    $authenticated = Send-Authentication -Client $client -AuthToken $Token
    if (-not $authenticated) {
        $client.Close()
        exit 1
    }
}

# Execute requested operation
try {
    if ($Status) {
        Get-ServerStatus -Client $client
    }
    elseif ($Interactive) {
        Start-InteractiveSession -Client $client
    }
    elseif ($Command -ne "") {
        Invoke-RemoteCommand -Client $client -Cmd $Command
    }
    else {
        Write-ColorOutput "`nNo operation specified. Use -Command, -Status, or -Interactive" "Yellow"
        Write-Host "`nExamples:"
        Write-Host "  .\wsl_client.ps1 -Token 'xxx' -Command 'ls -la'"
        Write-Host "  .\wsl_client.ps1 -Token 'xxx' -Status"
        Write-Host "  .\wsl_client.ps1 -Token 'xxx' -Interactive"
    }
}
finally {
    $client.Close()
    Write-ColorOutput "`nConnection closed.`n" "Cyan"
}
