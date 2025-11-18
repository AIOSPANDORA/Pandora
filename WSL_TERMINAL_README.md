# Pandora AIOS WSL Access Terminal

## Overview

The WSL Access Terminal provides a seamless bridge between Windows and Windows Subsystem for Linux (WSL), enabling full control and interaction with Pandora AIOS from Windows environments.

## Features

### Core Capabilities
- **Bidirectional Communication**: Send commands from Windows to WSL and receive results
- **Secure Authentication**: Token-based authentication system
- **File Transfer**: Upload/download files between Windows and WSL
- **Port Forwarding**: Network port management and forwarding
- **Real-time Diagnostics**: Run Pandora diagnostics from Windows
- **Session Management**: Persistent sessions with history
- **Interactive Mode**: Command-line interface for direct interaction

### Security
- Token-based authentication
- Encrypted connections
- Session logging
- Client tracking
- Configurable access control

## Installation

### WSL (Linux) Side

1. Ensure Python 3 is installed:
```bash
python3 --version
```

2. Start the WSL Access Terminal server:
```bash
python3 wsl_access_terminal.py --mode server --host 0.0.0.0 --port 9000
```

3. Note the authentication token displayed on startup

### Windows Side

#### Option 1: PowerShell (Recommended)

1. Open PowerShell
2. Navigate to the Pandora directory
3. Run the client:
```powershell
.\wsl_client.ps1 -Token "your-token-here" -Interactive
```

#### Option 2: Batch File

1. Edit `wsl_client.bat` and set your token
2. Double-click to run in interactive mode
3. Or use from command line:
```batch
wsl_client.bat --token your-token ls -la
```

#### Option 3: Python Client

```bash
python wsl_access_terminal.py --mode client --host localhost --port 9000 --token your-token --command "ls -la"
```

## Usage

### Server Mode (WSL)

Start the server:
```bash
python3 wsl_access_terminal.py --mode server
```

With custom settings:
```bash
python3 wsl_access_terminal.py --mode server --host 0.0.0.0 --port 9000
```

### Interactive Server Mode

Once started, the server provides an interactive console:

```
pandora-wsl> status        # Show server status
pandora-wsl> clients       # List connected clients
pandora-wsl> history       # Show command history
pandora-wsl> exec ls -la   # Execute local command
pandora-wsl> help          # Show help
pandora-wsl> exit          # Stop server
```

### Client Mode (Windows)

#### Execute Single Command

PowerShell:
```powershell
.\wsl_client.ps1 -Token "abc123" -Command "ls -la /home"
```

Batch:
```batch
wsl_client.bat --token abc123 ls -la /home
```

#### Interactive Mode

PowerShell:
```powershell
.\wsl_client.ps1 -Token "abc123" -Interactive
```

Then use the interactive prompt:
```
wsl> ls -la
wsl> cd /home/user
wsl> cat file.txt
wsl> status
wsl> exit
```

#### Get Server Status

```powershell
.\wsl_client.ps1 -Token "abc123" -Status
```

### File Transfer

Upload file from Windows to WSL:
```json
{
    "type": "file_transfer",
    "operation": "upload",
    "path": "/home/user/file.txt",
    "content": "base64_encoded_content"
}
```

Download file from WSL to Windows:
```json
{
    "type": "file_transfer",
    "operation": "download",
    "path": "/home/user/file.txt"
}
```

### Run Diagnostics

Request diagnostic report:
```json
{
    "type": "diagnostics"
}
```

## Protocol

### Message Format

All messages are JSON objects sent with a 4-byte big-endian length prefix:

```
[4 bytes: message length][JSON message]
```

### Request Types

#### Authentication
```json
{
    "type": "auth",
    "token": "authentication_token"
}
```

#### Execute Command
```json
{
    "type": "execute",
    "command": "ls -la"
}
```

#### Get Status
```json
{
    "type": "status"
}
```

#### Diagnostics
```json
{
    "type": "diagnostics"
}
```

#### Ping
```json
{
    "type": "ping"
}
```

### Response Format

All responses include a `type` field indicating response type:

```json
{
    "type": "execute_response",
    "success": true,
    "command": "ls -la",
    "stdout": "output here",
    "stderr": "",
    "returncode": 0
}
```

## Configuration

### Server Configuration

Default settings:
- Host: `0.0.0.0` (all interfaces)
- Port: `9000`
- Authentication: Enabled
- Log path: `/tmp/wsl_terminal_{session_id}.log`

### Client Configuration

Edit `wsl_client.bat` for default settings:
```batch
SET HOST=localhost
SET PORT=9000
SET TOKEN=your_default_token_here
```

Or use command-line parameters:
```powershell
.\wsl_client.ps1 -Host "192.168.1.100" -Port 9000 -Token "token"
```

## Security Considerations

1. **Token Security**: Keep your authentication token secret
2. **Network Exposure**: Be cautious when binding to `0.0.0.0`
3. **Command Execution**: Server executes commands with its user privileges
4. **File Access**: Server has access to files readable by its user
5. **Logging**: All commands are logged for audit purposes

### Best Practices

- Use strong, random tokens
- Bind to `localhost` when only local access is needed
- Run server with minimal necessary privileges
- Regularly rotate authentication tokens
- Monitor logs for suspicious activity
- Use firewall rules to restrict access

## Troubleshooting

### Connection Refused

**Problem**: Windows client cannot connect to WSL server

**Solutions**:
1. Ensure server is running in WSL
2. Check firewall settings
3. Verify port is not already in use
4. Try binding to specific IP instead of 0.0.0.0

### Authentication Failed

**Problem**: Authentication token rejected

**Solutions**:
1. Copy token exactly from server startup message
2. Check for extra spaces or line breaks
3. Restart server to get new token

### Command Timeout

**Problem**: Commands taking too long to execute

**Solutions**:
1. Increase timeout in code (default 30s)
2. Run long commands asynchronously
3. Check WSL system resources

### File Transfer Fails

**Problem**: Cannot transfer files

**Solutions**:
1. Check file permissions
2. Verify path exists and is accessible
3. Ensure sufficient disk space
4. Check base64 encoding is correct

## Advanced Usage

### Automated Scripts

Create Windows scripts that automate WSL tasks:

```powershell
# backup.ps1
$token = "your-token"
$files = @("/etc/config", "/home/user/data")

foreach ($file in $files) {
    .\wsl_client.ps1 -Token $token -Command "tar czf /tmp/backup.tar.gz $file"
    # Transfer to Windows
    .\wsl_client.ps1 -Token $token -Command "base64 /tmp/backup.tar.gz"
}
```

### Integration with Pandora AIOS

Run Pandora diagnostics from Windows:
```powershell
.\wsl_client.ps1 -Token $token -Command "python3 diagnostic_system.py --full"
```

Start Pandora services:
```powershell
.\wsl_client.ps1 -Token $token -Command "bash launch.sh"
```

### Custom Clients

Create your own clients in any language that supports sockets:

```python
import socket
import json

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 9000))

# Send command
message = json.dumps({"type": "execute", "command": "ls"})
length = len(message).to_bytes(4, 'big')
client.send(length + message.encode())

# Receive response
length = int.from_bytes(client.recv(4), 'big')
response = json.loads(client.recv(length).decode())
print(response)
```

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │
│  Windows Client │◄───────►│   WSL Server    │
│  (PowerShell)   │  TCP    │   (Python)      │
│                 │  9000   │                 │
└─────────────────┘         └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │                 │
                            │  Pandora AIOS   │
                            │  (Linux/WSL)    │
                            │                 │
                            └─────────────────┘
```

## API Reference

### Server API

```python
from wsl_access_terminal import WSLAccessTerminal

terminal = WSLAccessTerminal(host="0.0.0.0", port=9000)
terminal.start_server()
terminal.interactive_mode()
```

### Client API

```python
from wsl_access_terminal import WSLClient

client = WSLClient(host="localhost", port=9000)
client.connect()
client.authenticate("token")
result = client.execute("ls -la")
print(result)
client.close()
```

## Contributing

Contributions welcome! Areas for improvement:
- Enhanced security features
- GUI client for Windows
- File browser interface
- Performance optimizations
- Additional command types
- Better error handling

## License

Part of Pandora AIOS project - see main LICENSE file

## Support

For issues, questions, or contributions:
- GitHub Issues: [Create an issue]
- Documentation: See main Pandora AIOS docs
- Diagnostics: Run with `--verbose` flag for detailed logs
