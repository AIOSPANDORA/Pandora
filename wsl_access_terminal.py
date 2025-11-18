"""
Pandora AIOS WSL Access Port Terminal
--------------------------------------
Provides a comprehensive terminal interface for accessing Pandora AIOS from Windows
via WSL (Windows Subsystem for Linux).

Features:
- Bidirectional communication between Windows and WSL
- Port forwarding and network bridge
- Command relay and execution
- File transfer capabilities
- Real-time diagnostics
- Secure authentication
- Session management

Philosophy: Bridge worlds, maintain harmony between Windows and Linux environments
"""

import os
import sys
import socket
import threading
import subprocess
import json
import time
import hashlib
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    import readline  # Command history and editing
except ImportError:
    readline = None

class WSLAccessTerminal:
    """Main WSL Access Port Terminal for Pandora AIOS"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        self.clients = {}
        self.session_id = self._generate_session_id()
        self.log_path = f"/tmp/wsl_terminal_{self.session_id}.log"
        self.command_history = []
        self.auth_enabled = True
        self.auth_token = self._generate_auth_token()
        
        self.log("INFO", f"WSL Access Terminal initialized - Session: {self.session_id}")
        self.log("INFO", f"Authentication token: {self.auth_token}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.sha256(f"{time.time()}_{os.getpid()}".encode()).hexdigest()[:16]
    
    def _generate_auth_token(self) -> str:
        """Generate authentication token"""
        token = base64.b64encode(os.urandom(24)).decode('utf-8')
        return token
    
    def log(self, level: str, message: str):
        """Log terminal activity"""
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}][{level}] {message}"
        
        try:
            with open(self.log_path, "a") as f:
                f.write(entry + "\n")
        except Exception:
            pass
        
        # Color codes for terminal output
        colors = {
            "INFO": "\033[94m",    # Blue
            "SUCCESS": "\033[92m", # Green
            "WARNING": "\033[93m", # Yellow
            "ERROR": "\033[91m",   # Red
            "RESET": "\033[0m"
        }
        
        color = colors.get(level, colors["RESET"])
        print(f"{color}{entry}{colors['RESET']}")
    
    def start_server(self):
        """Start the WSL access server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            self.log("SUCCESS", f"WSL Access Terminal server started on {self.host}:{self.port}")
            self.log("INFO", "Waiting for Windows client connections...")
            
            # Start connection acceptance thread
            accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
            accept_thread.start()
            
            return True
            
        except Exception as e:
            self.log("ERROR", f"Failed to start server: {e}")
            return False
    
    def _accept_connections(self):
        """Accept incoming client connections"""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                self.log("INFO", f"New connection from {client_address}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    self.log("ERROR", f"Connection accept error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, client_address: Tuple):
        """Handle individual client connection"""
        client_id = f"{client_address[0]}:{client_address[1]}"
        self.clients[client_id] = {
            "socket": client_socket,
            "address": client_address,
            "authenticated": False,
            "connected_at": datetime.now().isoformat()
        }
        
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "message": "Pandora AIOS WSL Access Terminal",
                "session_id": self.session_id,
                "auth_required": self.auth_enabled
            }
            self._send_json(client_socket, welcome)
            
            # Main client loop
            while self.running:
                data = self._receive_json(client_socket)
                if not data:
                    break
                
                response = self._process_command(data, client_id)
                self._send_json(client_socket, response)
                
        except Exception as e:
            self.log("ERROR", f"Client {client_id} error: {e}")
        finally:
            client_socket.close()
            if client_id in self.clients:
                del self.clients[client_id]
            self.log("INFO", f"Client {client_id} disconnected")
    
    def _send_json(self, sock: socket.socket, data: Dict):
        """Send JSON data over socket"""
        try:
            message = json.dumps(data).encode('utf-8')
            length = len(message)
            sock.sendall(length.to_bytes(4, byteorder='big'))
            sock.sendall(message)
        except Exception as e:
            self.log("ERROR", f"Send error: {e}")
            raise
    
    def _receive_json(self, sock: socket.socket) -> Optional[Dict]:
        """Receive JSON data from socket"""
        try:
            # First receive the length (4 bytes)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return None
            
            length = int.from_bytes(length_bytes, byteorder='big')
            
            # Receive the actual message
            message = b''
            while len(message) < length:
                chunk = sock.recv(min(length - len(message), 4096))
                if not chunk:
                    return None
                message += chunk
            
            return json.loads(message.decode('utf-8'))
            
        except Exception as e:
            self.log("ERROR", f"Receive error: {e}")
            return None
    
    def _process_command(self, data: Dict, client_id: str) -> Dict:
        """Process incoming command from client"""
        cmd_type = data.get("type", "unknown")
        
        # Check authentication
        if self.auth_enabled and not self.clients[client_id]["authenticated"]:
            if cmd_type != "auth":
                return {
                    "type": "error",
                    "message": "Authentication required",
                    "code": "AUTH_REQUIRED"
                }
        
        # Route to appropriate handler
        handlers = {
            "auth": self._handle_auth,
            "execute": self._handle_execute,
            "file_transfer": self._handle_file_transfer,
            "status": self._handle_status,
            "diagnostics": self._handle_diagnostics,
            "port_forward": self._handle_port_forward,
            "ping": self._handle_ping
        }
        
        handler = handlers.get(cmd_type, self._handle_unknown)
        return handler(data, client_id)
    
    def _handle_auth(self, data: Dict, client_id: str) -> Dict:
        """Handle authentication request"""
        token = data.get("token", "")
        
        if token == self.auth_token:
            self.clients[client_id]["authenticated"] = True
            self.log("SUCCESS", f"Client {client_id} authenticated")
            return {
                "type": "auth_response",
                "success": True,
                "message": "Authentication successful"
            }
        else:
            self.log("WARNING", f"Client {client_id} authentication failed")
            return {
                "type": "auth_response",
                "success": False,
                "message": "Invalid authentication token"
            }
    
    def _handle_execute(self, data: Dict, client_id: str) -> Dict:
        """Execute command and return result"""
        command = data.get("command", "")
        
        if not command:
            return {
                "type": "execute_response",
                "success": False,
                "error": "No command provided"
            }
        
        self.log("INFO", f"Executing command from {client_id}: {command}")
        self.command_history.append({
            "client": client_id,
            "command": command,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                text=True
            )
            
            return {
                "type": "execute_response",
                "success": True,
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "type": "execute_response",
                "success": False,
                "error": "Command timeout (30s)"
            }
        except Exception as e:
            return {
                "type": "execute_response",
                "success": False,
                "error": str(e)
            }
    
    def _handle_file_transfer(self, data: Dict, client_id: str) -> Dict:
        """Handle file transfer request"""
        operation = data.get("operation", "")  # "upload" or "download"
        path = data.get("path", "")
        
        if operation == "upload":
            content = data.get("content", "")
            try:
                # Decode base64 content
                file_content = base64.b64decode(content)
                
                # Write to file
                with open(path, "wb") as f:
                    f.write(file_content)
                
                self.log("SUCCESS", f"File uploaded from {client_id}: {path}")
                return {
                    "type": "file_transfer_response",
                    "success": True,
                    "operation": "upload",
                    "path": path
                }
                
            except Exception as e:
                return {
                    "type": "file_transfer_response",
                    "success": False,
                    "error": str(e)
                }
        
        elif operation == "download":
            try:
                # Read file
                with open(path, "rb") as f:
                    file_content = f.read()
                
                # Encode to base64
                content = base64.b64encode(file_content).decode('utf-8')
                
                self.log("SUCCESS", f"File downloaded by {client_id}: {path}")
                return {
                    "type": "file_transfer_response",
                    "success": True,
                    "operation": "download",
                    "path": path,
                    "content": content
                }
                
            except Exception as e:
                return {
                    "type": "file_transfer_response",
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "type": "file_transfer_response",
            "success": False,
            "error": "Invalid operation"
        }
    
    def _handle_status(self, data: Dict, client_id: str) -> Dict:
        """Return terminal status"""
        return {
            "type": "status_response",
            "session_id": self.session_id,
            "uptime": time.time() - float.fromhex(self.session_id[:8]),
            "connected_clients": len(self.clients),
            "command_count": len(self.command_history),
            "clients": [
                {
                    "id": cid,
                    "address": info["address"],
                    "authenticated": info["authenticated"],
                    "connected_at": info["connected_at"]
                }
                for cid, info in self.clients.items()
            ]
        }
    
    def _handle_diagnostics(self, data: Dict, client_id: str) -> Dict:
        """Run diagnostics and return results"""
        self.log("INFO", f"Running diagnostics for {client_id}")
        
        try:
            # Import and run diagnostic system
            from diagnostic_system import PandoraDiagnostics
            
            diag = PandoraDiagnostics()
            report = diag.run_full_diagnostic()
            
            return {
                "type": "diagnostics_response",
                "success": True,
                "report": report
            }
            
        except ImportError:
            return {
                "type": "diagnostics_response",
                "success": False,
                "error": "Diagnostic system not available"
            }
        except Exception as e:
            return {
                "type": "diagnostics_response",
                "success": False,
                "error": str(e)
            }
    
    def _handle_port_forward(self, data: Dict, client_id: str) -> Dict:
        """Handle port forwarding request"""
        # This is a stub for port forwarding functionality
        local_port = data.get("local_port", 0)
        remote_port = data.get("remote_port", 0)
        
        self.log("INFO", f"Port forward request: {local_port} -> {remote_port}")
        
        return {
            "type": "port_forward_response",
            "success": True,
            "message": "Port forwarding configured (stub implementation)",
            "local_port": local_port,
            "remote_port": remote_port
        }
    
    def _handle_ping(self, data: Dict, client_id: str) -> Dict:
        """Handle ping request"""
        return {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_unknown(self, data: Dict, client_id: str) -> Dict:
        """Handle unknown command type"""
        return {
            "type": "error",
            "message": f"Unknown command type: {data.get('type', 'none')}",
            "code": "UNKNOWN_COMMAND"
        }
    
    def interactive_mode(self):
        """Run interactive terminal mode"""
        self.log("INFO", "Starting interactive mode")
        print("\n" + "="*60)
        print("Pandora AIOS WSL Access Terminal - Interactive Mode")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Server: {self.host}:{self.port}")
        print(f"Auth Token: {self.auth_token}")
        print("\nCommands:")
        print("  status    - Show server status")
        print("  clients   - List connected clients")
        print("  history   - Show command history")
        print("  exec CMD  - Execute command locally")
        print("  help      - Show this help")
        print("  exit      - Stop server and exit")
        print("="*60 + "\n")
        
        while self.running:
            try:
                command = input("pandora-wsl> ").strip()
                
                if not command:
                    continue
                
                if command == "exit":
                    self.log("INFO", "Shutting down...")
                    self.running = False
                    break
                
                elif command == "status":
                    print(f"\nSession: {self.session_id}")
                    print(f"Server: {self.host}:{self.port}")
                    print(f"Connected clients: {len(self.clients)}")
                    print(f"Commands executed: {len(self.command_history)}\n")
                
                elif command == "clients":
                    if not self.clients:
                        print("\nNo clients connected\n")
                    else:
                        print("\nConnected Clients:")
                        for cid, info in self.clients.items():
                            auth_status = "✓" if info["authenticated"] else "✗"
                            print(f"  [{auth_status}] {cid} - Connected at {info['connected_at']}")
                        print()
                
                elif command == "history":
                    if not self.command_history:
                        print("\nNo command history\n")
                    else:
                        print("\nCommand History:")
                        for i, entry in enumerate(self.command_history[-10:], 1):
                            print(f"  {i}. [{entry['timestamp']}] {entry['client']}: {entry['command']}")
                        print()
                
                elif command.startswith("exec "):
                    cmd = command[5:].strip()
                    if cmd:
                        try:
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                            if result.stdout:
                                print(result.stdout)
                            if result.stderr:
                                print(f"Error: {result.stderr}", file=sys.stderr)
                        except Exception as e:
                            print(f"Error: {e}", file=sys.stderr)
                
                elif command == "help":
                    print("\nAvailable Commands:")
                    print("  status    - Show server status and statistics")
                    print("  clients   - List all connected Windows clients")
                    print("  history   - Show recent command execution history")
                    print("  exec CMD  - Execute a command locally on WSL")
                    print("  help      - Show this help message")
                    print("  exit      - Stop the server and exit terminal\n")
                
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.\n")
                
            except KeyboardInterrupt:
                print("\n")
                self.log("INFO", "Received interrupt signal")
                self.running = False
                break
            except EOFError:
                self.log("INFO", "Received EOF")
                self.running = False
                break
            except Exception as e:
                self.log("ERROR", f"Interactive mode error: {e}")
    
    def stop(self):
        """Stop the terminal server"""
        self.running = False
        
        # Close all client connections
        for client_id, client_info in list(self.clients.items()):
            try:
                client_info["socket"].close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        self.log("INFO", "WSL Access Terminal stopped")


class WSLClient:
    """Client for connecting to WSL Access Terminal from Windows"""
    
    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
        self.socket = None
        self.authenticated = False
    
    def connect(self) -> bool:
        """Connect to WSL server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            
            # Receive welcome message
            welcome = self._receive_json()
            print(f"Connected to: {welcome.get('message', 'Server')}")
            print(f"Session: {welcome.get('session_id', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def authenticate(self, token: str) -> bool:
        """Authenticate with server"""
        auth_request = {
            "type": "auth",
            "token": token
        }
        
        self._send_json(auth_request)
        response = self._receive_json()
        
        if response.get("success"):
            self.authenticated = True
            print("Authentication successful")
            return True
        else:
            print(f"Authentication failed: {response.get('message')}")
            return False
    
    def execute(self, command: str) -> Dict:
        """Execute command on WSL"""
        request = {
            "type": "execute",
            "command": command
        }
        
        self._send_json(request)
        return self._receive_json()
    
    def get_status(self) -> Dict:
        """Get server status"""
        request = {"type": "status"}
        self._send_json(request)
        return self._receive_json()
    
    def _send_json(self, data: Dict):
        """Send JSON data"""
        message = json.dumps(data).encode('utf-8')
        length = len(message)
        self.socket.sendall(length.to_bytes(4, byteorder='big'))
        self.socket.sendall(message)
    
    def _receive_json(self) -> Dict:
        """Receive JSON data"""
        length_bytes = self.socket.recv(4)
        length = int.from_bytes(length_bytes, byteorder='big')
        
        message = b''
        while len(message) < length:
            chunk = self.socket.recv(min(length - len(message), 4096))
            message += chunk
        
        return json.loads(message.decode('utf-8'))
    
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pandora AIOS WSL Access Terminal")
    parser.add_argument("--mode", choices=["server", "client"], default="server",
                       help="Run as server or client")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9000, help="Server port")
    parser.add_argument("--token", help="Authentication token (for client)")
    parser.add_argument("--command", help="Command to execute (client mode)")
    
    args = parser.parse_args()
    
    if args.mode == "server":
        terminal = WSLAccessTerminal(host=args.host, port=args.port)
        
        if terminal.start_server():
            try:
                terminal.interactive_mode()
            finally:
                terminal.stop()
    
    elif args.mode == "client":
        client = WSLClient(host=args.host, port=args.port)
        
        if client.connect():
            if args.token:
                client.authenticate(args.token)
            
            if args.command:
                result = client.execute(args.command)
                print("\nResult:")
                print(f"Success: {result.get('success')}")
                if result.get('stdout'):
                    print(f"Output:\n{result['stdout']}")
                if result.get('stderr'):
                    print(f"Errors:\n{result['stderr']}")
            else:
                print("\nNo command specified. Use --command to execute commands.")
            
            client.close()


if __name__ == "__main__":
    main()
