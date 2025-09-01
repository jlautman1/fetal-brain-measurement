#!/usr/bin/env python3
"""
Simple test script to verify OpenRecon server connectivity
Tests basic socket connection without requiring ismrmrd libraries
"""

import socket
import time
import sys

def test_server_connection(host='localhost', port=9002, timeout=5):
    """Test basic TCP connection to OpenRecon server"""
    print(f"🔌 Testing connection to OpenRecon server at {host}:{port}...")
    
    try:
        # Create a socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Try to connect
        result = sock.connect_ex((host, port))
        
        if result == 0:
            print(f"✅ SUCCESS: Server is listening on {host}:{port}")
            print(f"🎉 OpenRecon dummy server is ready for connections!")
            
            # Try to send a simple test
            try:
                sock.send(b"TEST")
                print(f"📡 Test data sent successfully")
            except Exception as e:
                print(f"⚠️  Note: Could not send test data (expected): {e}")
            
            sock.close()
            return True
        else:
            print(f"❌ FAILED: Could not connect to {host}:{port}")
            print(f"   Error code: {result}")
            return False
            
    except socket.timeout:
        print(f"❌ TIMEOUT: Server did not respond within {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass

def main():
    print("🧪 OpenRecon Server Connection Test")
    print("=" * 50)
    
    # Test connection
    success = test_server_connection()
    
    if success:
        print("\n🏆 RESULT: Dummy OpenRecon server is working correctly!")
        print("📋 Next steps:")
        print("   1. ✅ Server connectivity: PASSED")
        print("   2. 🧠 Ready for real fetal brain implementation")
        print("   3. 📦 Ready for OpenRecon package creation")
        sys.exit(0)
    else:
        print("\n❌ RESULT: Server connection failed")
        print("🔧 Troubleshooting:")
        print("   1. Check if Docker container is running: docker ps")
        print("   2. Check container logs: docker logs test-dummy")
        print("   3. Verify port mapping: docker port test-dummy")
        sys.exit(1)

if __name__ == "__main__":
    main()
