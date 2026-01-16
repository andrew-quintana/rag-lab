#!/usr/bin/env python3
"""Check DATABASE_URL connection string format"""

import sys
import urllib.parse
from pathlib import Path

# Try to get from Azure Functions or environment
if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    # Try to get from .env.prod
    env_file = Path(__file__).parent.parent / ".env.prod"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("DATABASE_URL="):
                    url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    else:
        print("❌ No DATABASE_URL provided and .env.prod not found")
        print("Usage: python scripts/check_db_connection_string.py <DATABASE_URL>")
        sys.exit(1)

print("=" * 70)
print("DATABASE_URL Connection String Analysis")
print("=" * 70)
print()

try:
    parts = urllib.parse.urlparse(url)
    
    # Extract username and password
    if '@' in parts.netloc:
        user_pass, host_port = parts.netloc.split('@', 1)
        if ':' in user_pass:
            username, password = user_pass.split(':', 1)
        else:
            username = user_pass
            password = ""
    else:
        username = ""
        password = ""
        host_port = parts.netloc
    
    # Extract host and port
    if ':' in host_port:
        hostname, port = host_port.split(':', 1)
    else:
        hostname = host_port
        port = ""
    
    print("✅ Connection String Format:")
    print(f"   Protocol: {parts.scheme}")
    print(f"   Username: {username}")
    print(f"   Password: {'*' * len(password)} ({len(password)} characters)")
    print(f"   Host: {hostname}")
    print(f"   Port: {port}")
    print(f"   Database: {parts.path.strip('/')}")
    
    # Check query parameters
    query_params = urllib.parse.parse_qs(parts.query)
    if query_params:
        print(f"   Query params: {parts.query}")
    
    print()
    print("🔍 Validation:")
    
    # Check username format
    if username.startswith("postgres."):
        project_ref = username.split(".", 1)[1]
        print(f"   ✅ Username format correct (postgres.{project_ref})")
    elif username == "postgres":
        print(f"   ⚠️  Username is 'postgres' - should be 'postgres.{{project_ref}}' for pooler")
    else:
        print(f"   ⚠️  Username format: {username}")
    
    # Check port
    if port == "6543":
        print(f"   ✅ Port correct (6543 = transaction pooler)")
    elif port == "5432":
        print(f"   ⚠️  Port 5432 (direct connection) - consider using 6543 for pooler")
    else:
        print(f"   ⚠️  Port: {port}")
    
    # Check host
    if "pooler.supabase.com" in hostname:
        print(f"   ✅ Host correct (using pooler)")
    else:
        print(f"   ⚠️  Host: {hostname}")
    
    # Check SSL
    if "sslmode=require" in parts.query or "sslmode=prefer" in parts.query:
        print(f"   ✅ SSL mode set")
    else:
        print(f"   ⚠️  SSL mode not set (should add ?sslmode=require)")
    
    # Check for special characters in password that need encoding
    special_chars = ['@', ':', '/', '%', '?', '=', '#', '[', ']']
    found_special = [c for c in special_chars if c in password]
    if found_special:
        print()
        print("⚠️  WARNING: Password contains special characters that may need URL encoding:")
        print(f"   Found: {', '.join(found_special)}")
        print()
        print("   Special characters in passwords must be URL-encoded:")
        print("   @ → %40")
        print("   : → %3A")
        print("   / → %2F")
        print("   % → %25")
        print("   ? → %3F")
        print("   = → %3D")
        print()
        print("   Example: If password is 'pass@word', use 'pass%40word'")
    else:
        print(f"   ✅ Password has no special characters requiring encoding")
    
    print()
    print("=" * 70)
    print("💡 If you're getting 'Tenant or user not found' error:")
    print("   1. Verify password is correct in Supabase Dashboard")
    print("   2. URL-encode any special characters in password")
    print("   3. Ensure username format is: postgres.{{project_ref}}")
    print("   4. Use transaction pooler (port 6543)")
    print("=" * 70)
    
except Exception as e:
    print(f"❌ Error parsing connection string: {e}")
    sys.exit(1)
