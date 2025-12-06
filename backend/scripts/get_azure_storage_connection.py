"""Helper script to get Azure Storage Account connection string

This script helps retrieve the connection string from Azure CLI
and optionally add it to .env.local
"""

import subprocess
import sys
from pathlib import Path

def get_storage_accounts():
    """List all storage accounts"""
    try:
        result = subprocess.run(
            ["az", "storage", "account", "list", "--output", "table"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Available Storage Accounts:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error listing storage accounts: {e}")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("Azure CLI not found. Please install it or set connection string manually.")
        return False

def get_connection_string(storage_account_name, resource_group=None):
    """Get connection string for a storage account"""
    try:
        cmd = ["az", "storage", "account", "show-connection-string", 
               "--name", storage_account_name, "--output", "tsv"]
        if resource_group:
            cmd.extend(["--resource-group", resource_group])
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        connection_string = result.stdout.strip()
        return connection_string
    except subprocess.CalledProcessError as e:
        print(f"Error getting connection string: {e}")
        print(e.stderr)
        return None

def main():
    print("Azure Storage Connection String Helper")
    print("=" * 50)
    print()
    
    # List storage accounts
    if not get_storage_accounts():
        return
    
    print()
    print("To get connection string, run:")
    print("  az storage account show-connection-string --name <STORAGE_ACCOUNT_NAME> --output tsv")
    print()
    print("Or if you know the storage account name and resource group:")
    print("  az storage account show-connection-string --name <STORAGE_ACCOUNT_NAME> --resource-group <RESOURCE_GROUP> --output tsv")
    print()
    
    # Try to get from command line args
    if len(sys.argv) >= 2:
        storage_account = sys.argv[1]
        resource_group = sys.argv[2] if len(sys.argv) >= 3 else None
        
        print(f"Getting connection string for: {storage_account}")
        conn_str = get_connection_string(storage_account, resource_group)
        
        if conn_str:
            print()
            print("Connection String:")
            print(conn_str)
            print()
            print("Add this to your .env.local file:")
            print(f"AZURE_BLOB_CONNECTION_STRING={conn_str}")
        else:
            print("Failed to retrieve connection string")
    else:
        print("Usage:")
        print("  python get_azure_storage_connection.py <STORAGE_ACCOUNT_NAME> [RESOURCE_GROUP]")

if __name__ == "__main__":
    main()

