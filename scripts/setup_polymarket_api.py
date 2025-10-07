#!/usr/bin/env python3
"""
Polymarket API Key Setup Script

This script creates or derives API keys for your Polymarket account.
Run this once to generate your API credentials, which will be saved to .env

Usage:
    uv run python setup_polymarket_api.py
"""

import os
from dotenv import load_dotenv, set_key
from py_clob_client.client import ClobClient

def main():
    print("=" * 60)
    print("Polymarket API Key Setup")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Get credentials from .env
    private_key = os.getenv('PRIVATE_KEY')
    funder_address = os.getenv('FUNDER_ADDRESS')
    signature_type = int(os.getenv('SIGNATURE_TYPE', 1))
    chain_id = int(os.getenv('CHAIN_ID', 137))
    host = os.getenv('CLOB_HOST', 'https://clob.polymarket.com')

    # Validate required credentials
    if not private_key or not funder_address:
        print("\n❌ Error: Missing credentials in .env file")
        print("Please ensure PRIVATE_KEY and FUNDER_ADDRESS are set in .env")
        return

    print(f"\n📋 Configuration:")
    print(f"   Host: {host}")
    print(f"   Chain ID: {chain_id}")
    print(f"   Funder Address: {funder_address}")
    print(f"   Signature Type: {signature_type} {'(Email/Magic)' if signature_type == 1 else '(EOA)' if signature_type == 0 else '(Gnosis Safe)'}")

    try:
        print("\n🔄 Initializing CLOB client...")

        # Initialize client with signature type and funder
        client = ClobClient(
            host=host,
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder_address
        )

        print("✅ CLOB client initialized successfully")

        print("\n🔑 Creating/Deriving API keys...")

        # Create or derive API key (this returns ApiKeyCreds object)
        creds = client.create_or_derive_api_creds()

        print("✅ API keys generated successfully!")

        # Display the credentials
        print("\n" + "=" * 60)
        print("Your API Credentials:")
        print("=" * 60)
        print(f"API Key:        {creds.api_key}")
        print(f"API Secret:     {creds.api_secret}")
        print(f"API Passphrase: {creds.api_passphrase}")
        print("=" * 60)

        # Save credentials to .env file
        env_file = '.env'
        print(f"\n💾 Saving credentials to {env_file}...")

        set_key(env_file, 'POLY_API_KEY', creds.api_key)
        set_key(env_file, 'POLY_API_SECRET', creds.api_secret)
        set_key(env_file, 'POLY_API_PASSPHRASE', creds.api_passphrase)

        print("✅ Credentials saved to .env file")

        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\n🎉 Your Polymarket API is now configured!")
        print("\nNext steps:")
        print("  1. Keep your .env file secure (never commit to git)")
        print("  2. Run 'uv run python polymarket_data_access.py' to test data access")
        print("  3. Use the client in your own scripts")

        print("\n⚠️  Security Warning:")
        print("   - Never share your private key or API credentials")
        print("   - The .env file is already in .gitignore")
        print("   - Store backups securely")

    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")
        print("\nTroubleshooting:")
        print("  - Verify your private key is correct")
        print("  - Ensure your funder address matches your Polymarket account")
        print("  - Check your internet connection")
        print("  - Make sure you have USDC deposited in your Polymarket account")
        raise

if __name__ == "__main__":
    main()
