import time
import json
from web3 import Web3
from datetime import datetime, timedelta
import os
import pandas as pd
from tqdm import tqdm
import requests
from functools import lru_cache

class EthereumScanner:
    def __init__(self, rpc_url=None):
        """Initialize the Ethereum scanner with an RPC endpoint."""
        # Default to Infura if no RPC URL is provided
        self.rpc_url = rpc_url or "https://eth-mainnet.public.blastapi.io"
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")
        
        self.wallets_data = {}
        self.profitable_metrics = {}
        
        # Known exchange and service wallets (simplified initial database)
        self.known_exchanges = {
            # Major exchanges - these are examples, not comprehensive
            "binance": [
                "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
                "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF"
            ],
            "coinbase": [
                "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3",
                "0x503828976D22510aad0201ac7EC88293211D23Da"
            ],
            "kraken": [
                "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2"
            ],
            "bitfinex": [
                "0x1151314c646Ce4E0eCd76F7a2310367BB18CE5a4",
                "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
            ],
            "ftx": [
                "0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2"
            ],
            "huobi": [
                "0xaB5C66752a9e8167967685F1450532fB96d5d24f",
                "0x6748F50f686bFfB8dB7a3006b2f2b6476022d271"
            ]
        }
        
        # Initialize wallet classification cache
        self.wallet_classifications = {}
    
    def check_connection(self):
        """Verify the connection to the Ethereum node."""
        return {
            "connected": self.w3.is_connected(),
            "current_block": self.w3.eth.block_number if self.w3.is_connected() else None
        }
    
    def scan_blocks(self, blocks_to_scan=100):
        """Scan the latest blocks for transactions."""
        current_block = self.w3.eth.block_number
        start_block = current_block - blocks_to_scan
        
        print(f"Scanning blocks {start_block} to {current_block}...")
        
        for block_num in tqdm(range(start_block, current_block + 1)):
            try:
                block = self.w3.eth.get_block(block_num, full_transactions=True)
                self._process_block_transactions(block)
            except Exception as e:
                print(f"Error processing block {block_num}: {str(e)}")
                continue
    
    def _process_block_transactions(self, block):
        """Process all transactions in a block to track wallet activity."""
        for tx in block.transactions:
            # Skip contract creation transactions
            if tx.get('to') is None:
                continue
                
            from_addr = tx['from']
            to_addr = tx['to']
            value = tx['value']
            gas_price = tx['gasPrice']
            gas = tx['gas']
            tx_fee = gas_price * gas
            
            # Track sender activity
            self._update_wallet_data(from_addr, -value, tx_fee, True)
            
            # Track receiver activity
            self._update_wallet_data(to_addr, value, 0, False)
    
    def _is_contract(self, address):
        """Check if an address is a contract by looking for code at the address."""
        try:
            checksum_address = Web3.to_checksum_address(address)
            code = self.w3.eth.get_code(checksum_address)
            # If there's code at this address, it's a contract
            return len(code) > 0 and code != '0x'
        except Exception as e:
            print(f"Error checking if address {address} is a contract: {str(e)}")
            # Default to treating as EOA if we can't determine
            return False
    
    @lru_cache(maxsize=1000)
    def _classify_exchange_wallet(self, address):
        """Identify if a wallet belongs to a known exchange."""
        checksum_address = Web3.to_checksum_address(address.lower())
        
        # Direct match with known exchange addresses
        for exchange, addresses in self.known_exchanges.items():
            if checksum_address in addresses:
                return exchange
            
        return None
    
    def _classify_wallet_type(self, wallet_data):
        """
        Classify a wallet as hot, cold, or intermediate based on activity patterns.
        
        Hot wallets: High transaction frequency, recent activity
        Cold wallets: Low transaction frequency, may have large holdings
        """
        tx_count = wallet_data.get('tx_count', 0)
        last_active = wallet_data.get('last_active', datetime.now())
        days_since_active = (datetime.now() - last_active).days if last_active else 0
        
        # Basic classification heuristics
        if tx_count >= 50 and days_since_active < 7:
            return "hot"
        elif tx_count < 10 and days_since_active > 30:
            return "cold"
        else:
            return "intermediate"
    
    def _update_wallet_data(self, address, value_change, tx_fee, is_sender):
        """Update wallet tracking data."""
        # Store lowercase version for consistent dictionary keys
        address_lower = address.lower()
        
        if address_lower not in self.wallets_data:
            # Initialize wallet data structure
            try:
                # Use checksum address for Web3.py methods
                checksum_address = Web3.to_checksum_address(address_lower)
                balance = self.w3.eth.get_balance(checksum_address)
                
                # Check if this is a contract address
                is_contract = self._is_contract(checksum_address)
                
                # Check if this is a known exchange wallet
                exchange = self._classify_exchange_wallet(checksum_address)
                
                self.wallets_data[address_lower] = {
                    'balance': balance,
                    'tx_count': 0,
                    'total_sent': 0,
                    'total_received': 0,
                    'total_fees': 0,
                    'last_active': datetime.now(),
                    'checksum_address': checksum_address,  # Store checksum version for future use
                    'is_contract': is_contract,  # Store whether this is a contract
                    'wallet_type': 'contract' if is_contract else 'eoa',  # EOA = Externally Owned Account
                    'exchange': exchange,
                }
            except Exception as e:
                print(f"Error processing address {address}: {str(e)}")
                return
        
        wallet = self.wallets_data[address_lower]
        wallet['tx_count'] += 1
        wallet['last_active'] = datetime.now()
        
        if is_sender:
            wallet['total_sent'] += value_change
            wallet['total_fees'] += tx_fee
        else:
            wallet['total_received'] += value_change
    
    def analyze_wallets(self):
        """Calculate profitability metrics for scanned wallets."""
        for address_lower, data in self.wallets_data.items():
            try:
                # Use the stored checksum address or convert again if needed
                checksum_address = data.get('checksum_address') or Web3.to_checksum_address(address_lower)
                
                # Get current balance using checksum address
                current_balance = self.w3.eth.get_balance(checksum_address)
                
                # Safely handle potentially large or negative values
                try:
                    # For total_volume, ensure we're not exceeding uint256 max value
                    # We use max(0, value) to ensure we don't have negative values
                    total_sent_abs = abs(data['total_sent']) if data['total_sent'] < 0 else data['total_sent']
                    total_volume = min(total_sent_abs + max(0, data['total_received']), 2**256 - 1)
                    
                    # For net_flow, clamp within safe range
                    net_flow = max(-(2**255), min(data['total_received'] + data['total_sent'], 2**255 - 1))
                    
                    # Determine if wallet is hot or cold based on transaction patterns
                    wallet_category = self._classify_wallet_type(data)
                    
                    # Calculate metrics with value validation
                    metrics = {
                        'address': checksum_address,  # Use checksum address in results
                        'current_balance': current_balance,
                        'balance_eth': float(self.w3.from_wei(current_balance, 'ether')),
                        'tx_count': data['tx_count'],
                        'total_volume': total_volume,
                        'volume_eth': float(self.w3.from_wei(total_volume, 'ether')),
                        'net_flow': net_flow,
                        'net_flow_eth': float(self.w3.from_wei(abs(net_flow), 'ether')) * (-1 if net_flow < 0 else 1),
                        'total_fees_eth': float(self.w3.from_wei(min(data['total_fees'], 2**256 - 1), 'ether')),
                        'last_active': data['last_active'],
                        'is_contract': data['is_contract'],
                        'wallet_type': data['wallet_type'],
                        'exchange': data.get('exchange', None),
                        'wallet_category': wallet_category,
                        'days_since_active': (datetime.now() - data['last_active']).days
                    }
                    
                    # Simple profitability score calculation
                    recency_factor = 1.0
                    if (datetime.now() - data['last_active']) > timedelta(days=30):
                        recency_factor = 0.5
                        
                    # Use validated values for score calculation
                    balance_eth_score = float(metrics['balance_eth']) 
                    volume_eth_score = float(metrics['volume_eth'])
                    net_flow_eth_score = float(max(0, metrics['net_flow_eth'])) 
                    tx_count_score = float(metrics['tx_count'])
                    
                    # Apply maximum caps to prevent extreme values from skewing scores
                    balance_eth_score = min(balance_eth_score, 10000)
                    volume_eth_score = min(volume_eth_score, 10000)
                    net_flow_eth_score = min(net_flow_eth_score, 10000)
                    tx_count_score = min(tx_count_score, 1000)
                    
                    metrics['profitability_score'] = (
                        balance_eth_score * 0.4 +
                        volume_eth_score * 0.3 +
                        net_flow_eth_score * 0.2 +
                        tx_count_score * 0.1
                    ) * recency_factor
                    
                    self.profitable_metrics[address_lower] = metrics
                    
                except (OverflowError, ValueError) as e:
                    print(f"Math error calculating metrics for {address_lower}: {str(e)}")
                    continue
                
            except Exception as e:
                print(f"Error analyzing wallet {address_lower}: {str(e)}")
                continue
    
    def get_known_exchanges(self):
        """Return list of known exchange names in the database."""
        return list(self.known_exchanges.keys())
    
    def load_exchange_database(self, file_path=None):
        """Load exchange database from a JSON file."""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    self.known_exchanges = json.load(f)
                print(f"Loaded {len(self.known_exchanges)} exchanges from database.")
            except Exception as e:
                print(f"Error loading exchange database: {str(e)}")
    
    def save_exchange_database(self, file_path="exchange_addresses.json"):
        """Save the current exchange database to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.known_exchanges, f, indent=2)
            print(f"Exchange database saved to {file_path}")
        except Exception as e:
            print(f"Error saving exchange database: {str(e)}")
    
    def get_profitable_wallets(self, limit=20, min_balance_eth=1.0, wallet_type='eoa', exchange_filter=None, 
                              wallet_category=None):
        """Return the most profitable wallets based on calculated metrics.
        
        Args:
            limit: Maximum number of wallets to return
            min_balance_eth: Minimum balance in ETH to consider wallet profitable
            wallet_type: Type of wallet to filter by ('eoa', 'contract', or 'all')
            exchange_filter: Filter by exchange name or None for all
            wallet_category: Filter by wallet category ('hot', 'cold', 'intermediate', or None for all)
        """
        # Filter by criteria
        filtered_wallets = {}
        for addr, data in self.profitable_metrics.items():
            # Skip if below minimum balance
            if data['balance_eth'] < min_balance_eth:
                continue
                
            # Skip if wallet type doesn't match
            if wallet_type != 'all' and data['wallet_type'] != wallet_type:
                continue
                
            # Skip if exchange filter is specified and doesn't match
            if exchange_filter and data.get('exchange') != exchange_filter:
                continue
                
            # Skip if wallet category filter is specified and doesn't match
            if wallet_category and data.get('wallet_category') != wallet_category:
                continue
                
            filtered_wallets[addr] = data
        
        # Sort by profitability score
        sorted_wallets = sorted(
            filtered_wallets.values(), 
            key=lambda x: x['profitability_score'], 
            reverse=True
        )
        
        return sorted_wallets[:limit]
    
    def export_results(self, filepath=None, wallet_type='eoa'):
        """Export results to CSV file."""
        if not self.profitable_metrics:
            print("No data to export. Run scan_blocks() and analyze_wallets() first.")
            return
            
        if filepath is None:
            filepath = f"profitable_eth_{wallet_type}_wallets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Filter by wallet type
        if wallet_type != 'all':
            wallets_to_export = [w for w in self.profitable_metrics.values() if w['wallet_type'] == wallet_type]
        else:
            wallets_to_export = list(self.profitable_metrics.values())
            
        df = pd.DataFrame(wallets_to_export)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")


if __name__ == "__main__":
    # Example usage
    scanner = EthereumScanner()
    print("Connection status:", scanner.check_connection())
    
    # Optional: Load exchange database from file
    # scanner.load_exchange_database("exchange_addresses.json")
    
    # Scan recent blocks
    scanner.scan_blocks(blocks_to_scan=1)  # Adjust number based on desired depth
    
    # Analyze wallet profitability
    scanner.analyze_wallets()
    
    # Get and display results - only EOAs (personal wallets)
    profitable_wallets = scanner.get_profitable_wallets(
        limit=10, 
        min_balance_eth=5.0, 
        wallet_type='eoa'
    )
    
    print("\nTop Profitable Personal Wallets (EOAs):")
    for i, wallet in enumerate(profitable_wallets, 1):
        print(f"{i}. Address: {wallet['address']}")
        print(f"   Balance: {wallet['balance_eth']:.4f} ETH")
        print(f"   Volume: {wallet['volume_eth']:.4f} ETH")
        print(f"   Net Flow: {wallet['net_flow_eth']:.4f} ETH")
        print(f"   Transactions: {wallet['tx_count']}")
        print(f"   Score: {wallet['profitability_score']:.2f}")
        print(f"   Type: {wallet['wallet_type'].upper()}")
        
        # Display additional wallet information
        if wallet.get('exchange'):
            print(f"   Exchange: {wallet['exchange'].upper()}")
        
        print(f"   Category: {wallet.get('wallet_category', 'unknown').upper()}")
        print(f"   Days since activity: {wallet.get('days_since_active', 0)}")
        print()
    
    # Export results - only EOAs
    scanner.export_results(wallet_type='eoa')
    
    # Optionally save updated exchange database
    # scanner.save_exchange_database()
