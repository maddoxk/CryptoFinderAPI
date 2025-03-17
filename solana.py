from datetime import datetime, timezone, timedelta
import logging
import time
import requests
import json
import base64
from typing import List, Dict, Set, Any, Optional
from chain_scanner_template import BlockchainScanner
from profit_calculator import ProfitCalculator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("solana_scanner")

# Solana RPC endpoints - public nodes
SOLANA_MAINNET_ENDPOINTS = [
    "https://mainnet.helius-rpc.com/?api-key=b8df5aa2-2ba9-4438-bc97-99850e9dceda",
    "https://api.mainnet-beta.solana.com",
    "https://solana-api.projectserum.com",
    "https://rpc.ankr.com/solana"

]

class SolanaScanner(BlockchainScanner):
    """
    Scanner for the Solana blockchain to identify profitable wallets.
    Designed to be easily extendable for future features.
    """
    
    def __init__(self, node_url=None):
        """
        Initialize the Solana scanner with a node URL.
        
        Args:
            node_url: Optional Solana RPC URL, defaults to public endpoints
        """
        # Try provided URL first, then fall back to public endpoints
        super().__init__(node_url or SOLANA_MAINNET_ENDPOINTS[0])
        
        # Connection testing and fallback logic
        self.connected = False
        self.selected_endpoint = None
        self._connect_with_fallback()
        
        # Constants for Solana
        self.LAMPORTS_PER_SOL = 1_000_000_000  # 1 SOL = 10^9 lamports
        
        # Cache to improve performance 
        self.slot_to_time_cache = {}
        self.signature_cache = {}
        
    def _connect_with_fallback(self):
        """Test connection to RPC nodes and select working one."""
        endpoints = [self.node_url] + [ep for ep in SOLANA_MAINNET_ENDPOINTS if ep != self.node_url]
        
        for endpoint in endpoints:
            try:
                logger.info(f"Testing connection to Solana RPC endpoint: {endpoint}")
                response = self._make_rpc_call(endpoint, "getHealth", [])
                if response.get("result") == "ok":
                    self.connected = True
                    self.selected_endpoint = endpoint
                    slot = self._make_rpc_call(endpoint, "getSlot", []).get("result", 0)
                    logger.info(f"Connected to Solana node at {endpoint}, current slot: {slot}")
                    return True
            except Exception as e:
                logger.warning(f"Failed to connect to {endpoint}: {str(e)}")
                continue
                
        logger.error("Failed to connect to any Solana RPC endpoint")
        raise ConnectionError("Cannot connect to Solana network")
    
    def _make_rpc_call(self, endpoint: str, method: str, params: list) -> dict:
        """
        Make an RPC call to the Solana node.
        
        Args:
            endpoint: RPC endpoint URL
            method: RPC method name
            params: Parameters for the RPC method
            
        Returns:
            Response from the RPC call
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "error" in result:
            raise ValueError(f"RPC error: {result['error']['message']}")
            
        return result
    
    def _rpc_call(self, method: str, params: list) -> Any:
        """
        Make an RPC call to the currently selected Solana endpoint.
        
        Args:
            method: RPC method name
            params: Parameters for the method
            
        Returns:
            Result from the RPC call
        """
        result = self._make_rpc_call(self.selected_endpoint, method, params)
        return result.get("result")
    
    def get_block_by_timestamp(self, target_timestamp: int) -> int:
        """
        Find the slot number closest to the target timestamp.
        
        On Solana, we work with slots instead of block numbers.
        """
        logger.info(f"Finding Solana slot for timestamp {target_timestamp}")
        
        # Get the current slot
        current_slot = self._rpc_call("getSlot", [])
        
        # Get the current timestamp
        current_time = int(time.time())
        
        # Estimate slot based on average 400ms slot time
        slots_per_second = 2.5  # Solana aims for 400ms per slot = 2.5 slots/second
        seconds_diff = current_time - target_timestamp
        estimated_slot_diff = int(seconds_diff * slots_per_second)
        estimated_slot = max(1, current_slot - estimated_slot_diff)
        
        logger.info(f"Estimated slot {estimated_slot} for timestamp {target_timestamp}")
        
        # Refine our estimate with binary search (uncomment if greater precision is needed)
        # In practice, the estimate is usually good enough given Solana's high TPS
        # This code is included for future extension if needed
        """
        low, high = 1, current_slot
        while low < high:
            mid = (low + high) // 2
            try:
                block_time = self._get_block_time(mid)
                if block_time < target_timestamp:
                    low = mid + 1
                else:
                    high = mid
            except Exception as e:
                # If we can't get the block time, reduce range and retry
                logger.warning(f"Error getting block time for slot {mid}: {e}")
                high = mid
        
        logger.info(f"Found slot {low} for timestamp {target_timestamp}")
        return low
        """
        
        return estimated_slot
    
    def _get_block_time(self, slot: int) -> int:
        """Get the timestamp for a given slot, with caching."""
        if slot in self.slot_to_time_cache:
            return self.slot_to_time_cache[slot]
            
        block_time = self._rpc_call("getBlockTime", [slot])
        self.slot_to_time_cache[slot] = block_time
        return block_time
    
    def get_sampled_blocks(self, start_block: int, end_block: int, num_samples: int = 100) -> List[int]:
        """
        Get a list of slot numbers sampled evenly over the range.
        
        On Solana, these are slots rather than block numbers.
        """
        if num_samples == 0:
            # Return all slots in range - warning: could be MANY on Solana
            logger.info(f"Returning all {end_block - start_block + 1} slots in range")
            return list(range(start_block, end_block + 1))
            
        range_size = end_block - start_block
        step = max(1, range_size // (num_samples - 1))
        slots = []
        for i in range(num_samples):
            slot_number = start_block + i * step
            if slot_number > end_block:
                break
            slots.append(slot_number)
            
        return slots
    
    def get_active_addresses_from_blocks(self, block_numbers: List[int]) -> Set[str]:
        """
        Collect unique addresses from transactions in the given slots.
        
        On Solana, we need to extract addresses from the transaction accounts.
        """
        addresses = set()
        successful_blocks = 0
        total_blocks = len(block_numbers)
        start_time = time.time()
        last_log_time = start_time
        
        logger.info(f"Starting to process {total_blocks} Solana slots...")
        
        for i, slot in enumerate(block_numbers):
            current_time = time.time()
            # Log progress periodically
            if i % 10 == 0 or current_time - last_log_time > 30:
                elapsed = current_time - start_time
                progress = (i / total_blocks) * 100 if total_blocks else 0
                addresses_found = len(addresses)
                logger.info(f"Progress: {i}/{total_blocks} slots ({progress:.2f}%) | Found {addresses_found} addresses | Elapsed: {format_time(elapsed)}")
                last_log_time = current_time
                
            try:
                # Get block with transactions and metadata
                block = self._rpc_call("getBlock", [slot, {"encoding": "json", "transactionDetails": "full", "maxSupportedTransactionVersion": 0}])
                
                if not block or "transactions" not in block:
                    continue
                    
                tx_count = len(block["transactions"])
                
                for tx in block["transactions"]:
                    # Extract addresses from transaction
                    if "transaction" not in tx or "message" not in tx["transaction"]:
                        continue
                        
                    # Account for the various ways addresses can appear in transactions
                    message = tx["transaction"]["message"]
                    if "accountKeys" in message:
                        for account_key in message["accountKeys"]:
                            try:
                                if isinstance(account_key, str):
                                    addresses.add(account_key)
                                elif isinstance(account_key, dict) and "pubkey" in account_key:
                                    addresses.add(account_key["pubkey"])
                            except Exception:
                                continue
                
                successful_blocks += 1
                
                # Add small delay to avoid rate limiting
                if i % 5 == 0:
                    time.sleep(0.2)
                    
            except Exception as e:
                logger.warning(f"Error processing slot {slot}: {str(e)}")
                continue
                
        total_time = time.time() - start_time
        logger.info(f"Slot processing completed: {successful_blocks}/{total_blocks} slots in {format_time(total_time)}")
        logger.info(f"Collected {len(addresses)} unique addresses")
            
        return addresses
    
    def is_valid_wallet(self, address: str) -> bool:
        """
        Check if an address is a valid wallet (vs program account).
        
        In Solana, we identify "normal" wallets vs program accounts.
        """
        try:
            # Check if this is a program account
            program_info = self._rpc_call("getAccountInfo", [address, {"encoding": "base64"}])
            
            # If account doesn't exist or has no data, treat as potential wallet
            if not program_info or "data" not in program_info:
                return True
                
            # Check executable flag - executable accounts are programs
            if program_info.get("executable", False):
                return False
                
            # Check if it has a SOL balance (account data is empty)
            if program_info.get("data") == ["", "base64"]:
                return True
                
            # Otherwise it's likely a PDA or data account
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if {address} is valid wallet: {str(e)}")
            return False
    
    def get_balance(self, address: str, block: int) -> float:
        """
        Get the SOL balance of an address at a specific slot.
        
        Args:
            address: Solana account address
            block: Slot number (not block height)
            
        Returns:
            Balance in SOL
        """
        # Check cache first for performance
        cache_key = f"{address}_{block}"
        cached_balance = ProfitCalculator.get_cached_balance(address, block)
        if cached_balance is not None:
            return cached_balance
            
        try:
            # For Solana, we need commitment level and slot context
            account_info = self._rpc_call("getAccountInfo", [
                address, 
                {"commitment": "confirmed", "encoding": "jsonParsed", "dataSlice": {"offset": 0, "length": 0}}
            ])
            
            if not account_info:
                return 0
                
            # Extract lamports and convert to SOL
            lamports = account_info.get("lamports", 0)
            sol_balance = lamports / self.LAMPORTS_PER_SOL
            
            # Cache the result
            ProfitCalculator.cache_balance(address, block, sol_balance)
            return sol_balance
            
        except Exception as e:
            logger.debug(f"Error getting balance for {address} at slot {block}: {str(e)}")
            return 0
    
    def get_token_price_on_date(self, date: str) -> float:
        """
        Fetch SOL price in USD for a given date (dd-mm-yyyy).
        
        Args:
            date: Date string in format dd-mm-yyyy
            
        Returns:
            SOL price in USD
        """
        try:
            url = f"https://api.coingecko.com/api/v3/coins/solana/history?date={date}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data['market_data']['current_price']['usd']
        except (requests.RequestException, KeyError) as e:
            logger.error(f"Error fetching SOL price for {date}: {str(e)}")
            raise ValueError(f"Could not get SOL price for {date}")
    
    def get_transaction_history(self, address: str, start_block: int, end_block: int) -> List[Dict]:
        """
        Get transaction history for a Solana address between start and end slots.
        
        Args:
            address: Solana account address
            start_block: Starting slot number 
            end_block: Ending slot number
            
        Returns:
            List of transaction details
        """
        transactions = []
        
        try:
            # Get signatures for address
            # Note: This has limitations as it can only fetch a limited history
            signatures = self._rpc_call("getSignaturesForAddress", [
                address,
                {"limit": 50}  # Adjust limit as needed
            ])
            
            if not signatures:
                return []
                
            for sig_info in signatures:
                signature = sig_info.get("signature")
                if not signature:
                    continue
                    
                # Check if we already processed this signature
                if signature in self.signature_cache:
                    continue
                    
                # Get the transaction details
                try:
                    tx_details = self._rpc_call("getTransaction", [
                        signature,
                        {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
                    ])
                    
                    if not tx_details:
                        continue
                        
                    # Check if this transaction is in our slot range
                    tx_slot = tx_details.get("slot", 0)
                    if tx_slot < start_block or tx_slot > end_block:
                        continue
                        
                    # Extract relevant transaction info
                    pre_balances = tx_details.get("meta", {}).get("preBalances", [])
                    post_balances = tx_details.get("meta", {}).get("postBalances", [])
                    account_keys = tx_details.get("transaction", {}).get("message", {}).get("accountKeys", [])
                    
                    # Find the index of our address in the account keys
                    address_index = -1
                    for i, key in enumerate(account_keys):
                        if isinstance(key, str) and key == address:
                            address_index = i
                            break
                        elif isinstance(key, dict) and key.get("pubkey") == address:
                            address_index = i
                            break
                            
                    if address_index >= 0 and address_index < len(pre_balances) and address_index < len(post_balances):
                        pre_balance = pre_balances[address_index] / self.LAMPORTS_PER_SOL
                        post_balance = post_balances[address_index] / self.LAMPORTS_PER_SOL
                        
                        # Determine if this is incoming or outgoing
                        tx_type = "unknown"
                        if post_balance > pre_balance:
                            tx_type = "in"
                        elif pre_balance > post_balance:
                            tx_type = "out"
                            
                        # Store transaction info
                        tx_info = {
                            'signature': signature,
                            'slot': tx_slot,
                            'type': tx_type,
                            'pre_balance': pre_balance,
                            'post_balance': post_balance,
                            'change': post_balance - pre_balance,
                            'block_time': tx_details.get("blockTime", 0),
                            'success': tx_details.get("meta", {}).get("err") is None
                        }
                        
                        transactions.append(tx_info)
                        self.signature_cache[signature] = tx_info
                        
                except Exception as e:
                    logger.debug(f"Error getting transaction {signature}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error getting transaction history for {address}: {str(e)}")
            
        return transactions
    
    def scan_profitable_wallets(self, days: int, min_profit: float, num_samples: int = 100, dev_mode: bool = False) -> List[Dict]:
        """
        Scan Solana for wallets with at least min_profit% over the past days.
        
        Args:
            days: Number of days to look back
            min_profit: Minimum profit percentage threshold
            num_samples: Number of slots to sample for efficiency
            dev_mode: Enable faster development mode with limited processing
            
        Returns:
            List of dicts with addresses and their profit percentages
        """
        logger.info(f"Scanning Solana for profitable wallets over {days} days with {min_profit}% minimum profit")
        
        # Determine time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        start_timestamp = int(start_time.timestamp())
        
        # Get current slot
        end_slot = self._rpc_call("getSlot", [])
        logger.info(f"Current Solana slot: {end_slot}")
        
        # Get start slot
        try:
            start_slot = self.get_block_by_timestamp(start_timestamp)
        except Exception as e:
            logger.error(f"Error getting start slot: {str(e)}")
            # Fallback: estimate block based on 400ms slot time
            slots_per_day = 24 * 60 * 60 * 2.5  # ~216,000 slots per day
            start_slot = max(0, end_slot - int(slots_per_day * days))
            logger.info(f"Using estimated start slot: {start_slot}")
        
        logger.info(f"Scan range: Slot {start_slot} to {end_slot} (approx. {days} days)")
        
        # For Solana, we need to be very conservative with sampling due to high TPS
        solana_samples = min(num_samples, 50)
        if dev_mode:
            solana_samples = min(solana_samples, 10)
            
        sampled_slots = self.get_sampled_blocks(start_slot, end_slot, solana_samples)
        
        # Collect active addresses
        logger.info(f"Collecting addresses from {len(sampled_slots)} sampled slots")
        addresses = self.get_active_addresses_from_blocks(sampled_slots)
        logger.info(f"Found {len(addresses)} unique addresses")
        
        # Filter to find valid wallets
        valid_wallets = []
        logger.info(f"Checking {len(addresses)} addresses to filter for valid wallets...")
        wallet_check_start = time.time()
        
        # For Solana, we'll check a smaller sample due to RPC limitations
        max_addresses = 500
        if len(addresses) > max_addresses:
            logger.info(f"Sampling {max_addresses} addresses out of {len(addresses)} for wallet checks")
            address_sample = list(addresses)[:max_addresses]
        else:
            address_sample = addresses
            
        for addr in address_sample:
            try:
                if self.is_valid_wallet(addr):
                    valid_wallets.append(addr)
            except Exception as e:
                logger.debug(f"Error checking address {addr}: {str(e)}")
                continue
                
        wallet_check_time = time.time() - wallet_check_start
        logger.info(f"Wallet filtering completed in {format_time(wallet_check_time)}: found {len(valid_wallets)} valid wallets")
        
        # If in dev mode, limit the number of wallets
        if dev_mode:
            max_wallets = 5
            if len(valid_wallets) > max_wallets:
                logger.info(f"Limiting to {max_wallets} wallets for dev mode")
                valid_wallets = valid_wallets[:max_wallets]
        
        # Get historical SOL prices
        start_date = start_time.strftime("%d-%m-%Y")
        end_date = end_time.strftime("%d-%m-%Y")
        
        try:
            start_price = self.get_token_price_on_date(start_date)
            end_price = self.get_token_price_on_date(end_date)
            logger.info(f"SOL price: {start_price} USD on {start_date}, {end_price} USD on {end_date}")
        except ValueError as e:
            logger.error(f"Failed to get SOL price: {str(e)}")
            return []
            
        # Calculate profits for each wallet
        profitable_wallets = []
        realistic_wallets = []
        logger.info(f"Analyzing wallet profitability for {len(valid_wallets)} wallets...")
        
        # Process wallets in batches
        batch_size = 10
        wallet_counter = 0
        
        for batch_start in range(0, len(valid_wallets), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_wallets))
            batch = valid_wallets[batch_start:batch_end]
            
            logger.info(f"Processing wallet batch {batch_start//batch_size + 1}/{(len(valid_wallets)-1)//batch_size + 1}")
            
            for addr in batch:
                try:
                    wallet_counter += 1
                    
                    # Log progress occasionally
                    if wallet_counter % 5 == 0:
                        logger.info(f"Processed {wallet_counter}/{len(valid_wallets)} wallets, found {len(profitable_wallets)} profitable")
                    
                    # Get balances
                    start_balance = self.get_balance(addr, start_slot)
                    end_balance = self.get_balance(addr, end_slot)
                    
                    # Get transaction history
                    transactions = self.get_transaction_history(addr, start_slot, end_slot)
                    
                    # Analyze transaction history
                    tx_analysis = ProfitCalculator.analyze_transaction_history(transactions, addr)
                    
                    # Calculate profit metrics
                    profit_metrics = ProfitCalculator.calculate_profit_with_transfers(
                        start_balance,
                        end_balance,
                        start_price,
                        end_price,
                        tx_analysis.get('transfers_in', 0),
                        tx_analysis.get('transfers_out', 0)
                    )
                    
                    # Calculate confidence score
                    confidence = ProfitCalculator.get_confidence_score({**profit_metrics, **tx_analysis})
                    
                    # Use realistic profit for filtering
                    realistic_profit = profit_metrics['realistic_profit_percent']
                    
                    if realistic_profit >= min_profit:
                        wallet_data = {
                            'address': addr,
                            'profit_percent': realistic_profit,
                            'raw_profit_percent': profit_metrics['adjusted_profit_percent'],
                            'start_balance': start_balance,
                            'end_balance': end_balance,
                            'simple_profit_percent': profit_metrics['simple_profit_percent'],
                            'hodl_profit_percent': profit_metrics['hodl_profit_percent'],
                            'absolute_profit_usd': profit_metrics['absolute_profit_usd'],
                            'significance': profit_metrics.get('significance', 'unknown'),
                            'is_realistic': profit_metrics.get('is_realistic', True),
                            'transfers_in': tx_analysis.get('transfers_in', 0),
                            'transfers_out': tx_analysis.get('transfers_out', 0),
                            'transaction_count': tx_analysis.get('transaction_count', 0),
                            'significant_txns': tx_analysis.get('significant_transaction_count', 0),
                            'confidence_score': confidence,
                            'blockchain': 'solana',
                            'price_change_percent': profit_metrics['price_change_percent']
                        }
                        
                        profitable_wallets.append(wallet_data)
                        
                        # Track realistic wallets separately
                        if profit_metrics.get('is_realistic', True) and confidence > 30:
                            realistic_wallets.append(wallet_data)
                            
                except Exception as e:
                    logger.warning(f"Error calculating profit for {addr}: {str(e)}")
                    continue
                    
            # Add delay between batches to avoid rate limiting
            if batch_end < len(valid_wallets):
                time.sleep(1)
                
        # Sort wallets by confidence and profit
        profitable_wallets.sort(key=lambda x: (x['confidence_score'], x['profit_percent']), reverse=True)
        
        # If we have enough realistic wallets, prefer those
        if len(realistic_wallets) >= 3:
            logger.info(f"Using {len(realistic_wallets)} realistic profitable wallets")
            return realistic_wallets
            
        logger.info(f"Found {len(profitable_wallets)} profitable wallets with ≥{min_profit}% profit")
        return profitable_wallets
    
    def estimate_scan_time(self, days: int, num_samples: int = 100) -> Dict:
        """
        Estimate how long the Solana scan will take.
        
        Args:
            days: Number of days to look back
            num_samples: Number of slots to sample
            
        Returns:
            Dict with time estimates
        """
        # Base time for connecting
        base_time = 10
        
        # Time to find start slot
        slot_search_time = 5
        
        # Time to collect addresses from slots
        # Solana blocks are larger and have more transactions
        block_fetch_time = min(num_samples, 50) * 3
        
        # Wallet checking time (assume ~500 addresses, 0.15s each)
        address_check_time = min(500, 300) * 0.15
        
        # Price fetch time
        price_fetch_time = 4
        
        # Balance checking time
        wallet_count = min(300, 100)
        balance_check_time = wallet_count * 2 * 0.3
        
        # Total estimated time - Solana RPC can be slower and less predictable
        min_time = base_time + slot_search_time + block_fetch_time * 0.8 + address_check_time * 0.8 + price_fetch_time + balance_check_time * 0.8
        max_time = base_time * 1.5 + slot_search_time * 2 + block_fetch_time * 2.5 + address_check_time * 2 + price_fetch_time * 1.5 + balance_check_time * 2
        
        return {
            "min_seconds": min_time,
            "max_seconds": max_time,
            "min_time": format_time(min_time),
            "max_time": format_time(max_time),
            "factors_affecting_time": [
                "Solana network congestion",
                "RPC endpoint responsiveness",
                "Rate limiting on public nodes",
                "High transaction volume in Solana blocks",
                "Number of valid wallets found"
            ]
        }

def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        return f"{int(seconds/60)} minutes and {int(seconds%60)} seconds"
    else:
        return f"{int(seconds/3600)} hours, {int((seconds%3600)/60)} minutes"

# Example usage
if __name__ == "__main__":
    scanner = SolanaScanner()
    days = 7
    min_profit = 50
    samples = 50
    
    # Estimate scan time
    time_estimate = scanner.estimate_scan_time(days, samples)
    print(f"Estimated scan time: {time_estimate['min_time']} to {time_estimate['max_time']}")
    
    print("Starting scan...")
    start_time = time.time()
    wallets = scanner.scan_profitable_wallets(days, min_profit, num_samples=samples, dev_mode=False)
    elapsed = time.time() - start_time
    
    print(f"Scan completed in {format_time(elapsed)}")
    print(f"Found {len(wallets)} wallets with >= {min_profit}% profit:")
    for wallet in wallets:
        print(f"Address: {wallet['address']}, Profit: {wallet['profit_percent']:.2f}%")
