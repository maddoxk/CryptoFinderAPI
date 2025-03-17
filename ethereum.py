from web3 import Web3
import requests
from datetime import datetime, timezone, timedelta
import logging
import time
import random
from profit_calculator import ProfitCalculator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ethereum_scanner")

# Connect to the public node
PUBLIC_NODE_URL = "https://eth-mainnet.public.blastapi.io"

class EthereumScanner:
    def __init__(self, node_url=PUBLIC_NODE_URL):
        self.w3 = Web3(Web3.HTTPProvider(node_url))
        if not self.w3.is_connected():
            logger.error("Failed to connect to Ethereum node")
            raise ConnectionError("Cannot connect to Ethereum node")
        logger.info(f"Connected to Ethereum node, current block: {self.w3.eth.block_number}")
        
    def get_block_by_timestamp(self, target_timestamp):
        """Find the block number closest to the target timestamp using binary search."""
        low, high = 0, self.w3.eth.block_number
        while low < high:
            mid = (low + high) // 2
            try:
                mid_block = self.w3.eth.get_block(mid)
                mid_timestamp = mid_block['timestamp']
                if mid_timestamp < target_timestamp:
                    low = mid + 1
                else:
                    high = mid
            except Exception as e:
                logger.warning(f"Error getting block {mid}: {e}")
                # On error, reduce range and try again
                high = mid
                time.sleep(0.5)  # Add delay to avoid rate limits
        return low
        
    def get_sampled_blocks(self, start_block, end_block, num_samples=100):
        """
        Get a list of block numbers sampled evenly over the range.
        If num_samples is 0, returns all blocks in the range.
        """
        if num_samples == 0:
            # Return all blocks in range if num_samples is 0
            logger.info(f"Returning all {end_block - start_block + 1} blocks in range")
            return list(range(start_block, end_block + 1))
            
        range_size = end_block - start_block
        step = max(1, range_size // (num_samples - 1))  # Ensure at least 1 block apart
        blocks = []
        for i in range(num_samples):
            block_number = start_block + i * step
            if block_number > end_block:
                break
            blocks.append(block_number)
        return blocks
        
    def get_active_addresses_from_blocks(self, block_numbers):
        """Collect unique addresses from transactions in the given blocks."""
        addresses = set()
        successful_blocks = 0
        total_blocks = len(block_numbers)
        start_time = time.time()
        last_log_time = start_time
        
        logger.info(f"Starting to process {total_blocks} blocks...")
        
        for i, block_number in enumerate(block_numbers):
            current_time = time.time()
            # Log progress every 10 blocks or 30 seconds, whichever comes first
            if i % 10 == 0 or current_time - last_log_time > 30:
                elapsed = current_time - start_time
                progress = (i / total_blocks) * 100 if total_blocks else 0
                addresses_found = len(addresses)
                logger.info(f"Progress: {i}/{total_blocks} blocks ({progress:.2f}%) | Found {addresses_found} addresses | Elapsed: {format_time(elapsed)}")
                last_log_time = current_time
                
            try:
                block_fetch_start = time.time()
                block = self.w3.eth.get_block(block_number, full_transactions=True)
                block_fetch_time = time.time() - block_fetch_start
                
                tx_count = len(block['transactions'])
                logger.debug(f"Block {block_number}: fetched in {block_fetch_time:.2f}s, contains {tx_count} transactions")
                
                for tx in block['transactions']:
                    if tx['from']:
                        addresses.add(tx['from'])
                    if tx['to']:  # Exclude contract creations where 'to' is None
                        addresses.add(tx['to'])
                successful_blocks += 1
                
                # Add small delay to avoid rate limiting
                if random.random() < 0.3:  # 30% chance to sleep
                    time.sleep(0.2)
                    
            except Exception as e:
                logger.warning(f"Error processing block {block_number}: {e}")
                continue
                
        total_time = time.time() - start_time
        logger.info(f"Block processing completed: {successful_blocks}/{total_blocks} blocks in {format_time(total_time)}")
        
        if successful_blocks > 0:
            avg_time_per_block = total_time / successful_blocks
            logger.info(f"Average time per successful block: {avg_time_per_block:.2f}s")
            logger.info(f"Collected {len(addresses)} unique addresses from {successful_blocks} blocks")
            
        # If we couldn't process enough blocks, warn but continue
        if successful_blocks < total_blocks * 0.5:  # Less than 50% success
            logger.warning(f"Limited data due to block retrieval issues (only {successful_blocks}/{total_blocks} blocks)")
            
        return addresses
        
    def is_eoa(self, address):
        """Return True if the address is an EOA (no code) with caching."""
        # Check cache first
        cached_code = ProfitCalculator.get_cached_code(address)
        if cached_code is not None:
            return cached_code == b''
            
        try:
            code = self.w3.eth.get_code(address)
            # Cache the result
            ProfitCalculator.cache_code(address, code)
            return code == b''
        except Exception as e:
            logger.debug(f"Error checking if {address} is EOA: {e}")
            # If we can't determine, assume it's not an EOA
            return False
        
    def get_balance(self, address, block):
        """Get the ETH balance of an address at a specific block with caching."""
        # Check cache first
        cached_balance = ProfitCalculator.get_cached_balance(address, block)
        if cached_balance is not None:
            return cached_balance
            
        try:
            balance = self.w3.eth.get_balance(address, block_identifier=block) / 1e18
            # Cache the result
            ProfitCalculator.cache_balance(address, block, balance)
            return balance
        except Exception as e:
            logger.debug(f"Error getting balance for {address} at block {block}: {e}")
            return 0
        
    def get_eth_price_on_date(self, date):
        """Fetch ETH price in USD for a given date (dd-mm-yyyy)."""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/ethereum/history?date={date}"
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            return data['market_data']['current_price']['usd']
        except (requests.RequestException, KeyError) as e:
            logger.error(f"Error fetching ETH price for {date}: {e}")
            raise ValueError(f"Could not get ETH price for {date}")
        
    def get_transaction_history(self, address, start_block, end_block):
        """Get transaction history using a more efficient block sampling approach."""
        transactions = []
        address_lower = address.lower()
        
        try:
            # Optimize by checking blocks in reverse chronological order
            # and only checking a limited number
            max_blocks_to_check = 20
            blocks_to_check = []
            
            # First try some recent blocks
            recent_range = min(20, end_block - start_block)
            for i in range(end_block, end_block - recent_range, -1):
                blocks_to_check.append(i)
                
            # Then try some blocks from the middle of the range
            if start_block < end_block - recent_range:
                mid_point = (start_block + end_block) // 2
                for i in range(5):  # 5 blocks around midpoint
                    offset = i - 2
                    if start_block <= mid_point + offset <= end_block:
                        blocks_to_check.append(mid_point + offset)
                        
            # Finally check some blocks from the start of the range
            if start_block < end_block - recent_range:
                early_range = min(5, end_block - start_block)
                for i in range(start_block, start_block + early_range):
                    blocks_to_check.append(i)
                    
            # Remove duplicates and limit
            blocks_to_check = list(set(blocks_to_check))[:max_blocks_to_check]
            logger.debug(f"Checking {len(blocks_to_check)} blocks for transactions involving {address_lower}")
            
            # Process blocks
            for block_num in blocks_to_check:
                try:
                    block = self.w3.eth.get_block(block_num, full_transactions=True)
                    for tx in block['transactions']:
                        # Skip if transaction doesn't involve our address
                        tx_from = tx.get('from', '').lower() 
                        tx_to = tx.get('to', '').lower() if tx.get('to') else None
                        
                        if tx_from == address_lower or tx_to == address_lower:
                            # Add transaction to history
                            status = 1  # Assume success if included in block
                            value = tx.get('value', 0)
                            
                            transactions.append({
                                'hash': tx.get('hash', '').hex(),
                                'from': tx_from,
                                'to': tx_to,
                                'value': value,
                                'block': block_num,
                                'status': status
                            })
                except Exception as e:
                    logger.debug(f"Error getting block {block_num}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error getting transaction history for {address}: {e}")
            
        return transactions

    def calculate_profit(self, start_balance, end_balance, start_price, end_price):
        """Calculate profit percentage based on balance changes and prices."""
        # Use the simple profit calculation from ProfitCalculator
        return ProfitCalculator.calculate_simple_profit(start_balance, end_balance, start_price, end_price)
        
    def scan_profitable_wallets(self, days, min_profit, num_samples=100, dev_mode=False):
        """
        Scan Ethereum for wallets with at least min_profit% over the past days.
        
        Args:
            days (int): Number of days to look back.
            min_profit (float): Minimum profit percentage threshold.
            num_samples (int): Number of blocks to sample for efficiency.
            dev_mode (bool): Enable faster development mode with limited processing.
        
        Returns:
            list: Dict with addresses and their profit percentages.
        """
        logger.info(f"Scanning Ethereum for profitable wallets over {days} days with {min_profit}% minimum profit")
        
        # Determine time range - using timezone-aware datetime
        end_datetime = datetime.now(timezone.utc)
        start_datetime = end_datetime - timedelta(days=days)
        start_timestamp = int(start_datetime.timestamp())
        end_block = self.w3.eth.block_number
        
        # Get start block with error handling
        try:
            start_block = self.get_block_by_timestamp(start_timestamp)
        except Exception as e:
            logger.error(f"Error getting start block: {e}")
            # Fallback: estimate block based on average block time (13 seconds)
            blocks_per_day = 24 * 60 * 60 / 13  # ~6646 blocks per day
            start_block = max(0, end_block - int(blocks_per_day * days))
            logger.info(f"Using estimated start block: {start_block}")
        
        logger.info(f"Scan range: Block {start_block} to {end_block} (approx. {days} days)")

        # Reduce the sample size if the range is too large, unless num_samples is 0
        if num_samples != 0 and end_block - start_block > 1000000:  # Huge range
            logger.warning("Large block range detected, reducing sample size")
            num_samples = min(50, num_samples)  # Reduce sample size to avoid rate limits

        # Sample blocks
        sampled_blocks = self.get_sampled_blocks(start_block, end_block, num_samples)
        
        # Collect active addresses and filter EOAs
        logger.info(f"Collecting addresses from {len(sampled_blocks)} sampled blocks")
        addresses = self.get_active_addresses_from_blocks(sampled_blocks)
        logger.info(f"Found {len(addresses)} unique addresses")
        
        # Filter to find EOAs
        eoas = []
        logger.info(f"Checking {len(addresses)} addresses to filter for EOAs...")
        eoa_start_time = time.time()  # Changed variable name from start_time to eoa_start_time
        last_log_time = eoa_start_time
        
        for i, addr in enumerate(addresses):
            current_time = time.time()
            # Log progress every 100 addresses or 30 seconds
            if i % 100 == 0 or current_time - last_log_time > 30:
                elapsed = current_time - eoa_start_time
                progress = (i / len(addresses)) * 100 if addresses else 0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(addresses) - i) / rate if rate > 0 else 0
                logger.info(f"EOA check progress: {i}/{len(addresses)} addresses ({progress:.1f}%) | Found {len(eoas)} EOAs | Rate: {rate:.1f} addr/sec | ETA: {format_time(eta)}")
                last_log_time = current_time
            
            try:
                if self.is_eoa(addr):
                    eoas.append(addr)
            except Exception as e:
                logger.warning(f"Error checking address {addr}: {e}")
                continue
            
        total_time = time.time() - eoa_start_time
        logger.info(f"EOA filtering completed in {format_time(total_time)}: checked {len(addresses)} addresses, found {len(eoas)} EOAs")
                
        logger.info(f"Filtered to {len(eoas)} EOA addresses")

        # If in dev mode, severely limit the number of EOAs to process
        if dev_mode:
            logger.info("Running in development mode - limiting EOA processing")
            max_eoas = 5
            if len(eoas) > max_eoas:
                logger.info(f"Limiting to {max_eoas} EOAs for development mode")
                eoas = eoas[:max_eoas]

        # Get historical ETH prices
        start_date = start_datetime.strftime("%d-%m-%Y")  # Changed from start_time to start_datetime
        end_date = end_datetime.strftime("%d-%m-%Y")  # Changed from end_time to end_datetime
        
        try:
            start_price = self.get_eth_price_on_date(start_date)
            end_price = self.get_eth_price_on_date(end_date)
            logger.info(f"ETH price: {start_price} USD on {start_date}, {end_price} USD on {end_date}")
        except ValueError as e:
            logger.error(f"Failed to get ETH price: {e}")
            return []

        # Calculate profits for each EOA with improved logic
        profitable_wallets = []
        realistic_wallets = []  # Track wallets with realistic profits separately
        logger.info(f"Analyzing wallet profitability for {len(eoas)} EOAs...")
        
        # Process EOAs in smaller batches to improve responsiveness
        batch_size = 10
        wallet_counter = 0  # Fix for undefined 'i' variable
        
        for batch_start in range(0, len(eoas), batch_size):
            batch_end = min(batch_start + batch_size, len(eoas))
            batch = eoas[batch_start:batch_end]
            
            logger.info(f"Processing EOA batch {batch_start//batch_size + 1}/{(len(eoas)-1)//batch_size + 1} " +
                       f"({batch_start}-{batch_end-1} of {len(eoas)})")
            
            for addr in batch:
                try:
                    wallet_counter += 1  # Increment counter for each wallet processed
                    
                    # Log progress occasionally
                    if wallet_counter % 2 == 0 and wallet_counter > 0:
                        logger.info(f"Processed {wallet_counter}/{len(eoas)} wallets, found {len(profitable_wallets)} profitable")
                        
                    # Get balances at start and end blocks
                    start_balance = self.get_balance(addr, start_block)
                    end_balance = self.get_balance(addr, end_block)
                    
                    # Get transaction history between blocks
                    transactions = self.get_transaction_history(addr, start_block, end_block)
                    
                    # Analyze transaction history
                    tx_analysis = ProfitCalculator.analyze_transaction_history(transactions, addr)
                    
                    # Calculate profit metrics using improved method
                    profit_metrics = ProfitCalculator.calculate_profit_with_transfers(
                        start_balance, 
                        end_balance, 
                        start_price, 
                        end_price,
                        tx_analysis.get('transfers_in', 0),
                        tx_analysis.get('transfers_out', 0)
                    )
                    
                    # Calculate confidence score with new metrics
                    confidence = ProfitCalculator.get_confidence_score({**profit_metrics, **tx_analysis})
                    
                    # Use realistic profit percent for filtering
                    realistic_profit = profit_metrics['realistic_profit_percent']
                    
                    # Filter out insignificant wallets (low confidence and unrealistic)
                    if realistic_profit >= min_profit:
                        wallet_data = {
                            'address': addr,
                            'profit_percent': realistic_profit,  # Use realistic profit
                            'raw_profit_percent': profit_metrics['adjusted_profit_percent'],  # Keep raw value for reference
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
                            'blockchain': 'ethereum',
                            'price_change_percent': profit_metrics['price_change_percent']
                        }
                        
                        profitable_wallets.append(wallet_data)
                        
                        # Track realistic wallets separately
                        if profit_metrics.get('is_realistic', True) and confidence > 30:
                            realistic_wallets.append(wallet_data)
                            
                except Exception as e:
                    logger.warning(f"Error calculating profit for {addr}: {e}")
                    continue
                
            # Add a small delay between batches to prevent rate limiting
            if batch_end < len(eoas):
                time.sleep(1)

        # Sort wallets by confidence and profit
        profitable_wallets.sort(key=lambda x: (x['confidence_score'], x['profit_percent']), reverse=True)
        
        # If we have enough realistic wallets, prefer those
        if len(realistic_wallets) >= 5:
            logger.info(f"Using {len(realistic_wallets)} realistic profitable wallets instead of {len(profitable_wallets)} total profitable wallets")
            return realistic_wallets
            
        logger.info(f"Found {len(profitable_wallets)} profitable wallets with ≥{min_profit}% profit")
        return profitable_wallets

    def estimate_scan_time(self, days, num_samples=100):
        """
        Estimate how long the scan will take based on parameters.
        
        Args:
            days (int): Number of days to look back
            num_samples (int): Number of blocks to sample
            
        Returns:
            dict: Estimated time ranges in seconds and human-readable format
        """
        # Base time for connecting and initialization
        base_time = 5
        
        # Time to find start block (binary search)
        block_search_time = 10
        
        # Time to collect addresses (depends on num_samples)
        # Each block fetch might take 1-3 seconds with potential rate limiting
        block_fetch_time = num_samples * 2
        
        # EOA checking time (assume ~500 addresses, 0.1s each)
        address_check_time = min(1000, 500) * 0.1
        
        # Price fetch time
        price_fetch_time = 4
        
        # Balance checking time (depends on number of EOAs, ~0.5s each)
        # Assume ~30% of addresses are EOAs
        eoa_count = min(1000, 150)
        balance_check_time = eoa_count * 2 * 0.5  # Each address needs 2 balance checks
        
        # Total estimated time
        min_time = base_time + block_search_time + block_fetch_time * 0.7 + address_check_time * 0.7 + price_fetch_time + balance_check_time * 0.7
        max_time = base_time + block_search_time * 2 + block_fetch_time * 2 + address_check_time * 1.5 + price_fetch_time * 2 + balance_check_time * 1.5
        
        # Format time estimates
        def format_time(seconds):
            if seconds < 60:
                return f"{int(seconds)} seconds"
            elif seconds < 3600:
                return f"{int(seconds/60)} minutes and {int(seconds%60)} seconds"
            else:
                return f"{int(seconds/3600)} hours, {int((seconds%3600)/60)} minutes"
                
        return {
            "min_seconds": min_time,
            "max_seconds": max_time,
            "min_time": format_time(min_time),
            "max_time": format_time(max_time),
            "factors_affecting_time": [
                "Network congestion",
                "RPC endpoint responsiveness",
                "Number of active addresses in sampled blocks",
                "Rate limiting on blockchain node",
                "Number of EOAs found"
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
    scanner = EthereumScanner()
    days = 1
    min_profit = 50
    num_samples = 2 # 0 = all, min = 2
    
    # First estimate the time
    time_estimate = scanner.estimate_scan_time(days, num_samples)
    print(f"Estimated scan time: {time_estimate['min_time']} to {time_estimate['max_time']}")
    
    print("Starting scan...")
    start_time = time.time()
    wallets = scanner.scan_profitable_wallets(days, min_profit, num_samples=num_samples, dev_mode=False)
    elapsed = time.time() - start_time
    
    print(f"Scan completed in {format_time(elapsed)}")
    print(f"Found {len(wallets)} wallets with >= {min_profit}% profit:")
    for wallet in wallets:
        print(f"Address: {wallet['address']}, Profit: {wallet['profit_percent']:.2f}%")