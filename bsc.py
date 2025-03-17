from web3 import Web3
import requests
from datetime import datetime, timezone, timedelta
import logging
import time
import random
from chain_scanner_template import BlockchainScanner
from profit_calculator import ProfitCalculator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bsc_scanner")

# BSC RPC endpoints
BSC_NODE_URL = "https://bsc-dataseed.binance.org/"

class BscScanner(BlockchainScanner):
    def __init__(self, node_url=BSC_NODE_URL):
        super().__init__(node_url)
        self.w3 = Web3(Web3.HTTPProvider(node_url))
        
        # Fix for BSC PoA chain
        # Try a simplified approach that's more compatible with web3.py 6.0+
        try:
            # Simple way to ignore extraData validation errors
            from web3.exceptions import ExtraDataLengthError
            
            def ignore_poa_middleware(make_request, w3):
                """Middleware to ignore extraData validation errors in PoA chains."""
                def middleware(method, params):
                    try:
                        return make_request(method, params)
                    except ExtraDataLengthError:
                        logger.debug(f"Bypassing ExtraDataLengthError for method {method}")
                        # Return empty result for problematic methods
                        if method == "eth_getBlockByNumber" or method == "eth_getBlockByHash":
                            # Retry with a different approach
                            try:
                                # Try with different params
                                if method == "eth_getBlockByNumber":
                                    # For eth_getBlockByNumber, try without full transactions
                                    if len(params) > 1:
                                        return make_request(method, [params[0], False])
                                return make_request(method, params)
                            except:
                                logger.warning(f"Still failed to process {method} after retry")
                                return {"jsonrpc": "2.0", "id": 1, "result": None}
                        return make_request(method, params)
                return middleware
            
            # Add the middleware
            self.w3.middleware_onion.add(ignore_poa_middleware)
            logger.info("Added custom PoA middleware to handle BSC compatibility")
            
        except Exception as e:
            logger.warning(f"Failed to add PoA middleware: {str(e)}")
            logger.warning("BSC may not work correctly without PoA middleware")
        
        # Try a different RPC endpoint if the first one doesn't connect
        try:
            if not self.w3.is_connected():
                # Try alternative BSC endpoints
                alt_endpoints = [
                    "https://bsc-dataseed1.binance.org/",
                    "https://bsc-dataseed2.binance.org/",
                    "https://bsc-dataseed3.binance.org/",
                    "https://bsc-dataseed4.binance.org/"
                ]
                
                for endpoint in alt_endpoints:
                    logger.info(f"Trying alternative BSC endpoint: {endpoint}")
                    try:
                        self.w3 = Web3(Web3.HTTPProvider(endpoint))
                        if self.w3.is_connected():
                            logger.info(f"Connected to alternative BSC endpoint: {endpoint}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to connect to {endpoint}: {str(e)}")
            
            if not self.w3.is_connected():
                logger.error("Failed to connect to any BSC node")
                raise ConnectionError("Cannot connect to BSC node")
                
            # Test a simple call before proceeding
            current_block = self.w3.eth.block_number
            logger.info(f"Connected to BSC node, current block: {current_block}")
            
        except Exception as e:
            logger.error(f"Failed to connect to BSC node: {str(e)}")
            raise ConnectionError(f"Cannot connect to BSC node: {str(e)}")
        
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
        """Get a list of block numbers sampled evenly over the range."""
        range_size = end_block - start_block
        step = max(1, range_size // (num_samples - 1))
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
        
        logger.info(f"Starting to process {total_blocks} BSC blocks...")
        
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
                    if tx['to']:
                        addresses.add(tx['to'])
                successful_blocks += 1
                
                # Add small delay to avoid rate limiting
                if random.random() < 0.2:  # 20% chance to sleep
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
    
    def is_valid_wallet(self, address):
        """Check if an address is a valid wallet (EOA vs contract) with caching."""
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
            return False
    
    def get_balance(self, address, block):
        """Get the BNB balance of an address at a specific block with caching."""
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
    
    def get_token_price_on_date(self, date):
        """Fetch BNB price in USD for a given date (dd-mm-yyyy)."""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/binancecoin/history?date={date}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data['market_data']['current_price']['usd']
        except (requests.RequestException, KeyError) as e:
            logger.error(f"Error fetching BNB price for {date}: {e}")
            raise ValueError(f"Could not get BNB price for {date}")
    
    def scan_profitable_wallets(self, days, min_profit, num_samples=100, dev_mode=False):
        """Scan BSC for wallets with at least min_profit% over the past days."""
        logger.info(f"Scanning BSC for profitable wallets over {days} days with {min_profit}% minimum profit")
        
        # Determine time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        start_timestamp = int(start_time.timestamp())
        end_block = self.w3.eth.block_number
        
        # BSC has faster block times, be careful with block estimation
        try:
            start_block = self.get_block_by_timestamp(start_timestamp)
        except Exception as e:
            logger.error(f"Error getting start block: {e}")
            # Fallback: estimate block based on BSC's 3 second block time
            blocks_per_day = 24 * 60 * 60 / 3  # ~28,800 blocks per day
            start_block = max(0, end_block - int(blocks_per_day * days))
            logger.info(f"Using estimated start block: {start_block}")
        
        logger.info(f"Scan range: Block {start_block} to {end_block} (approx. {days} days)")
        
        # Adjust sample size for BSC's larger block count
        adjusted_num_samples = min(num_samples, 80)  # Limit to 80 samples due to BSC's high TPS
        sampled_blocks = self.get_sampled_blocks(start_block, end_block, adjusted_num_samples)
        
        logger.info(f"Collecting addresses from {len(sampled_blocks)} sampled blocks")
        addresses = self.get_active_addresses_from_blocks(sampled_blocks)
        logger.info(f"Found {len(addresses)} unique addresses")
        
        # Limit address count to avoid timeouts
        max_addresses = 800
        if len(addresses) > max_addresses:
            logger.info(f"Limiting to {max_addresses} addresses for wallet check")
            address_sample = random.sample(list(addresses), max_addresses)
        else:
            address_sample = addresses
            
        valid_wallets = []
        for addr in address_sample:
            try:
                if self.is_valid_wallet(addr):
                    valid_wallets.append(addr)
            except Exception as e:
                logger.warning(f"Error checking address {addr}: {e}")
                continue
                
        logger.info(f"Filtered to {len(valid_wallets)} valid wallet addresses")

        # If in dev mode, severely limit the number of wallets to process
        if dev_mode:
            logger.info("Running in development mode - limiting wallet processing")
            max_wallets = 5
            if len(valid_wallets) > max_wallets:
                logger.info(f"Limiting to {max_wallets} wallets for development mode")
                valid_wallets = valid_wallets[:max_wallets]

        # Get historical BNB prices
        start_date = start_time.strftime("%d-%m-%Y")
        end_date = end_time.strftime("%d-%m-%Y")
        
        try:
            start_price = self.get_token_price_on_date(start_date)
            end_price = self.get_token_price_on_date(end_date)
            logger.info(f"BNB price: {start_price} USD on {start_date}, {end_price} USD on {end_date}")
        except ValueError as e:
            logger.error(f"Failed to get BNB price: {e}")
            return []

        # Calculate profits for each wallet with improved logic
        profitable_wallets = []
        logger.info(f"Analyzing wallet profitability for {len(valid_wallets)} wallets...")
        
        # Process wallets in smaller batches to improve responsiveness
        batch_size = 10
        for batch_start in range(0, len(valid_wallets), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_wallets))
            batch = valid_wallets[batch_start:batch_end]
            
            logger.info(f"Processing wallet batch {batch_start//batch_size + 1}/{(len(valid_wallets)-1)//batch_size + 1} " +
                       f"({batch_start}-{batch_end-1} of {len(valid_wallets)})")
            
            for addr in batch:
                try:
                    # Log progress occasionally
                    if batch_start % 50 == 0 and batch_start > 0:
                        logger.info(f"Processed {batch_start}/{len(valid_wallets)} wallets, found {len(profitable_wallets)} profitable")
                        
                    # Get balances at start and end blocks
                    start_balance = self.get_balance(addr, start_block)
                    end_balance = self.get_balance(addr, end_block)
                    
                    # Get transaction history between blocks
                    transactions = self.get_transaction_history(addr, start_block, end_block)
                    
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
                    
                    # Use adjusted profit percent for comparison with threshold
                    profit_percent = profit_metrics['adjusted_profit_percent']
                    
                    if profit_percent >= min_profit:
                        profitable_wallets.append({
                            'address': addr,
                            'profit_percent': profit_percent,
                            'start_balance': start_balance,
                            'end_balance': end_balance,
                            'simple_profit_percent': profit_metrics['simple_profit_percent'],
                            'hodl_profit_percent': profit_metrics['hodl_profit_percent'],
                            'absolute_profit_usd': profit_metrics['absolute_profit_usd'],
                            'transfers_in': tx_analysis.get('transfers_in', 0),
                            'transfers_out': tx_analysis.get('transfers_out', 0),
                            'transaction_count': tx_analysis.get('transaction_count', 0),
                            'confidence_score': confidence,
                            'blockchain': 'bsc',
                            'price_change_percent': profit_metrics['price_change_percent']
                        })
                except Exception as e:
                    logger.warning(f"Error calculating profit for {addr}: {e}")
                    continue
                
            # Add a small delay between batches to prevent rate limiting
            if batch_end < len(valid_wallets):
                time.sleep(1)

        logger.info(f"Found {len(profitable_wallets)} profitable wallets with ≥{min_profit}% profit")
        return profitable_wallets

    def get_transaction_history(self, address, start_block, end_block):
        """Get transaction history using a more efficient block sampling approach."""
        transactions = []
        address_lower = address.lower()
        
        try:
            # Optimize by checking blocks in reverse chronological order
            # and only checking a limited number
            max_blocks_to_check = 15  # BSC has more TPS so check fewer blocks
            blocks_to_check = []
            
            # First try some recent blocks
            recent_range = min(15, end_block - start_block)
            for i in range(end_block, end_block - recent_range, -1):
                blocks_to_check.append(i)
                
            # Then try some blocks from the middle of the range
            if start_block < end_block - recent_range:
                mid_point = (start_block + end_block) // 2
                for i in range(3):  # 3 blocks around midpoint (fewer than ETH)
                    offset = i - 1
                    if start_block <= mid_point + offset <= end_block:
                        blocks_to_check.append(mid_point + offset)
                        
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
        return ProfitCalculator.calculate_simple_profit(start_balance, end_balance, start_price, end_price)
    
    def estimate_scan_time(self, days, num_samples=100):
        """
        Estimate how long the scan will take based on parameters.
        
        Args:
            days (int): Number of days to look back
            num_samples (int): Number of blocks to sample
            
        Returns:
            dict: Estimated time ranges in seconds and human-readable format
        """
        # Note: BSC has faster blocks but sometimes less reliable public nodes
        
        # Base time for connecting and initialization
        base_time = 5
        
        # Time to find start block (binary search)
        block_search_time = 15  # BSC might have more blocks to search
        
        # Time to collect addresses (depends on num_samples)
        # Each block fetch might take 1-3 seconds with potential rate limiting
        adjusted_num_samples = min(num_samples, 80)
        block_fetch_time = adjusted_num_samples * 2.5  # BSC blocks might be larger
        
        # Wallet checking time (assume ~500 addresses, 0.1s each)
        address_check_time = min(800, 500) * 0.1
        
        # Price fetch time
        price_fetch_time = 4
        
        # Balance checking time (depends on number of EOAs, ~0.5s each)
        # Assume ~30% of addresses are EOAs
        eoa_count = min(800, 200) 
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
                "Rate limiting on BSC node",
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
    scanner = BscScanner()
    days = 30
    min_profit = 50
    num_samples = 50
    
    # First estimate the time
    time_estimate = scanner.estimate_scan_time(days, num_samples)
    print(f"Estimated scan time: {time_estimate['min_time']} to {time_estimate['max_time']}")
    
    print("Starting scan...")
    start_time = time.time()
    wallets = scanner.scan_profitable_wallets(days, min_profit, num_samples=num_samples)
    elapsed = time.time() - start_time
    
    print(f"Scan completed in {format_time(elapsed)}")
    print(f"Found {len(wallets)} wallets with >= {min_profit}% profit:")
    for wallet in wallets:
        print(f"Address: {wallet['address']}, Profit: {wallet['profit_percent']:.2f}%")