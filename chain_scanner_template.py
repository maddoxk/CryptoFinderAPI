from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import logging

class BlockchainScanner(ABC):
    """Abstract base class for blockchain scanners."""
    
    def __init__(self, node_url):
        """Initialize the blockchain scanner with a node URL."""
        self.node_url = node_url
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    def get_block_by_timestamp(self, target_timestamp):
        """Find the block number closest to the target timestamp."""
        pass
    
    @abstractmethod
    def get_sampled_blocks(self, start_block, end_block, num_samples):
        """Get a list of block numbers sampled evenly over the range."""
        pass
    
    @abstractmethod
    def get_active_addresses_from_blocks(self, block_numbers):
        """Collect unique addresses from transactions in the given blocks."""
        pass
    
    @abstractmethod
    def is_valid_wallet(self, address):
        """Check if an address is a valid wallet (vs contract or other entity)."""
        pass
    
    @abstractmethod
    def get_balance(self, address, block):
        """Get the native token balance of an address at a specific block."""
        pass
    
    @abstractmethod
    def get_token_price_on_date(self, date):
        """Fetch token price in USD for a given date."""
        pass
    
    def calculate_profit(self, start_balance, end_balance, start_price, end_price):
        """Calculate profit percentage based on balance changes and prices."""
        start_value = start_balance * start_price
        end_value = end_balance * end_price
        if start_value > 0:
            return ((end_value - start_value) / start_value) * 100
        return 0
    
    @abstractmethod
    def scan_profitable_wallets(self, days, min_profit, num_samples=100):
        """
        Scan blockchain for wallets with at least min_profit% over the past days.
        
        Returns:
            list: Dict with addresses and their profit percentages.
        """
        pass

    @abstractmethod
    def estimate_scan_time(self, days, num_samples=100):
        """
        Estimate how long the scan will take based on parameters.
        
        Args:
            days (int): Number of days to look back
            num_samples (int): Number of blocks to sample
            
        Returns:
            dict: Estimated time ranges in seconds and human-readable format
        """
        pass
