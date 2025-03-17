import logging
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
import time
import statistics
from datetime import datetime

logger = logging.getLogger("profit_calculator")

class ProfitCalculator:
    """
    Utility class for accurate profit calculations across blockchain wallets.
    With enhanced accuracy filters to prevent unrealistic profit reporting.
    """
    # Add caching for performance optimization
    _transaction_cache = {}
    _balance_cache = {}
    _code_cache = {}  # For EOA checks
    
    # Minimum thresholds for meaningful profit calculation
    MIN_START_VALUE_USD = 5.0  # Minimum starting value in USD
    MIN_END_VALUE_USD = 5.0    # Minimum ending value in USD
    MIN_ABSOLUTE_PROFIT_USD = 1.0  # Minimum absolute profit to consider
    
    @staticmethod
    def calculate_simple_profit(start_balance: float, end_balance: float, 
                               start_price: float, end_price: float) -> float:
        """Simple profit calculation based only on start and end balances."""
        start_value = start_balance * start_price
        end_value = end_balance * end_price
        
        if start_value > 0:
            return ((end_value - start_value) / start_value) * 100
        return 0
    
    @staticmethod
    def calculate_profit_with_transfers(start_balance: float, end_balance: float,
                                       start_price: float, end_price: float,
                                       transfers_in: float, transfers_out: float) -> Dict:
        """
        Calculate profit accounting for transfers with enhanced accuracy measures.
        """
        # Calculate USD values
        start_value_usd = start_balance * start_price
        end_value_usd = end_balance * end_price
        transfers_in_usd = transfers_in * end_price  # Assuming average price for simplicity
        transfers_out_usd = transfers_out * end_price  # Assuming average price for simplicity
        
        # Adjust end balance to account for transfers
        adjusted_end_value = end_value_usd - transfers_in_usd + transfers_out_usd
        
        # Calculate absolute profit in USD
        absolute_profit_usd = end_value_usd - start_value_usd
        
        # Calculate adjusted absolute profit (accounting for transfers)
        adjusted_absolute_profit_usd = adjusted_end_value - start_value_usd
        
        # Calculate simple profit percentage
        simple_profit_percent = 0
        if start_value_usd > ProfitCalculator.MIN_START_VALUE_USD:
            simple_profit_percent = ((end_value_usd - start_value_usd) / start_value_usd) * 100
        else:
            # For tiny starting values, use a more conservative calculation
            simple_profit_percent = min(1000, ((end_value_usd - start_value_usd) * 10))
        
        # Calculate adjusted profit percentage (accounting for transfers)
        adjusted_profit_percent = 0
        realistic_profit_percent = 0
        
        if start_value_usd > ProfitCalculator.MIN_START_VALUE_USD:
            adjusted_profit_percent = ((adjusted_end_value - start_value_usd) / start_value_usd) * 100
            
            # Apply caps for realistic profit to prevent unreasonable values
            if adjusted_profit_percent > 1000:
                # For extremely high percentages, use a logarithmic scale
                realistic_profit_percent = 1000 + (100 * (adjusted_profit_percent / 1000))
            else:
                realistic_profit_percent = adjusted_profit_percent
        else:
            # For tiny starting values, base profit percentage on absolute profit instead
            if adjusted_absolute_profit_usd > ProfitCalculator.MIN_ABSOLUTE_PROFIT_USD:
                adjusted_profit_percent = 100  # Default modest profit
                realistic_profit_percent = min(500, adjusted_absolute_profit_usd * 10)
        
        # Calculate HODL performance
        hodl_value = start_balance * end_price
        hodl_profit_percent = 0
        if start_value_usd > 0:
            hodl_profit_percent = ((hodl_value - start_value_usd) / start_value_usd) * 100
        
        # Determine significance level
        significance = "insignificant"
        if adjusted_absolute_profit_usd >= 1000:
            significance = "very_significant"
        elif adjusted_absolute_profit_usd >= 100:
            significance = "significant" 
        elif adjusted_absolute_profit_usd >= 10:
            significance = "moderate"
        elif adjusted_absolute_profit_usd >= ProfitCalculator.MIN_ABSOLUTE_PROFIT_USD:
            significance = "minor"
            
        # Detect unrealistic profits due to dust values or mathematical edge cases
        is_realistic = True
        reason = ""
        
        if start_value_usd < ProfitCalculator.MIN_START_VALUE_USD:
            is_realistic = False
            reason = "starting_value_too_small"
        elif end_value_usd < ProfitCalculator.MIN_END_VALUE_USD:
            is_realistic = False
            reason = "ending_value_too_small"
        elif absolute_profit_usd < ProfitCalculator.MIN_ABSOLUTE_PROFIT_USD:
            is_realistic = False
            reason = "absolute_profit_too_small"
        elif adjusted_profit_percent > 10000 and adjusted_absolute_profit_usd < 100:
            is_realistic = False
            reason = "unrealistic_percentage_for_small_amount"
            
        return {
            "simple_profit_percent": simple_profit_percent,
            "adjusted_profit_percent": adjusted_profit_percent,
            "realistic_profit_percent": realistic_profit_percent,  # New field with more reasonable values
            "hodl_profit_percent": hodl_profit_percent,
            "absolute_profit_usd": absolute_profit_usd,
            "adjusted_absolute_profit_usd": adjusted_absolute_profit_usd,
            "start_value_usd": start_value_usd,
            "end_value_usd": end_value_usd,
            "transfers_in_usd": transfers_in_usd,
            "transfers_out_usd": transfers_out_usd,
            "price_change_percent": ((end_price - start_price) / start_price * 100) if start_price > 0 else 0,
            "significance": significance,
            "is_realistic": is_realistic,
            "unrealistic_reason": reason
        }
    
    @staticmethod
    def analyze_transaction_history(transactions: List[Dict], address: str) -> Dict:
        """
        Analyze transaction history with enhanced metrics for better accuracy.
        """
        transfers_in = 0.0
        transfers_out = 0.0
        
        # Keep track of value-weighted metrics
        value_weighted_txns = 0.0
        total_txn_value = 0.0
        
        # Calculate unique addresses interacted with
        unique_addresses = set()
        significant_txns = 0  # Transactions with value > 0.01 ETH/BNB
        
        # Transaction count by type
        tx_counts = {"in": 0, "out": 0, "self": 0}
        
        # Value-based metrics
        tx_values = []  # Store transaction values for statistical analysis
        
        for tx in transactions:
            # Skip failed transactions
            if tx.get("status") == 0:
                continue
                
            from_addr = tx.get("from", "").lower()
            to_addr = tx.get("to", "").lower() if tx.get("to") else None
            value = float(tx.get("value", 0)) / 1e18  # Convert from wei/gwei to ETH/BNB
            
            # Skip dust transactions
            if value < 0.0001:  # Ignore extremely small transactions
                continue
                
            # Record transaction value for statistical analysis
            tx_values.append(value)
            total_txn_value += value
            
            # Determine transaction type
            if from_addr == address.lower() and to_addr == address.lower():
                tx_counts["self"] += 1
            elif from_addr == address.lower():
                tx_counts["out"] += 1
                transfers_out += value
                if to_addr:
                    unique_addresses.add(to_addr)
            elif to_addr == address.lower():
                tx_counts["in"] += 1
                transfers_in += value
                if from_addr:
                    unique_addresses.add(from_addr)
                    
            # Count significant transactions
            if value >= 0.01:
                significant_txns += 1
                
            # Value-weighted transaction count (gives more weight to larger transactions)
            value_weighted_txns += min(1.0, value)  # Cap at 1.0 per transaction
        
        # Calculate activity metrics
        transaction_count = sum(tx_counts.values())
        unique_address_count = len(unique_addresses)
        
        # Statistical metrics if we have enough transactions
        avg_tx_value = median_tx_value = tx_value_stdev = 0
        if tx_values:
            avg_tx_value = sum(tx_values) / len(tx_values)
            try:
                median_tx_value = statistics.median(tx_values)
                if len(tx_values) > 1:
                    tx_value_stdev = statistics.stdev(tx_values)
            except Exception:
                # Fall back if statistics functions fail
                median_tx_value = avg_tx_value
        
        # Transaction value distribution metrics
        has_meaningful_activity = significant_txns >= 2
        txn_quality_score = min(100, (significant_txns * 10) + (value_weighted_txns * 20))
        
        return {
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "transaction_count": transaction_count,
            "significant_transaction_count": significant_txns,
            "unique_addresses": unique_address_count,
            "tx_in_count": tx_counts["in"],
            "tx_out_count": tx_counts["out"],
            "tx_self_count": tx_counts["self"],
            "avg_tx_value": avg_tx_value,
            "median_tx_value": median_tx_value, 
            "total_tx_value": total_txn_value,
            "value_weighted_txns": value_weighted_txns,
            "has_meaningful_activity": has_meaningful_activity,
            "txn_quality_score": txn_quality_score
        }
    
    @staticmethod
    def get_confidence_score(metrics: Dict) -> int:
        """Calculate confidence score for profit calculations (0-100)."""
        score = 50  # Start with neutral score
        
        # Adjust based on transaction quality
        if metrics.get("txn_quality_score", 0) > 60:
            score += 15
        elif metrics.get("txn_quality_score", 0) > 30:
            score += 7
            
        # Higher confidence with more significant transactions
        if metrics.get("significant_transaction_count", 0) > 5:
            score += 15
        elif metrics.get("significant_transaction_count", 0) > 2:
            score += 8
            
        # Lower confidence for unrealistic profits
        if metrics.get("is_realistic", True) == False:
            score -= 30
            
        # Higher confidence with higher USD values
        if metrics.get("end_value_usd", 0) > 1000:
            score += 10
        elif metrics.get("end_value_usd", 0) > 100:
            score += 5
            
        # Lower score for tiny values
        if metrics.get("adjusted_absolute_profit_usd", 0) < 10:
            score -= 15
        
        # Significance directly impacts confidence
        significance = metrics.get("significance", "insignificant")
        if significance == "very_significant":
            score += 20
        elif significance == "significant":
            score += 10
        elif significance == "moderate":
            score += 5
        elif significance == "insignificant":
            score -= 15
        
        # Cap score between 0-100
        return max(0, min(100, score))
        
    @classmethod
    def cache_transaction(cls, tx_hash, transaction_data):
        """Cache transaction data by hash"""
        cls._transaction_cache[tx_hash] = transaction_data
        
    @classmethod
    def get_cached_transaction(cls, tx_hash):
        """Retrieve transaction from cache"""
        return cls._transaction_cache.get(tx_hash)
        
    @classmethod
    def cache_balance(cls, address, block, balance):
        """Cache balance for an address at a specific block"""
        key = f"{address.lower()}_{block}"
        cls._balance_cache[key] = (balance, time.time())
        
    @classmethod
    def get_cached_balance(cls, address, block):
        """Get cached balance if available and not too old"""
        key = f"{address.lower()}_{block}"
        if key in cls._balance_cache:
            balance, timestamp = cls._balance_cache[key]
            # Cache is valid for 10 minutes
            if time.time() - timestamp < 600:
                return balance
        return None
        
    @classmethod
    def cache_code(cls, address, code):
        """Cache contract code check results"""
        cls._code_cache[address.lower()] = code
        
    @classmethod
    def get_cached_code(cls, address):
        """Get cached code check result"""
        return cls._code_cache.get(address.lower())
