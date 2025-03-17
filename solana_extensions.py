"""
Extension modules for the Solana scanner.
This file defines interfaces that can be implemented to enhance the Solana scanner.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import time

class SolanaTokenAnalyzer(ABC):
    """Interface for analyzing SPL tokens in Solana wallets."""
    
    @abstractmethod
    def get_token_balances(self, wallet_address: str) -> Dict[str, float]:
        """
        Get all SPL token balances for a wallet.
        
        Args:
            wallet_address: Solana wallet public key
            
        Returns:
            Dictionary mapping token addresses to token balances
        """
        pass
        
    @abstractmethod
    def get_token_price_history(self, token_address: str, start_time: int, end_time: int) -> Dict[int, float]:
        """
        Get historical price data for a token.
        
        Args:
            token_address: Token mint address
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Dictionary mapping timestamps to prices
        """
        pass
        
    @abstractmethod
    def calculate_token_profit(self, wallet_address: str, token_address: str, 
                             start_time: int, end_time: int) -> Dict[str, Any]:
        """
        Calculate profit for a specific token in a wallet.
        
        Returns:
            Dictionary with profit metrics
        """
        pass

class SolanaProgramDetector(ABC):
    """Interface for identifying interactions with specific Solana programs."""
    
    @abstractmethod
    def detect_program_interactions(self, wallet_address: str, 
                                  program_ids: List[str]) -> Dict[str, int]:
        """
        Detect interactions with specific programs.
        
        Args:
            wallet_address: Wallet to analyze
            program_ids: List of program IDs to check for
            
        Returns:
            Dictionary mapping program IDs to interaction counts
        """
        pass
        
    @abstractmethod
    def get_common_programs(self) -> Dict[str, str]:
        """
        Get a dictionary of common Solana program IDs and their names.
        
        Returns:
            Dictionary mapping program IDs to human-readable names
        """
        pass

class SolanaWalletProfiler:
    """Helper class to profile Solana wallet activity patterns."""
    
    def __init__(self, scanner):
        self.scanner = scanner
        self.activity_cache = {}
        
    def generate_wallet_profile(self, address: str) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of wallet activity.
        
        Args:
            address: Wallet address to profile
            
        Returns:
            Dictionary with wallet profile metrics
        """
        # Implementation would analyze:
        # - Activity patterns (times of day, days of week)
        # - Transaction frequency over time
        # - Typical transaction sizes
        # - Common interaction partners
        # - Program usage patterns
        # This is a placeholder for future implementation
        return {
            "address": address,
            "profile_generated": time.time(),
            "activity_level": "unknown",
            "wallet_age_days": 0,
            "transaction_pattern": "unknown",
        }
