#!/usr/bin/env python3
"""
Tennis-505: Match Data Processor

Processes real tennis match data and extracts quantifiable statistics
for use with the evidence-based prediction system.

Supports multiple data formats:
- CSV files with match results
- Real-time tournament data
- Historical match databases

Integrates with quantifiable_stats.py for statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import re
from datetime import datetime

# Import our quantifiable stats module
try:
    from ..features.quantifiable_stats import (
        QuantifiableTennisStats, ServeStatistics, ReturnStatistics, Surface
    )
except ImportError:
    # For standalone testing
    import sys
    sys.path.append('..')
    from features.quantifiable_stats import (
        QuantifiableTennisStats, ServeStatistics, ReturnStatistics, Surface
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchData:
    """Structured match data with all relevant information."""
    
    # Match metadata
    date: datetime
    tournament: str
    round_name: str
    surface: Surface
    
    # Players
    player1: str
    player2: str
    winner: str
    
    # Score and match info
    score: str
    duration_minutes: Optional[int] = None
    
    # Statistics (if available)
    player1_stats: Optional[Dict] = None
    player2_stats: Optional[Dict] = None

class MatchProcessor:
    """
    Processes tennis match data and extracts quantifiable statistics
    for prediction system analysis.
    """
    
    def __init__(self):
        """Initialize match processor with statistics calculator."""
        self.stats_calculator = QuantifiableTennisStats()
        
        # Surface mappings for different data formats
        self.surface_mappings = {
            'clay': Surface.CLAY,
            'hard': Surface.HARD, 
            'grass': Surface.GRASS,
            'carpet': Surface.HARD,  # Treat carpet as hard court
            'indoor hard': Surface.HARD,
            'outdoor hard': Surface.HARD,
            'red clay': Surface.CLAY,
            'green clay': Surface.CLAY
        }
        
        logger.info("Match processor initialized with quantifiable stats calculator")
    
    def load_csv_matches(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load tennis matches from CSV file.
        
        Expected CSV format:
        Date, Tournament, Category, Round, Player1, Player2, Winner, Score
        
        Args:
            file_path: Path to CSV file with match data
            
        Returns:
            DataFrame with processed match data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['Date', 'Tournament', 'Player1', 'Player2', 'Winner', 'Score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Infer surface from tournament name if not provided
            if 'Surface' not in df.columns:
                df['Surface'] = df['Tournament'].apply(self._infer_surface_from_tournament)
            
            logger.info(f"Loaded {len(df)} matches from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV matches: {e}")
            raise
    
    def _infer_surface_from_tournament(self, tournament_name: str) -> str:
        """
        Infer court surface from tournament name using common patterns.
        
        Args:
            tournament_name: Name of the tournament
            
        Returns:
            Surface type as string
        """
        tournament_lower = tournament_name.lower()
        
        # Clay court tournaments
        clay_indicators = ['french', 'roland garros', 'monte carlo', 'madrid', 
                          'rome', 'barcelona', 'hamburg', 'gstaad']
        if any(indicator in tournament_lower for indicator in clay_indicators):
            return 'clay'
        
        # Grass court tournaments  
        grass_indicators = ['wimbledon', 'queens', 'halle', 'eastbourne', 
                           'newport', 'mallorca']
        if any(indicator in tournament_lower for indicator in grass_indicators):
            return 'grass'
            
        # Default to hard court (most common)
        return 'hard'
    
    def extract_match_statistics(self, match_row: pd.Series) -> MatchData:
        """
        Extract structured match data from DataFrame row.
        
        Args:
            match_row: Single row from matches DataFrame
            
        Returns:
            MatchData object with structured information
        """
        # Determine surface
        surface_str = match_row.get('Surface', 'hard').lower()
        surface = self.surface_mappings.get(surface_str, Surface.HARD)
        
        # Extract match duration from score if possible
        duration = self._extract_duration_from_score(match_row['Score'])
        
        match_data = MatchData(
            date=match_row['Date'],
            tournament=match_row['Tournament'],
            round_name=match_row.get('Round', 'Unknown'),
            surface=surface,
            player1=match_row['Player1'],
            player2=match_row['Player2'], 
            winner=match_row['Winner'],
            score=match_row['Score'],
            duration_minutes=duration
        )
        
        return match_data
    
    def _extract_duration_from_score(self, score: str) -> Optional[int]:
        """
        Extract match duration from score string if available.
        
        Many data sources don't include duration, so this returns None
        for basic CSV data. Can be extended for richer data sources.
        
        Args:
            score: Match score string
            
        Returns:
            Duration in minutes if extractable, None otherwise
        """
        # Basic implementation - extend based on data format
        return None
    
    def analyze_player_performance_trends(self, matches_df: pd.DataFrame, 
                                        player_name: str,
                                        recent_matches: int = 10) -> Dict[str, any]:
        """
        Analyze recent performance trends for a specific player.
        
        Args:
            matches_df: DataFrame with match data
            player_name: Name of player to analyze
            recent_matches: Number of recent matches to analyze
            
        Returns:
            Dict with performance trend analysis
        """
        # Filter matches for this player
        player_matches = matches_df[
            (matches_df['Player1'] == player_name) | 
            (matches_df['Player2'] == player_name)
        ].copy()
        
        # Sort by date (most recent first)
        player_matches = player_matches.sort_values('Date', ascending=False)
        
        # Take recent matches
        recent = player_matches.head(recent_matches)
        
        if len(recent) == 0:
            return {"error": f"No matches found for player {player_name}"}
        
        # Calculate win rate
        wins = len(recent[recent['Winner'] == player_name])
        win_rate = wins / len(recent)
        
        # Surface performance breakdown
        surface_performance = {}
        for surface in recent['Surface'].unique():
            surface_matches = recent[recent['Surface'] == surface]
            surface_wins = len(surface_matches[surface_matches['Winner'] == player_name])
            surface_performance[surface] = {
                'matches': len(surface_matches),
                'wins': surface_wins,
                'win_rate': surface_wins / len(surface_matches) if len(surface_matches) > 0 else 0
            }
        
        # Recent form (last 5 matches)
        last_5 = recent.head(5)
        recent_form = [1 if match['Winner'] == player_name else 0 for _, match in last_5.iterrows()]
        
        return {
            'player': player_name,
            'total_recent_matches': len(recent),
            'overall_win_rate': win_rate,
            'recent_wins': wins,
            'recent_losses': len(recent) - wins,
            'surface_performance': surface_performance,
            'recent_form_5': recent_form,
            'momentum_score': sum(recent_form) / len(recent_form) if recent_form else 0,
            'analysis_period': {
                'from': recent['Date'].min().strftime('%Y-%m-%d'),
                'to': recent['Date'].max().strftime('%Y-%m-%d')
            }
        }
    
    def calculate_head_to_head(self, matches_df: pd.DataFrame, 
                             player1: str, player2: str) -> Dict[str, any]:
        """
        Calculate head-to-head statistics between two players.
        
        Args:
            matches_df: DataFrame with match data
            player1: First player name
            player2: Second player name
            
        Returns:
            Dict with head-to-head analysis
        """
        # Find all matches between these players
        h2h_matches = matches_df[
            ((matches_df['Player1'] == player1) & (matches_df['Player2'] == player2)) |
            ((matches_df['Player1'] == player2) & (matches_df['Player2'] == player1))
        ].copy()
        
        if len(h2h_matches) == 0:
            return {
                'player1': player1,
                'player2': player2,
                'total_matches': 0,
                'error': 'No head-to-head matches found'
            }
        
        # Count wins for each player
        player1_wins = len(h2h_matches[h2h_matches['Winner'] == player1])
        player2_wins = len(h2h_matches[h2h_matches['Winner'] == player2])
        
        # Surface breakdown
        surface_h2h = {}
        for surface in h2h_matches['Surface'].unique():
            surface_matches = h2h_matches[h2h_matches['Surface'] == surface]
            p1_surface_wins = len(surface_matches[surface_matches['Winner'] == player1])
            p2_surface_wins = len(surface_matches[surface_matches['Winner'] == player2])
            
            surface_h2h[surface] = {
                'total_matches': len(surface_matches),
                f'{player1}_wins': p1_surface_wins,
                f'{player2}_wins': p2_surface_wins
            }
        
        # Recent encounters (last 3 matches)
        recent_h2h = h2h_matches.sort_values('Date', ascending=False).head(3)
        recent_results = []
        for _, match in recent_h2h.iterrows():
            recent_results.append({
                'date': match['Date'].strftime('%Y-%m-%d'),
                'tournament': match['Tournament'],
                'surface': match['Surface'],
                'winner': match['Winner'],
                'score': match['Score']
            })
        
        return {
            'player1': player1,
            'player2': player2,
            'total_matches': len(h2h_matches),
            'player1_wins': player1_wins,
            'player2_wins': player2_wins,
            'player1_win_rate': player1_wins / len(h2h_matches),
            'player2_win_rate': player2_wins / len(h2h_matches),
            'surface_breakdown': surface_h2h,
            'recent_encounters': recent_results,
            'first_meeting': h2h_matches['Date'].min().strftime('%Y-%m-%d'),
            'last_meeting': h2h_matches['Date'].max().strftime('%Y-%m-%d')
        }
    
    def generate_match_prediction_features(self, matches_df: pd.DataFrame,
                                         player1: str, player2: str,
                                         surface: str = 'hard') -> Dict[str, any]:
        """
        Generate prediction features for an upcoming match between two players.
        
        Combines recent form, head-to-head, and surface-specific performance.
        
        Args:
            matches_df: DataFrame with historical match data
            player1: First player name
            player2: Second player name  
            surface: Court surface for upcoming match
            
        Returns:
            Dict with prediction features for both players
        """
        # Get recent performance for both players
        p1_performance = self.analyze_player_performance_trends(matches_df, player1, 10)
        p2_performance = self.analyze_player_performance_trends(matches_df, player2, 10)
        
        # Get head-to-head record
        h2h = self.calculate_head_to_head(matches_df, player1, player2)
        
        # Surface-specific performance
        p1_surface_perf = p1_performance.get('surface_performance', {}).get(surface, {
            'matches': 0, 'wins': 0, 'win_rate': 0.5
        })
        p2_surface_perf = p2_performance.get('surface_performance', {}).get(surface, {
            'matches': 0, 'wins': 0, 'win_rate': 0.5  
        })
        
        # Calculate momentum factors (based on our research)
        p1_momentum = self._calculate_momentum_score(p1_performance.get('recent_form_5', []))
        p2_momentum = self._calculate_momentum_score(p2_performance.get('recent_form_5', []))
        
        return {
            'match_info': {
                'player1': player1,
                'player2': player2,
                'surface': surface
            },
            'player1_features': {
                'recent_win_rate': p1_performance.get('overall_win_rate', 0.5),
                'surface_win_rate': p1_surface_perf['win_rate'],
                'surface_matches': p1_surface_perf['matches'],
                'momentum_score': p1_momentum,
                'recent_form': p1_performance.get('recent_form_5', [])
            },
            'player2_features': {
                'recent_win_rate': p2_performance.get('overall_win_rate', 0.5),
                'surface_win_rate': p2_surface_perf['win_rate'], 
                'surface_matches': p2_surface_perf['matches'],
                'momentum_score': p2_momentum,
                'recent_form': p2_performance.get('recent_form_5', [])
            },
            'head_to_head': {
                'total_matches': h2h.get('total_matches', 0),
                'player1_h2h_wins': h2h.get('player1_wins', 0),
                'player2_h2h_wins': h2h.get('player2_wins', 0),
                'player1_h2h_rate': h2h.get('player1_win_rate', 0.5),
                'player2_h2h_rate': h2h.get('player2_win_rate', 0.5)
            },
            'data_quality': {
                'p1_recent_matches': p1_performance.get('total_recent_matches', 0),
                'p2_recent_matches': p2_performance.get('total_recent_matches', 0),
                'h2h_sample_size': h2h.get('total_matches', 0)
            }
        }
    
    def _calculate_momentum_score(self, recent_form: List[int]) -> float:
        """
        Calculate momentum score based on recent match results.
        
        Uses research-based weighting where more recent matches
        have higher impact on momentum calculation.
        
        Args:
            recent_form: List of 1s (wins) and 0s (losses) in chronological order
            
        Returns:
            Momentum score between 0 and 1
        """
        if not recent_form:
            return 0.5
        
        # Weight recent matches more heavily (exponential decay)
        weights = [0.4, 0.3, 0.2, 0.1][:len(recent_form)]
        
        # Calculate weighted average
        weighted_sum = sum(result * weight for result, weight in zip(recent_form, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def export_processed_data(self, matches_df: pd.DataFrame, 
                            output_path: Union[str, Path]) -> None:
        """
        Export processed match data with additional features.
        
        Args:
            matches_df: DataFrame with processed match data
            output_path: Path for output CSV file
        """
        try:
            # Add derived features to each match
            processed_df = matches_df.copy()
            
            # Add match features
            processed_df['match_id'] = range(len(processed_df))
            processed_df['year'] = processed_df['Date'].dt.year
            processed_df['month'] = processed_df['Date'].dt.month
            
            # Export to CSV
            processed_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(processed_df)} processed matches to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting processed data: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = MatchProcessor()
    
    # Example: Process sample data
    sample_data = {
        'Date': ['2025-09-23', '2025-09-22', '2025-09-21'],
        'Tournament': ['US Open', 'Laver Cup', 'Davis Cup'],
        'Player1': ['Djokovic', 'Alcaraz', 'Sinner'],
        'Player2': ['Federer', 'Shelton', 'Fucsovics'],
        'Winner': ['Djokovic', 'Alcaraz', 'Sinner'],
        'Score': ['6-4, 6-2', '6-4, 6-4', '6-2, 6-4'],
        'Surface': ['hard', 'hard', 'hard']
    }
    
    df = pd.DataFrame(sample_data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Analyze player performance
    djokovic_analysis = processor.analyze_player_performance_trends(df, 'Djokovic', 5)
    print(f"\nDjokovic Recent Performance:")
    print(f"Win Rate: {djokovic_analysis.get('overall_win_rate', 0):.2f}")
    print(f"Recent Form: {djokovic_analysis.get('recent_form_5', [])}")
    
    # Generate prediction features
    features = processor.generate_match_prediction_features(df, 'Djokovic', 'Alcaraz', 'hard')
    print(f"\nPrediction Features:")
    print(f"Djokovic momentum: {features['player1_features']['momentum_score']:.3f}")
    print(f"Alcaraz momentum: {features['player2_features']['momentum_score']:.3f}")
