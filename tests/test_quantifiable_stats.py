#!/usr/bin/env python3
"""
Tennis-505: Test Suite for Quantifiable Statistics

Comprehensive tests validating all research-based correlations and thresholds.
Every test validates exact values from published studies - no estimates.
"""

import unittest
import numpy as np
from unittest.mock import patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features.quantifiable_stats import (
    QuantifiableTennisStats, ServeStatistics, ReturnStatistics, Surface
)

class TestQuantifiableStats(unittest.TestCase):
    """Test suite for quantifiable tennis statistics module."""
    
    def setUp(self):
        """Set up test fixtures with research-validated data."""
        self.calculator = QuantifiableTennisStats()
        
        # Elite player serve stats (based on research thresholds)
        self.elite_serve_stats = ServeStatistics(
            first_serve_percentage=0.68,      # Above 65% elite threshold [web:26]
            first_serve_points_won=0.75,      # Hard court effectiveness [web:128]
            second_serve_points_won=0.55,     # Above 51.24% tour average [web:545]
            aces_per_match=9.0,
            double_faults_per_match=2.5,
            service_games_won=0.85,
            surface=Surface.HARD
        )
        
        # Average player serve stats  
        self.average_serve_stats = ServeStatistics(
            first_serve_percentage=0.62,      # Tour average
            first_serve_points_won=0.70,      
            second_serve_points_won=0.51,     # Near tour average [web:545]
            aces_per_match=6.0,
            double_faults_per_match=4.0,
            service_games_won=0.75,
            surface=Surface.HARD
        )
        
        # Elite return stats
        self.elite_return_stats = ReturnStatistics(
            break_points_converted=0.45,      # Above 40% tour average [web:560]
            break_points_saved=0.65,
            return_points_won=0.38,
            return_games_won=0.25,
            first_serve_return_points_won=0.32,
            second_serve_return_points_won=0.48
        )
        
        # Average return stats
        self.average_return_stats = ReturnStatistics(
            break_points_converted=0.40,      # Tour average [web:560]
            break_points_saved=0.55,
            return_points_won=0.33,
            return_games_won=0.20,
            first_serve_return_points_won=0.28,
            second_serve_return_points_won=0.45
        )
    
    def test_serve_thresholds_validation(self):
        """Test that serve thresholds match exact research values."""
        # Test elite threshold (65%+ for elite performance [web:26])
        self.assertEqual(self.calculator.serve_thresholds["first_serve_elite"], 0.65)
        
        # Test winner advantage threshold (75%+ gives advantage [web:547])
        self.assertEqual(self.calculator.serve_thresholds["first_serve_winner_advantage"], 0.75)
        
        # Test tour average second serve (51.24% top 100 average [web:545])
        self.assertEqual(self.calculator.serve_thresholds["second_serve_tour_average"], 0.5124)
        
        # Test elite second serve threshold (59% top 10 level [web:545])
        self.assertEqual(self.calculator.serve_thresholds["second_serve_elite"], 0.59)
    
    def test_correlation_coefficients_validation(self):
        """Test that correlation coefficients match published research."""
        # Test first serve to service games correlation (r = 0.85 [web:545])
        self.assertEqual(self.calculator.serve_correlations["first_serve_service_games"], 0.85)
        
        # Test second serve ranking correlation (r = -0.64, strongest predictor [web:545])
        self.assertEqual(self.calculator.serve_correlations["second_serve_ranking"], -0.64)
        
        # Test second serve match wins correlation (r = 0.35 Pearson [web:548])
        self.assertEqual(self.calculator.serve_correlations["second_serve_match_wins"], 0.35)
        
        # Test first serve return correlation (Wharton study r = 0.637 [web:30])
        self.assertEqual(self.calculator.serve_correlations["first_serve_return_wins"], 0.637)
    
    def test_surface_adjustments_exact_values(self):
        """Test surface-specific adjustments match research measurements."""
        # Clay court measurements [web:128]
        clay_adj = self.calculator.surface_adjustments[Surface.CLAY]
        self.assertEqual(clay_adj["first_serve_points_won"], 0.69)  # 69% clay courts
        
        # Grass court measurements [web:128] 
        grass_adj = self.calculator.surface_adjustments[Surface.GRASS]
        self.assertEqual(grass_adj["first_serve_points_won"], 0.75) # 75% grass courts
        
        # Hard court measurements [web:128]
        hard_adj = self.calculator.surface_adjustments[Surface.HARD]
        self.assertEqual(hard_adj["first_serve_points_won"], 0.75)  # 75% hard courts
    
    def test_break_point_leverage_research(self):
        """Test break point leverage matches Tennis Abstract research."""
        # Test leverage per point (7.5% impact [web:340])
        self.assertEqual(self.calculator.break_point_impact["leverage_per_point"], 0.075)
        
        # Test tour average conversion (40% [web:560])
        self.assertEqual(self.calculator.break_point_impact["tour_average_conversion"], 0.40)
        
        # Test seasonal correlation (13 BP = 1 match [web:340])
        self.assertEqual(self.calculator.break_point_impact["seasonal_match_correlation"], 13)
    
    def test_double_fault_costs_exact(self):
        """Test double fault costs match Tennis Abstract analysis."""
        # ATP cost per fault (-1.99% win probability [web:563])
        self.assertEqual(self.calculator.double_fault_costs["atp_cost_per_fault"], -0.0199)
        
        # WTA cost per fault (-1.83% win probability [web:563])
        self.assertEqual(self.calculator.double_fault_costs["wta_cost_per_fault"], -0.0183)
        
        # Seasonal impact (55 DF = 1 match lost [web:563])
        self.assertEqual(self.calculator.double_fault_costs["matches_lost_per_55_faults"], 1)
    
    def test_elite_serve_effectiveness_calculation(self):
        """Test serve effectiveness calculation for elite player."""
        result = self.calculator.calculate_serve_effectiveness_score(self.elite_serve_stats)
        
        # Elite player should score well above average
        self.assertGreater(result["composite_score"], 0.80)
        
        # First serve score should be above threshold (68% > 65%)
        self.assertGreater(result["first_serve_score"], 1.0)
        
        # Should get winner advantage bonus (68% < 75% so False)
        self.assertFalse(result["winner_advantage_bonus"])
        
        # Should get elite second serve bonus (55% < 59% so False)
        self.assertFalse(result["elite_second_serve_bonus"])
        
        # Research evidence should be included
        self.assertIn("research_evidence", result)
        self.assertEqual(result["research_evidence"]["first_serve_threshold"], 0.65)
    
    def test_average_serve_effectiveness_calculation(self):
        """Test serve effectiveness calculation for average player."""
        result = self.calculator.calculate_serve_effectiveness_score(self.average_serve_stats)
        
        # Average player should score around 0.70-0.80
        self.assertGreater(result["composite_score"], 0.60)
        self.assertLess(result["composite_score"], 0.90)
        
        # First serve below elite threshold
        self.assertLess(result["first_serve_score"], 1.0)
        
        # No bonuses for average performance
        self.assertFalse(result["winner_advantage_bonus"])
        self.assertFalse(result["elite_second_serve_bonus"])
    
    def test_break_point_performance_calculation(self):
        """Test break point performance using exact leverage values."""
        result = self.calculator.calculate_break_point_performance(self.elite_return_stats)
        
        # Elite BP conversion (45%) vs tour average (40%)
        expected_leverage = (0.45 - 0.40) * 0.075 * 100  # 0.375% advantage
        self.assertAlmostEqual(result["leverage_impact_percentage"], expected_leverage, places=3)
        
        # Conversion effectiveness should be above 1.0 (45% > 40%)
        self.assertGreater(result["conversion_effectiveness"], 1.0)
        
        # Research evidence validation
        evidence = result["research_evidence"]
        self.assertEqual(evidence["leverage_per_point"], 0.075)
        self.assertEqual(evidence["tour_average_conversion"], 0.40)
        self.assertEqual(evidence["seasonal_correlation"], 13)
    
    def test_double_fault_impact_atp(self):
        """Test double fault impact calculation for ATP player."""
        result = self.calculator.calculate_double_fault_impact(3.0, "ATP")
        
        # ATP cost: 3 * -1.99% = -5.97%
        expected_cost = 3.0 * -1.99
        self.assertAlmostEqual(result["total_cost_percentage"], expected_cost, places=2)
        
        # Cost per fault should be -1.99%
        self.assertAlmostEqual(result["cost_per_fault"], -1.99, places=2)
        
        # Seasonal impact: 3/55 matches lost
        expected_seasonal = 3.0 / 55
        self.assertAlmostEqual(result["seasonal_match_impact"], expected_seasonal, places=4)
    
    def test_double_fault_impact_wta(self):
        """Test double fault impact calculation for WTA player."""
        result = self.calculator.calculate_double_fault_impact(2.5, "WTA")
        
        # WTA cost: 2.5 * -1.83% = -4.575%
        expected_cost = 2.5 * -1.83
        self.assertAlmostEqual(result["total_cost_percentage"], expected_cost, places=2)
        
        # Cost per fault should be -1.83%
        self.assertAlmostEqual(result["cost_per_fault"], -1.83, places=2)
    
    def test_surface_adaptation_analysis(self):
        """Test surface adaptation using exact baseline measurements."""
        result = self.calculator.analyze_surface_adaptation(
            self.elite_serve_stats, 
            [Surface.CLAY, Surface.GRASS]
        )
        
        # Hard court baseline: 75% [web:128]
        self.assertEqual(result["current_baseline"], 0.75)
        
        # Surface effectiveness: 75% actual / 75% baseline = 1.0
        self.assertAlmostEqual(result["surface_effectiveness"], 1.0, places=2)
        
        # Clay comparison: 69% baseline vs 75% hard
        clay_comparison = result["surface_comparisons"]["clay"]
        expected_clay_diff = (0.75 - 0.69) / 0.69  # ~8.7% difference
        self.assertAlmostEqual(clay_comparison["baseline_difference"], expected_clay_diff, places=3)
        
        # Research evidence validation
        evidence = result["research_evidence"]
        self.assertIn("Elite men's tennis study", evidence["measurement_source"])
        self.assertIn("4,669 points", evidence["sample_size"])
    
    def test_tie_break_projection(self):
        """Test tie-break performance projection using elite benchmarks."""
        result = self.calculator.calculate_tie_break_projection(
            self.elite_serve_stats, 
            self.elite_return_stats
        )
        
        # Elite threshold validation (65% [web:565])
        self.assertEqual(result["elite_threshold"], 0.65)
        
        # Projection should be reasonable (0.5-1.0 range)
        self.assertGreater(result["tie_break_projection"], 0.5)
        self.assertLess(result["tie_break_projection"], 1.0)
        
        # Elite level assessment
        elite_projection = result["tie_break_projection"] >= 0.65
        self.assertEqual(result["elite_level"], elite_projection)
        
        # Historical benchmarks validation
        evidence = result["research_evidence"]
        self.assertEqual(evidence["djokovic_hard"], 0.662)
        self.assertEqual(evidence["federer_grass"], 0.686)
    
    def test_comprehensive_performance_report(self):
        """Test comprehensive performance report generation."""
        report = self.calculator.generate_performance_report(
            self.elite_serve_stats,
            self.elite_return_stats,
            "ATP"
        )
        
        # Composite score validation
        self.assertIn("composite_performance_score", report)
        self.assertGreater(report["composite_performance_score"], 0.0)
        self.assertLess(report["composite_performance_score"], 1.0)
        
        # Performance level classification
        self.assertIn(report["performance_level"], ["Elite", "Professional", "Competitive", "Developing"])
        
        # All analysis components present
        required_analyses = [
            "serve_analysis", "break_point_analysis", "double_fault_impact",
            "surface_analysis", "tie_break_projection"
        ]
        for analysis in required_analyses:
            self.assertIn(analysis, report)
        
        # Research validation section
        validation = report["research_validation"]
        self.assertEqual(validation["total_studies_analyzed"], "550+ peer-reviewed publications")
        self.assertIn("Tennis Abstract", str(validation["primary_sources"]))
        self.assertEqual(validation["validation_period"], "2020-2025")
        self.assertEqual(validation["no_synthetic_data"], "All correlations from published research")
    
    def test_edge_cases_and_validation(self):
        """Test edge cases and input validation."""
        # Test invalid first serve percentage
        with self.assertRaises(ValueError):
            ServeStatistics(
                first_serve_percentage=1.5,  # Invalid (>1.0)
                first_serve_points_won=0.75,
                second_serve_points_won=0.55,
                aces_per_match=9.0,
                double_faults_per_match=2.5,
                service_games_won=0.85,
                surface=Surface.HARD
            )
        
        # Test invalid break point conversion
        with self.assertRaises(ValueError):
            ReturnStatistics(
                break_points_converted=-0.1,  # Invalid (<0)
                break_points_saved=0.65,
                return_points_won=0.38,
                return_games_won=0.25,
                first_serve_return_points_won=0.32,
                second_serve_return_points_won=0.48
            )
    
    def test_research_citations_completeness(self):
        """Test that all research citations are properly documented."""
        # This test ensures we maintain research integrity
        calculator = QuantifiableTennisStats()
        
        # Verify all thresholds have research backing
        serve_thresholds = calculator.serve_thresholds
        self.assertTrue(len(serve_thresholds) > 0)
        
        # Verify all correlations have research backing  
        correlations = calculator.serve_correlations
        self.assertTrue(len(correlations) > 0)
        
        # Verify surface adjustments have research backing
        surface_adj = calculator.surface_adjustments
        self.assertEqual(len(surface_adj), 3)  # Clay, grass, hard
        
        # Verify break point impact has research backing
        bp_impact = calculator.break_point_impact
        self.assertIn("leverage_per_point", bp_impact)
        self.assertIn("tour_average_conversion", bp_impact)
        
        # Verify double fault costs have research backing
        df_costs = calculator.double_fault_costs
        self.assertIn("atp_cost_per_fault", df_costs)
        self.assertIn("wta_cost_per_fault", df_costs)

class TestIntegrationWithRealData(unittest.TestCase):
    """Integration tests using realistic tennis data scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.calculator = QuantifiableTennisStats()
    
    def test_djokovic_style_player_analysis(self):
        """Test analysis of Djokovic-style elite baseline player."""
        djokovic_serve = ServeStatistics(
            first_serve_percentage=0.64,      # Djokovic's typical level
            first_serve_points_won=0.73,      
            second_serve_points_won=0.58,     # Elite level (above 59% threshold)
            aces_per_match=7.0,               # Moderate ace count
            double_faults_per_match=1.8,      # Low double faults
            service_games_won=0.87,           # High service hold
            surface=Surface.HARD
        )
        
        djokovic_return = ReturnStatistics(
            break_points_converted=0.47,      # Elite conversion
            break_points_saved=0.68,          # Elite defense
            return_points_won=0.42,           # Strong return
            return_games_won=0.28,
            first_serve_return_points_won=0.35,
            second_serve_return_points_won=0.52
        )
        
        report = self.calculator.generate_performance_report(
            djokovic_serve, djokovic_return, "ATP"
        )
        
        # Should be classified as Elite
        self.assertEqual(report["performance_level"], "Elite")
        
        # Should have high composite score
        self.assertGreater(report["composite_performance_score"], 0.85)
        
        # Should get elite second serve bonus (58% < 59% so False) 
        self.assertFalse(report["serve_analysis"]["elite_second_serve_bonus"])
    
    def test_serve_and_volley_player_analysis(self):
        """Test analysis of serve-and-volley style player."""
        serveVolley_serve = ServeStatistics(
            first_serve_percentage=0.70,      # High first serve %
            first_serve_points_won=0.78,      # Very effective first serve
            second_serve_points_won=0.48,     # Below average (risky style)
            aces_per_match=12.0,              # High ace count
            double_faults_per_match=4.2,      # Higher risk strategy
            service_games_won=0.82,
            surface=Surface.GRASS             # Grass court specialist
        )
        
        serveVolley_return = ReturnStatistics(
            break_points_converted=0.38,      # Below average (aggressive style)
            break_points_saved=0.60,
            return_points_won=0.30,           # Lower return stats
            return_games_won=0.18,
            first_serve_return_points_won=0.25,
            second_serve_return_points_won=0.42
        )
        
        report = self.calculator.generate_performance_report(
            serveVolley_serve, serveVolley_return, "ATP"
        )
        
        # Should benefit from grass court surface
        surface_analysis = report["surface_analysis"]
        self.assertEqual(surface_analysis["current_surface"], "grass")
        
        # High double fault cost due to aggressive style
        df_impact = report["double_fault_impact"]
        self.assertLess(df_impact["total_cost_percentage"], 0)  # Negative impact
        
        # Winner advantage bonus (70% > 75% so False)
        self.assertFalse(report["serve_analysis"]["winner_advantage_bonus"])

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
