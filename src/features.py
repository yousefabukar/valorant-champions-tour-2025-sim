# src/features.py

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from src.config import TOURNAMENT_WEIGHTS, get_tournament_weight

class VCTFeatureEngine:
    """Feature engineering for VCT match prediction"""
    
    def __init__(self):
        self.team_stats = {}
        self.player_stats = {}
        self.match_history = []
        
    def load_data(self, data_file: str = "data/raw/vct_data.json"):
        """Load scraped VCT data"""
        print(f"📊 Loading data from {data_file}")
        
        try:
            with open(data_file, 'r') as f:
                self.raw_data = json.load(f)
            
            print(f"✅ Loaded {len(self.raw_data.get('recent_matches', []))} matches")
            print(f"✅ Loaded {len(self.raw_data.get('detailed_matches', []))} detailed matches")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def extract_team_features(self) -> pd.DataFrame:
        """Extract features for each team"""
        print("🔧 Extracting team features...")
        
        teams_data = []
        
        # Get teams from current championship points (updated data)
        from src.config import CURRENT_CHAMPIONSHIP_POINTS
        qualified_teams = CURRENT_CHAMPIONSHIP_POINTS
        
        for region, teams in qualified_teams.items():
            for team in teams:
                team_features = self._calculate_team_features(
                    team['name'], 
                    region,
                    team.get('points', 0),
                    team.get('status', 'unknown')
                )
                teams_data.append(team_features)
        
        # Add helper functions for getting team data
        from src.config import get_team_championship_points, get_team_qualification_status, calculate_champions_probability
        # Add teams from recent matches that aren't in championship points
        recent_matches = self.raw_data.get('recent_matches', [])
        seen_teams = {team['name'] for region_teams in qualified_teams.values() for team in region_teams}
        
        for match in recent_matches:
            for team_name in [match['team1'], match['team2']]:
                if team_name not in seen_teams:
                    region = self._guess_team_region(team_name, match.get('tournament', ''))
                    team_features = self._calculate_team_features(team_name, region, 0, 'unknown')
                    teams_data.append(team_features)
                    seen_teams.add(team_name)
        
        df = pd.DataFrame(teams_data)
        print(f"✅ Created features for {len(df)} teams")
        
        return df
    
    def _calculate_team_features(self, team_name: str, region: str, points: int, status: str) -> Dict:
        """Calculate comprehensive features for a team"""
        
        # Get real championship points from config
        from src.config import get_team_championship_points, get_team_qualification_status, calculate_champions_probability
        
        real_points = get_team_championship_points(team_name)
        real_status = get_team_qualification_status(team_name)
        
        # Use real data if available, otherwise use provided data
        final_points = real_points if real_points > 0 else points
        final_status = real_status if real_status != 'unknown' else status
        
        # Basic info (using real championship points)
        features = {
            'team_name': team_name,
            'region': region,
            'championship_points': final_points,
            'qualification_status': final_status
        }
        
        # Calculate performance metrics from matches
        team_matches = self._get_team_matches(team_name)
        
        if team_matches:
            # Win rate and recent form
            features.update(self._calculate_performance_metrics(team_name, team_matches))
            
            # Tournament performance
            features.update(self._calculate_tournament_performance(team_name, team_matches))
            
            # Map performance
            features.update(self._calculate_map_performance(team_name))
        else:
            # Default values for teams without match data
            features.update({
                'win_rate': 0.5,
                'recent_form': 0.5,
                'avg_rating': 1.0,
                'tournaments_played': 0,
                'weighted_performance': 1000.0
            })
        
        # Regional strength modifier
        regional_multipliers = {
            'americas': 1.1,
            'emea': 1.15,  # Boosted due to Heretics EWC win
            'pacific': 1.05,
            'china': 0.95
        }
        features['regional_strength'] = regional_multipliers.get(region, 1.0)
        
        # Champions qualification probability (using real config function)
        features['champions_probability'] = calculate_champions_probability(team_name)
        
        return features
    
    def _get_team_matches(self, team_name: str) -> List[Dict]:
        """Get all matches for a specific team"""
        team_matches = []
        
        # From recent matches
        for match in self.raw_data.get('recent_matches', []):
            if team_name in [match['team1'], match['team2']]:
                team_matches.append(match)
        
        # From detailed matches (with player stats)
        for match in self.raw_data.get('detailed_matches', []):
            teams = match.get('teams', {})
            if team_name in [teams.get('team1'), teams.get('team2')]:
                team_matches.append(match)
        
        return team_matches
    
    def _calculate_performance_metrics(self, team_name: str, matches: List[Dict]) -> Dict:
        """Calculate win rate, recent form, etc."""
        
        wins = 0
        total_games = 0
        recent_results = []  # Last 10 matches
        
        for match in matches:
            if not match.get('completed', False):
                continue
                
            total_games += 1
            
            # Determine if team won
            score = match.get('score', '0-0')
            if '-' in score:
                try:
                    score1, score2 = map(int, score.split('-'))
                    team_won = False
                    
                    if match['team1'] == team_name:
                        team_won = score1 > score2
                    elif match['team2'] == team_name:
                        team_won = score2 > score1
                    
                    if team_won:
                        wins += 1
                    
                    # Track recent form (1 for win, 0 for loss)
                    recent_results.append(1 if team_won else 0)
                    
                except:
                    continue
        
        # Calculate metrics
        win_rate = wins / max(1, total_games)
        recent_form = np.mean(recent_results[-10:]) if recent_results else 0.5
        
        return {
            'total_matches': total_games,
            'wins': wins,
            'losses': total_games - wins,
            'win_rate': round(win_rate, 3),
            'recent_form': round(recent_form, 3),
            'recent_matches_count': len(recent_results)
        }
    
    def _calculate_tournament_performance(self, team_name: str, matches: List[Dict]) -> Dict:
        """Calculate weighted tournament performance"""
        
        tournament_scores = {}
        weighted_total = 0
        weight_sum = 0
        
        for match in matches:
            tournament = match.get('tournament', '')
            if not tournament:
                continue
                
            weight = get_tournament_weight(tournament)
            
            # Simple performance score (could be enhanced with detailed stats)
            if match.get('completed'):
                score = match.get('score', '0-0')
                performance = self._score_to_performance(score, team_name, match)
            else:
                performance = 1000  # Default rating
            
            if tournament not in tournament_scores:
                tournament_scores[tournament] = []
            
            tournament_scores[tournament].append(performance)
            weighted_total += performance * weight
            weight_sum += weight
        
        weighted_performance = weighted_total / max(1, weight_sum)
        tournaments_played = len(tournament_scores)
        
        return {
            'tournaments_played': tournaments_played,
            'weighted_performance': round(weighted_performance, 1),
            'tournament_diversity': len(set(match.get('tournament', '') for match in matches))
        }
    
    def _score_to_performance(self, score: str, team_name: str, match: Dict) -> float:
        """Convert match score to performance rating"""
        if '-' not in score:
            return 1000.0
        
        try:
            score1, score2 = map(int, score.split('-'))
            
            # Determine team's score
            if match['team1'] == team_name:
                team_score, opponent_score = score1, score2
            else:
                team_score, opponent_score = score2, score1
            
            # Performance based on score (simplified)
            if team_score > opponent_score:
                # Win - rating based on dominance
                dominance = team_score / max(1, team_score + opponent_score)
                return 1000 + (dominance * 500)  # 1000-1500 range
            else:
                # Loss - rating based on competitiveness
                competitiveness = team_score / max(1, team_score + opponent_score)
                return 1000 - ((1 - competitiveness) * 500)  # 500-1000 range
                
        except:
            return 1000.0
    
    def _calculate_map_performance(self, team_name: str) -> Dict:
        """Calculate map-specific performance"""
        
        map_stats = {}
        
        # Analyze detailed matches for map performance
        for match in self.raw_data.get('detailed_matches', []):
            teams = match.get('teams', {})
            if team_name not in [teams.get('team1'), teams.get('team2')]:
                continue
            
            for map_data in match.get('maps', []):
                map_name = map_data.get('map_name', 'Unknown')
                score = map_data.get('score', '0-0')
                
                if map_name not in map_stats:
                    map_stats[map_name] = {'wins': 0, 'total': 0}
                
                map_stats[map_name]['total'] += 1
                
                # Check if team won this map
                if '-' in score:
                    try:
                        score1, score2 = map(int, score.split('-'))
                        if teams.get('team1') == team_name and score1 > score2:
                            map_stats[map_name]['wins'] += 1
                        elif teams.get('team2') == team_name and score2 > score1:
                            map_stats[map_name]['wins'] += 1
                    except:
                        continue
        
        # Calculate map win rates
        best_map_wr = 0
        worst_map_wr = 1
        maps_played = len(map_stats)
        
        for map_name, stats in map_stats.items():
            if stats['total'] > 0:
                wr = stats['wins'] / stats['total']
                best_map_wr = max(best_map_wr, wr)
                worst_map_wr = min(worst_map_wr, wr)
        
        return {
            'maps_played': maps_played,
            'best_map_winrate': round(best_map_wr, 3),
            'worst_map_winrate': round(worst_map_wr, 3),
            'map_pool_strength': round((best_map_wr + worst_map_wr) / 2, 3)
        }
    
    def _guess_team_region(self, team_name: str, tournament: str) -> str:
        """Guess team region from name/tournament"""
        
        # From tournament name
        if 'Americas' in tournament:
            return 'americas'
        elif 'EMEA' in tournament:
            return 'emea'  
        elif 'Pacific' in tournament:
            return 'pacific'
        elif 'China' in tournament:
            return 'china'
        
        # From team name patterns (simplified)
        americas_keywords = ['Gaming', 'Esports', 'G2', 'Sentinels', 'Cloud9', 'NRG']
        emea_keywords = ['Fnatic', 'Liquid', 'Heretics', 'Vitality', 'NAVI']
        pacific_keywords = ['Rex', 'T1', 'Paper', 'DRX', 'Gen.G', 'ZETA']
        china_keywords = ['Gaming', 'EDward', 'Bilibili', 'TYLOO']
        
        team_lower = team_name.lower()
        
        if any(keyword.lower() in team_lower for keyword in americas_keywords):
            return 'americas'
        elif any(keyword.lower() in team_lower for keyword in emea_keywords):
            return 'emea'
        elif any(keyword.lower() in team_lower for keyword in pacific_keywords):
            return 'pacific'
        elif any(keyword.lower() in team_lower for keyword in china_keywords):
            return 'china'
        
        return 'unknown'
    
    def create_match_features(self, team1: str, team2: str) -> Dict:
        """Create features for a specific matchup"""
        
        team_df = self.extract_team_features()
        
        # Get team features
        team1_features = team_df[team_df['team_name'] == team1].iloc[0].to_dict() if len(team_df[team_df['team_name'] == team1]) > 0 else {}
        team2_features = team_df[team_df['team_name'] == team2].iloc[0].to_dict() if len(team_df[team_df['team_name'] == team2]) > 0 else {}
        
        if not team1_features or not team2_features:
            print(f"⚠️  Missing data for {team1} vs {team2}")
            return {}
        
        # Head-to-head features
        h2h_stats = self._calculate_head_to_head(team1, team2)
        
        # Matchup features
        matchup_features = {
            'team1_name': team1,
            'team2_name': team2,
            
            # Performance difference
            'win_rate_diff': team1_features.get('win_rate', 0.5) - team2_features.get('win_rate', 0.5),
            'form_diff': team1_features.get('recent_form', 0.5) - team2_features.get('recent_form', 0.5),
            'points_diff': team1_features.get('championship_points', 0) - team2_features.get('championship_points', 0),
            'performance_diff': team1_features.get('weighted_performance', 1000) - team2_features.get('weighted_performance', 1000),
            
            # Regional matchup
            'same_region': team1_features.get('region') == team2_features.get('region'),
            'region_strength_diff': team1_features.get('regional_strength', 1.0) - team2_features.get('regional_strength', 1.0),
            
            # Experience
            'experience_diff': team1_features.get('tournaments_played', 0) - team2_features.get('tournaments_played', 0),
            
            # Head-to-head
            'h2h_wins': h2h_stats.get('team1_wins', 0),
            'h2h_losses': h2h_stats.get('team2_wins', 0),
            'h2h_total': h2h_stats.get('total_matches', 0)
        }
        
        return matchup_features
    
    def _calculate_head_to_head(self, team1: str, team2: str) -> Dict:
        """Calculate head-to-head record between two teams"""
        
        h2h_matches = []
        
        # Look through all matches for these teams facing each other
        for match in self.raw_data.get('recent_matches', []) + self.raw_data.get('detailed_matches', []):
            match_teams = [match.get('team1'), match.get('team2')]
            if not match_teams[0] or not match_teams[1]:
                # Try getting from teams dict for detailed matches
                teams = match.get('teams', {})
                match_teams = [teams.get('team1'), teams.get('team2')]
            
            if team1 in match_teams and team2 in match_teams:
                h2h_matches.append(match)
        
        team1_wins = 0
        team2_wins = 0
        
        for match in h2h_matches:
            if not match.get('completed'):
                continue
                
            score = match.get('score', '0-0')
            if '-' in score:
                try:
                    score1, score2 = map(int, score.split('-'))
                    
                    # Determine which team is which
                    if match.get('team1') == team1 or match.get('teams', {}).get('team1') == team1:
                        if score1 > score2:
                            team1_wins += 1
                        else:
                            team2_wins += 1
                    else:
                        if score2 > score1:
                            team1_wins += 1
                        else:
                            team2_wins += 1
                except:
                    continue
        
        return {
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'total_matches': team1_wins + team2_wins
        }
    
    def save_features(self, df: pd.DataFrame, filename: str = "team_features.csv"):
        """Save features to CSV"""
        output_path = f"data/processed/{filename}"
        df.to_csv(output_path, index=False)
        print(f"💾 Features saved to {output_path}")


def main():
    """Test feature engineering"""
    print("🔧 Starting VCT Feature Engineering...")
    print("=" * 50)
    
    # Initialize feature engine
    feature_engine = VCTFeatureEngine()
    
    # Load data
    if not feature_engine.load_data():
        print("❌ Failed to load data")
        return
    
    # Extract team features
    team_features_df = feature_engine.extract_team_features()
    
    # Display team features
    print(f"\n📊 Team Features Summary:")
    print(team_features_df[['team_name', 'region', 'championship_points', 'win_rate', 'champions_probability']].to_string(index=False))
    
    # Create sample matchup features
    print(f"\n🥊 Sample Matchup Analysis:")
    
    # Test with known teams
    sample_matchups = [
        ('G2 Esports', 'Sentinels'),
        ('Paper Rex', 'T1'),
        ('Fnatic', 'Team Heretics')
    ]
    
    for team1, team2 in sample_matchups:
        print(f"\n{team1} vs {team2}:")
        matchup_features = feature_engine.create_match_features(team1, team2)
        
        if matchup_features:
            print(f"  Win Rate Diff: {matchup_features.get('win_rate_diff', 0):.3f}")
            print(f"  Form Diff: {matchup_features.get('form_diff', 0):.3f}")
            print(f"  Points Diff: {matchup_features.get('points_diff', 0)}")
            print(f"  H2H Record: {matchup_features.get('h2h_wins', 0)}-{matchup_features.get('h2h_losses', 0)}")
        else:
            print("  ⚠️  Insufficient data for analysis")
    
    # Save processed features
    feature_engine.save_features(team_features_df)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"🚀 Ready for ML model training!")

if __name__ == "__main__":
    main()