# src/model.py

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    from src.features import VCTFeatureEngine
    from src.config import MODEL_CONFIG, get_tournament_weight, calculate_champions_probability
except ImportError:
    from .features import VCTFeatureEngine
    from .config import MODEL_CONFIG, get_tournament_weight, calculate_champions_probability

class VCTPredictor:
    """ML model for predicting VCT match outcomes and tournament probabilities"""
    
    def __init__(self):
        self.match_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_engine = VCTFeatureEngine()
        self.feature_importance = {}
        self.model_metrics = {}
        
    def load_and_prepare_data(self):
        """Load data and prepare features for training"""
        print("📊 Loading and preparing training data...")
        
        # Load scraped data
        if not self.feature_engine.load_data():
            print("❌ Failed to load data")
            return False
        
        # Extract team features
        self.team_features = self.feature_engine.extract_team_features()
        print(f"✅ Loaded features for {len(self.team_features)} teams")
        
        # Create training data from recent matches
        self.training_data = self._create_training_data()
        print(f"✅ Created {len(self.training_data)} training examples")
        
        return len(self.training_data) > 0
    
    def _create_training_data(self) -> List[Dict]:
        """Create training examples from completed matches"""
        training_examples = []
        
        # Get recent matches from feature engine
        recent_matches = self.feature_engine.raw_data.get('recent_matches', [])
        detailed_matches = self.feature_engine.raw_data.get('detailed_matches', [])
        
        # Process completed matches
        for match in recent_matches + detailed_matches:
            if not match.get('completed', False):
                continue
                
            # Get teams
            team1 = match.get('team1')
            team2 = match.get('team2')
            if not team1 or not team2:
                # Try detailed match format
                teams = match.get('teams', {})
                team1 = teams.get('team1')
                team2 = teams.get('team2')
                
            if not team1 or not team2:
                continue
            
            # Determine winner from score
            winner = self._determine_winner(match, team1, team2)
            if winner is None:
                continue
            
            # Create match features
            match_features = self.feature_engine.create_match_features(team1, team2)
            if not match_features:
                continue
            
            # Add tournament info
            tournament = match.get('tournament', '')
            match_features['tournament_weight'] = get_tournament_weight(tournament)
            match_features['match_date'] = match.get('date', '')
            
            # Add target variable
            match_features['winner'] = 1 if winner == team1 else 0  # 1 if team1 wins, 0 if team2 wins
            match_features['winner_name'] = winner
            
            training_examples.append(match_features)
        
        return training_examples
    
    def _determine_winner(self, match: Dict, team1: str, team2: str) -> Optional[str]:
        """Determine match winner from score"""
        score = match.get('score', '0-0')
        
        if '-' not in score:
            return None
            
        try:
            score1, score2 = map(int, score.split('-'))
            
            if score1 > score2:
                return team1
            elif score2 > score1:
                return team2
            else:
                return None  # Tie (shouldn't happen in VALORANT)
        except:
            return None
    
    def train_match_predictor(self):
        """Train the match outcome prediction model"""
        print("🤖 Training match prediction model...")
        
        if not self.training_data:
            print("❌ No training data available")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(self.training_data)
        print(f"📊 Training data shape: {df.shape}")
        
        # Feature selection - only use numeric features for ML
        feature_columns = [
            'win_rate_diff', 'form_diff', 'points_diff', 'performance_diff',
            'region_strength_diff', 'experience_diff', 'tournament_weight',
            'h2h_wins', 'h2h_losses', 'h2h_total'
        ]
        
        # Add binary features
        df['same_region_int'] = df['same_region'].astype(int)
        feature_columns.append('same_region_int')
        
        # Prepare features and target
        X = df[feature_columns].fillna(0)  # Fill any missing values
        y = df['winner']
        
        print(f"✅ Feature matrix: {X.shape}")
        print(f"✅ Features used: {feature_columns}")
        
        # Split data
        test_size = MODEL_CONFIG.get('test_size', 0.2)
        random_state = MODEL_CONFIG.get('random_state', 42)
        
        if len(X) < 10:
            print("⚠️  Very limited training data - using simple validation")
            X_train, X_test = X, X  # Use all data for both train and test
            y_train, y_test = y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.match_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.match_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.match_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
        
        # Feature importance
        self.feature_importance = dict(zip(feature_columns, self.match_model.feature_importances_))
        
        # Store metrics
        self.model_metrics = {
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': feature_columns,
            'trained_at': datetime.now().isoformat()
        }
        
        # Cross-validation if enough data
        if len(X) >= 10:
            cv_scores = cross_val_score(self.match_model, X_train_scaled, y_train, cv=min(5, len(X_train)))
            self.model_metrics['cv_accuracy_mean'] = cv_scores.mean()
            self.model_metrics['cv_accuracy_std'] = cv_scores.std()
            print(f"✅ Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return True
    
    def predict_match(self, team1: str, team2: str) -> Dict:
        """Predict outcome of a specific match"""
        if not self.match_model:
            return {'error': 'Model not trained'}
        
        # Create match features
        match_features = self.feature_engine.create_match_features(team1, team2)
        if not match_features:
            return {'error': f'Could not create features for {team1} vs {team2}'}
        
        # Prepare features for prediction
        feature_columns = list(self.feature_importance.keys())
        
        # Convert to numeric format
        features_dict = {}
        for col in feature_columns:
            if col == 'same_region_int':
                features_dict[col] = int(match_features.get('same_region', False))
            else:
                features_dict[col] = match_features.get(col.replace('_int', ''), 0)
        
        # Create feature array
        X = np.array([list(features_dict.values())])
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.match_model.predict(X_scaled)[0]
        probabilities = self.match_model.predict_proba(X_scaled)[0]
        
        # Format results
        team1_win_prob = probabilities[1] if len(probabilities) > 1 else 0.5
        team2_win_prob = probabilities[0] if len(probabilities) > 1 else 0.5
        
        predicted_winner = team1 if prediction == 1 else team2
        confidence = max(team1_win_prob, team2_win_prob)
        
        return {
            'team1': team1,
            'team2': team2,
            'predicted_winner': predicted_winner,
            'team1_win_probability': round(team1_win_prob, 3),
            'team2_win_probability': round(team2_win_prob, 3),
            'confidence': round(confidence, 3),
            'features_used': features_dict
        }
    
    def predict_champions_probabilities(self) -> Dict:
        """Predict Champions qualification probabilities for all teams"""
        print("🏆 Calculating Champions qualification probabilities...")
        
        predictions = {
            'americas': [],
            'emea': [],
            'pacific': [],
            'china': []
        }
        
        # Use team features to get all teams
        for _, team_row in self.team_features.iterrows():
            team_name = team_row['team_name']
            region = team_row['region']
            
            if region not in predictions:
                continue
            
            # Get probability from config (based on championship points and status)
            champs_prob = calculate_champions_probability(team_name)
            
            # Enhance with ML-based performance adjustment
            performance_modifier = self._calculate_performance_modifier(team_row)
            final_prob = min(0.98, champs_prob * performance_modifier)
            
            team_prediction = {
                'team': team_name,
                'base_probability': champs_prob,
                'performance_modifier': performance_modifier,
                'final_probability': round(final_prob, 3),
                'championship_points': team_row.get('championship_points', 0),
                'status': team_row.get('qualification_status', 'unknown')
            }
            
            predictions[region].append(team_prediction)
        
        # Sort by probability
        for region in predictions:
            predictions[region].sort(key=lambda x: x['final_probability'], reverse=True)
        
        return predictions
    
    def _calculate_performance_modifier(self, team_row: pd.Series) -> float:
        """Calculate performance-based modifier for Champions probability"""
        
        # Base modifier
        modifier = 1.0
        
        # Win rate adjustment
        win_rate = team_row.get('win_rate', 0.5)
        if win_rate > 0.7:
            modifier *= 1.1
        elif win_rate < 0.4:
            modifier *= 0.9
        
        # Recent form adjustment
        recent_form = team_row.get('recent_form', 0.5)
        if recent_form > 0.7:
            modifier *= 1.05
        elif recent_form < 0.4:
            modifier *= 0.95
        
        # Tournament activity
        tournaments_played = team_row.get('tournaments_played', 0)
        if tournaments_played > 3:
            modifier *= 1.02
        elif tournaments_played < 2:
            modifier *= 0.98
        
        return modifier
    
    def simulate_tournament_bracket(self, num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation of Champions tournament"""
        print(f"🎲 Running {num_simulations} tournament simulations...")
        
        if not self.match_model:
            print("❌ Match prediction model not available")
            return {}
        
        # Get qualified/likely teams (simplified - top 4 from each region)
        champs_probs = self.predict_champions_probabilities()
        
        bracket_teams = []
        for region, teams in champs_probs.items():
            # Take top 4 teams by probability
            region_teams = sorted(teams, key=lambda x: x['final_probability'], reverse=True)[:4]
            for team in region_teams:
                bracket_teams.append({
                    'name': team['team'],
                    'region': region,
                    'seed': len(bracket_teams) + 1,
                    'qualification_prob': team['final_probability']
                })
        
        if len(bracket_teams) < 16:
            print(f"⚠️  Only {len(bracket_teams)} teams available for simulation")
            return {}
        
        # Run simulations
        winner_counts = {}
        regional_performance = {'americas': 0, 'emea': 0, 'pacific': 0, 'china': 0}
        
        for sim in range(num_simulations):
            tournament_winner = self._simulate_single_tournament(bracket_teams[:16])
            
            if tournament_winner:
                winner_name = tournament_winner['name']
                winner_region = tournament_winner['region']
                
                winner_counts[winner_name] = winner_counts.get(winner_name, 0) + 1
                regional_performance[winner_region] += 1
        
        # Calculate results
        total_sims = max(1, sum(winner_counts.values()))
        
        winner_probabilities = [
            {
                'team': team,
                'wins': count,
                'probability': round(count / total_sims, 3)
            }
            for team, count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        regional_stats = [
            {
                'region': region,
                'wins': count,
                'win_percentage': round(count / total_sims * 100, 1)
            }
            for region, count in sorted(regional_performance.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            'simulations_run': num_simulations,
            'winner_probabilities': winner_probabilities,
            'regional_performance': regional_stats,
            'tournament_bracket': bracket_teams
        }
    
    def _simulate_single_tournament(self, teams: List[Dict]) -> Optional[Dict]:
        """Simulate a single tournament bracket"""
        
        # Simplified single-elimination bracket
        remaining_teams = teams.copy()
        
        while len(remaining_teams) > 1:
            next_round = []
            
            # Pair teams and simulate matches
            for i in range(0, len(remaining_teams), 2):
                if i + 1 < len(remaining_teams):
                    team1 = remaining_teams[i]
                    team2 = remaining_teams[i + 1]
                    
                    # Simulate match
                    match_result = self.predict_match(team1['name'], team2['name'])
                    
                    if match_result.get('predicted_winner') == team1['name']:
                        next_round.append(team1)
                    else:
                        next_round.append(team2)
                else:
                    # Bye
                    next_round.append(remaining_teams[i])
            
            remaining_teams = next_round
        
        return remaining_teams[0] if remaining_teams else None
    
    def save_model(self, model_dir: str = "models/"):
        """Save trained model and metrics"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        if self.match_model:
            with open(f"{model_dir}match_predictor.pkl", 'wb') as f:
                pickle.dump(self.match_model, f)
            
            with open(f"{model_dir}scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save feature importance
        with open(f"{model_dir}feature_importance.json", 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        # Save metrics
        with open(f"{model_dir}model_metrics.json", 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
        
        print(f"💾 Model saved to {model_dir}")
    
    def load_model(self, model_dir: str = "models/"):
        """Load saved model"""
        try:
            with open(f"{model_dir}match_predictor.pkl", 'rb') as f:
                self.match_model = pickle.load(f)
            
            with open(f"{model_dir}scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(f"{model_dir}feature_importance.json", 'r') as f:
                self.feature_importance = json.load(f)
            
            with open(f"{model_dir}model_metrics.json", 'r') as f:
                self.model_metrics = json.load(f)
            
            print(f"✅ Model loaded from {model_dir}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def main():
    """Train and test the VCT prediction model"""
    print("🤖 VCT Machine Learning Model Training")
    print("=" * 50)
    
    # Initialize predictor
    predictor = VCTPredictor()
    
    # Load and prepare data
    if not predictor.load_and_prepare_data():
        print("❌ Failed to load training data")
        return
    
    # Train match predictor
    if not predictor.train_match_predictor():
        print("❌ Failed to train model")
        return
    
    # Show feature importance
    print(f"\n📊 Feature Importance:")
    for feature, importance in sorted(predictor.feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.3f}")
    
    # Test predictions
    print(f"\n🥊 Sample Match Predictions:")
    test_matchups = [
        ('G2 Esports', 'Sentinels'),
        ('Paper Rex', 'T1'),
        ('Fnatic', 'Team Heretics'),
        ('EDward Gaming', 'Bilibili Gaming')
    ]
    
    for team1, team2 in test_matchups:
        prediction = predictor.predict_match(team1, team2)
        
        if 'error' not in prediction:
            print(f"\n{team1} vs {team2}:")
            print(f"  🏆 Predicted Winner: {prediction['predicted_winner']}")
            print(f"  📊 Probabilities: {prediction['team1_win_probability']:.1%} vs {prediction['team2_win_probability']:.1%}")
            print(f"  🎯 Confidence: {prediction['confidence']:.1%}")
        else:
            print(f"\n{team1} vs {team2}: {prediction['error']}")
    
    # Champions probabilities
    print(f"\n🏆 Champions 2025 Qualification Probabilities:")
    champs_probs = predictor.predict_champions_probabilities()
    
    for region, teams in champs_probs.items():
        print(f"\n{region.upper()}:")
        for i, team in enumerate(teams[:6]):  # Top 6 per region
            status_emoji = {'qualified': '✅', 'likely': '🔥', 'contender': '⚡'}.get(team['status'], '❓')
            print(f"  {i+1}. {status_emoji} {team['team']}: {team['final_probability']:.1%} ({team['championship_points']} pts)")
    
    # Tournament simulation
    print(f"\n🎲 Tournament Simulation:")
    simulation = predictor.simulate_tournament_bracket(num_simulations=100)
    
    if simulation:
        print(f"📊 Winner Probabilities (top 8):")
        for i, result in enumerate(simulation['winner_probabilities'][:8]):
            print(f"  {i+1}. {result['team']}: {result['probability']:.1%} ({result['wins']} wins)")
        
        print(f"\n🌍 Regional Performance:")
        for region_stat in simulation['regional_performance']:
            print(f"  {region_stat['region'].upper()}: {region_stat['win_percentage']}% win rate")
    
    # Save model
    predictor.save_model()
    
    print(f"\n✅ Model training complete!")
    print(f"🚀 Ready for web application!")

if __name__ == "__main__":
    main()