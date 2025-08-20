# src/config.py

from datetime import datetime

# Tournament weights (simple recency-based)
TOURNAMENT_WEIGHTS = {
    # 2025 - Current season
    "VCT 2025 Stage 2": 1.0,
    "EWC 2025": 0.9,
    "VCT 2025 Masters Toronto": 0.8,
    "VCT 2025 Stage 1": 0.7,
    "VCT 2025 Masters Bangkok": 0.75,
    "VCT 2025 Kickoff": 0.6,
    
    # 2024 - Previous season
    "VCT 2024 Champions Seoul": 0.5,
    "VCT 2024 Masters Shanghai": 0.4,
    "VCT 2024 Stage 2": 0.35,
    "VCT 2024 Masters Madrid": 0.4,
    "VCT 2024 Stage 1": 0.3,
    
    # 2023 - Historical context
    "VCT 2023 Champions Los Angeles": 0.25,
    "VCT 2023 Masters Tokyo": 0.2,
    "VCT 2023 Stage 2": 0.15,
}

# Current map pool
MAP_POOL = [
    "Ascent", "Bind", "Haven", "Split", 
    "Lotus", "Sunset", "Breeze"
]

# Agent categories (simple)
AGENTS = {
    "duelist": ["Jett", "Raze", "Neon", "Reyna", "Phoenix", "Yoru"],
    "controller": ["Omen", "Viper", "Brimstone", "Astra", "Harbor", "Clove"],
    "initiator": ["Sova", "Breach", "Skye", "KAY/O", "Fade", "Gekko"],
    "sentinel": ["Cypher", "Killjoy", "Sage", "Chamber", "Deadlock"]
}

# Champions 2025 format
CHAMPIONS_FORMAT = {
    "teams_total": 16,
    "teams_per_region": 4,
    "regions": ["americas", "emea", "pacific", "china"],
    "group_stage_format": "4 groups of 4",
    "playoff_format": "double_elimination"
}

# UPDATED: Current Championship Points (from VCT official site - August 2025)
CURRENT_CHAMPIONSHIP_POINTS = {
    "americas": [
        {"name": "G2 Esports", "points": 19, "status": "qualified"},
        {"name": "Sentinels", "points": 11, "status": "qualified"},
        {"name": "MIBR", "points": 7, "status": "contender"},
        {"name": "KRÜ Esports", "points": 5, "status": "contender"},
        {"name": "Evil Geniuses", "points": 4, "status": "contender"},
        {"name": "100 Thieves", "points": 3, "status": "contender"},
        {"name": "Cloud9", "points": 3, "status": "contender"},
        {"name": "NRG", "points": 2, "status": "longshot"},
        {"name": "2Game Esports", "points": 1, "status": "longshot"},
        {"name": "Leviatán Esports", "points": 1, "status": "longshot"},
        {"name": "LOUD", "points": 0, "status": "eliminated"},
        {"name": "FURIA", "points": 0, "status": "eliminated"}
    ],
    "emea": [
        {"name": "Fnatic", "points": 14, "status": "qualified"},
        {"name": "Team Heretics", "points": 9, "status": "likely"},  # EWC winners
        {"name": "Team Liquid", "points": 8, "status": "likely"},
        {"name": "Team Vitality", "points": 6, "status": "contender"},
        {"name": "BBL Esports", "points": 5, "status": "contender"},
        {"name": "FUT Esports", "points": 4, "status": "contender"},
        {"name": "NAVI", "points": 4, "status": "contender"},
        {"name": "GIANTX", "points": 2, "status": "longshot"},
        {"name": "Karmine Corp", "points": 2, "status": "longshot"},
        {"name": "Movistar KOI", "points": 1, "status": "longshot"},
        {"name": "Gentle Mates", "points": 1, "status": "longshot"},
        {"name": "APEKS", "points": 0, "status": "eliminated"}
    ],
    "pacific": [
        {"name": "Paper Rex", "points": 12, "status": "qualified"},
        {"name": "T1", "points": 10, "status": "qualified"},
        {"name": "Gen.G", "points": 10, "status": "qualified"},
        {"name": "DRX", "points": 8, "status": "likely"},
        {"name": "Rex Regum Qeon", "points": 8, "status": "likely"},
        {"name": "Talon Esports", "points": 5, "status": "contender"},
        {"name": "BOOM Esports", "points": 5, "status": "contender"},
        {"name": "NONGSHIM RedForce", "points": 4, "status": "contender"},
        {"name": "Global Esports", "points": 2, "status": "longshot"},
        {"name": "ZETA DIVISION", "points": 2, "status": "longshot"},
        {"name": "DetonatioN FocusMe", "points": 1, "status": "longshot"},
        {"name": "Team Secret", "points": 1, "status": "longshot"}
    ],
    "china": [
        {"name": "EDward Gaming", "points": 11, "status": "likely"},
        {"name": "Xi Lai Gaming", "points": 10, "status": "likely"},
        {"name": "Bilibili Gaming", "points": 9, "status": "likely"},
        {"name": "Wolves Esports", "points": 9, "status": "likely"},
        {"name": "Trace Esports", "points": 6, "status": "contender"},
        {"name": "Dragon Ranger Gaming", "points": 4, "status": "contender"},
        {"name": "FunPlus Phoenix", "points": 4, "status": "contender"},
        {"name": "Nova Esports", "points": 3, "status": "contender"},
        {"name": "Titan Esports Club", "points": 3, "status": "contender"},
        {"name": "TYLOO Gaming", "points": 2, "status": "longshot"},
        {"name": "JD Mall JDG Esports", "points": 1, "status": "longshot"},
        {"name": "All Gamers", "points": 0, "status": "eliminated"}
    ]
}

# Legacy format for backwards compatibility
QUALIFIED_TEAMS = {
    "americas": ["G2 Esports", "Sentinels"],
    "emea": ["Fnatic"],
    "pacific": ["Paper Rex", "T1", "Gen.G"],
    "china": []  # Still being determined
}

# File paths
DATA_PATHS = {
    "raw": "data/raw/",
    "processed": "data/processed/",
    "models": "models/",
    "predictions": "data/predictions/"
}

# Scraping settings
SCRAPING = {
    "rate_limit": 1.0,  # seconds between requests
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "timeout": 10
}

# Model settings
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

def get_tournament_weight(tournament_name: str) -> float:
    """Get weight for a tournament"""
    return TOURNAMENT_WEIGHTS.get(tournament_name, 0.1)

def get_agent_role(agent_name: str) -> str:
    """Get role for an agent"""
    for role, agents in AGENTS.items():
        if agent_name in agents:
            return role
    return "unknown"

def is_qualified_team(team_name: str) -> bool:
    """Check if team is already qualified for Champions"""
    for region_teams in CURRENT_CHAMPIONSHIP_POINTS.values():
        for team in region_teams:
            if team['name'] == team_name and team['status'] == 'qualified':
                return True
    return False

def get_team_championship_points(team_name: str) -> int:
    """Get current championship points for a team"""
    for region_teams in CURRENT_CHAMPIONSHIP_POINTS.values():
        for team in region_teams:
            if team['name'] == team_name:
                return team['points']
    return 0

def get_team_qualification_status(team_name: str) -> str:
    """Get qualification status for a team"""
    for region_teams in CURRENT_CHAMPIONSHIP_POINTS.values():
        for team in region_teams:
            if team['name'] == team_name:
                return team['status']
    return 'unknown'

def get_region_standings(region: str) -> list:
    """Get championship points standings for a region"""
    return CURRENT_CHAMPIONSHIP_POINTS.get(region, [])

def calculate_champions_probability(team_name: str) -> float:
    """Calculate realistic Champions qualification probability"""
    
    points = get_team_championship_points(team_name)
    status = get_team_qualification_status(team_name)
    
    # Status-based probabilities
    status_probs = {
        'qualified': 0.95,
        'likely': 0.75,
        'contender': 0.35,
        'longshot': 0.15,
        'eliminated': 0.02,
        'unknown': 0.10
    }
    
    base_prob = status_probs.get(status, 0.1)
    
    # Adjust based on points (higher points = higher probability within status)
    if points > 15:
        return min(0.98, base_prob * 1.2)
    elif points > 10:
        return min(0.9, base_prob * 1.1)
    elif points > 5:
        return base_prob
    else:
        return max(0.02, base_prob * 0.8)

# Current date for calculations
CURRENT_DATE = datetime.now()

# Print current standings summary
def print_standings_summary():
    print("⚙️ VCT Predictor Config Loaded")
    print(f"📅 Current date: {CURRENT_DATE.strftime('%Y-%m-%d')}")
    print("\n🏆 Championship Points Leaders:")
    
    for region, teams in CURRENT_CHAMPIONSHIP_POINTS.items():
        print(f"\n{region.upper()}:")
        for i, team in enumerate(teams[:4]):  # Top 4 per region
            status_emoji = {
                'qualified': '✅',
                'likely': '🔥', 
                'contender': '⚡',
                'longshot': '🎯',
                'eliminated': '❌'
            }.get(team['status'], '❓')
            
            print(f"  {i+1}. {status_emoji} {team['name']} - {team['points']} pts ({team['status']})")
    
    total_qualified = sum(1 for region in CURRENT_CHAMPIONSHIP_POINTS.values() 
                         for team in region if team['status'] == 'qualified')
    print(f"\n📊 {total_qualified}/16 teams qualified for Champions 2025")

if __name__ == "__main__":
    print_standings_summary()