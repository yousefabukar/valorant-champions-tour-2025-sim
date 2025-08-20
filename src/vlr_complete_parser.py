import requests
from bs4 import BeautifulSoup
import json
import re
import time

class MultiTournamentParser:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # All the tournaments you mentioned
        self.tournaments = {
            "VCT 2025 Masters Toronto": "2282",
            "VCT 2025 Masters Bangkok": "2264", 
            "EWC 2025": "2290",
            "VCT 2024 Champions Seoul": "2177",
            "VCT 2024 Masters Shanghai": "2145",
            "VCT 2024 Masters Madrid": "2097",
            "VCT 2025 Stage 1 Americas": "2347",
            "VCT 2025 Stage 1 EMEA": "2348", 
            "VCT 2025 Stage 1 Pacific": "2379"
        }
    
    def parse_tournament(self, tournament_name, event_id):
        print(f"\n🔍 Parsing {tournament_name}...")
        
        url = f"https://www.vlr.gg/event/{event_id}"
        
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for bracket cards
            match_cards = soup.find_all('div', class_=['wf-card', 'mod-bracket'])
            
            matches = []
            
            for card in match_cards:
                text_lines = [line.strip() for line in card.get_text().split('\n') if line.strip()]
                
                # Focus on cards with substantial content
                if len(text_lines) < 15:
                    continue
                
                # Extract teams and scores
                teams = []
                scores = []
                
                for line in text_lines:
                    # Team names (expanded list)
                    if any(team in line for team in [
                        'Xi Lai Gaming', 'Sentinels', 'G2 Esports', 'Paper Rex', 'Fnatic', 'Gen.G', 
                        'Wolves', 'Team Heretics', 'Team Liquid', 'MIBR', 'Rex Regum Qeon', 
                        'Bilibili Gaming', 'FNATIC', 'EDward Gaming', 'T1', 'DRX', 'LOUD', 
                        'Cloud9', '100 Thieves', 'NRG', 'KRÜ Esports', 'Leviatán', 'FURIA',
                        'Team Vitality', 'NAVI', 'BBL Esports', 'FUT Esports', 'GIANTX',
                        'Talon Esports', 'BOOM Esports', 'Global Esports', 'ZETA DIVISION',
                        'Trace Esports', 'Dragon Ranger Gaming', 'FunPlus Phoenix', 'TYLOO'
                    ]):
                        teams.append(line)
                    
                    # Scores
                    elif re.match(r'^\d+$', line) and int(line) <= 5:
                        scores.append(int(line))
                
                # Pair up teams and scores
                team_idx = 0
                score_idx = 0
                
                while team_idx < len(teams) - 1 and score_idx < len(scores) - 1:
                    team1 = teams[team_idx]
                    team2 = teams[team_idx + 1]
                    score1 = scores[score_idx]
                    score2 = scores[score_idx + 1]
                    
                    match = {
                        'team1': team1,
                        'team2': team2,
                        'score': f"{score1}-{score2}",
                        'completed': True,
                        'tournament': tournament_name
                    }
                    
                    matches.append(match)
                    
                    team_idx += 2
                    score_idx += 2
                    
                    # Safety limit per tournament
                    if len(matches) >= 30:
                        break
            
            print(f"  ✅ Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            print(f"  ❌ Error parsing {tournament_name}: {e}")
            return []
    
    def parse_all_tournaments(self):
        print("🚀 Starting multi-tournament data collection...")
        
        all_matches = []
        
        for tournament_name, event_id in self.tournaments.items():
            matches = self.parse_tournament(tournament_name, event_id)
            all_matches.extend(matches)
            
            # Be respectful with rate limiting
            time.sleep(2)
        
        print(f"\n📊 TOTAL COLLECTED:")
        print(f"  🎯 {len(all_matches)} matches across {len(self.tournaments)} tournaments")
        
        # Show breakdown by tournament
        tournament_counts = {}
        for match in all_matches:
            tournament = match['tournament']
            tournament_counts[tournament] = tournament_counts.get(tournament, 0) + 1
        
        for tournament, count in tournament_counts.items():
            print(f"    📈 {tournament}: {count} matches")
        
        return all_matches
    
    def save_all_data(self, matches):
        data = {
            "recent_matches": matches,
            "detailed_matches": [],
            "qualified_teams": {},
            "collection_info": {
                "scraped_at": "2025-08-21T12:00:00",
                "total_matches": len(matches),
                "completed_matches": len(matches),
                "tournaments_scraped": len(self.tournaments),
                "source": "VLR.gg Multi-Tournament Scraping",
                "method": "automated_multi_tournament_parsing"
            }
        }
        
        with open('data/raw/vct_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(matches)} matches to data/raw/vct_data.json")

def main():
    parser = MultiTournamentParser()
    
    # Parse all tournaments
    all_matches = parser.parse_all_tournaments()
    
    if all_matches:
        parser.save_all_data(all_matches)
        print(f"\n🎉 SUCCESS! Ready to train ML model with {len(all_matches)} matches from multiple tournaments!")
        print("Now run: python3 -m src.model")
    else:
        print("⚠️  No matches found across all tournaments")

if __name__ == "__main__":
    main()
