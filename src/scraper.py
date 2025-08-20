# src/scraper.py

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Optional

class VCTScraper:
    """Simple VCT data scraper focused on main VCT tournaments"""
    
    def __init__(self):
        self.base_url = "https://www.vlr.gg"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        print("🔍 VCTScraper initialized!")
        
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Get page with rate limiting and error handling"""
        time.sleep(1)  # Be respectful to VLR
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return None
    
    def get_vct_matches(self, limit: int = 20) -> List[Dict]:
        """Get recent VCT matches (filtering out Challengers)"""
        print(f"🔍 Fetching VCT matches...")
        url = f"{self.base_url}/matches/results"
        
        soup = self._get_page(url)
        if not soup:
            return []
        
        # Find all match cards
        match_cards = soup.find_all('a', class_='wf-module-item')
        print(f"📋 Found {len(match_cards)} total matches on page")
        
        vct_matches = []
        for i, card in enumerate(match_cards):
            try:
                # Get tournament info
                tournament_elem = card.find('div', class_='match-item-event')
                tournament = tournament_elem.text.strip() if tournament_elem else ""
                
                # Filter for main VCT events only
                if not self._is_main_vct_event(tournament):
                    continue
                
                # Get match URL and extract proper match ID
                match_href = card.get('href', '')
                if not match_href:
                    continue
                
                # Get teams
                team_elements = card.find_all('div', class_='text-of')
                if len(team_elements) >= 2:
                    team1 = team_elements[0].text.strip()
                    team2 = team_elements[1].text.strip()
                    
                    # Get score
                    score_elem = card.find('div', class_='match-item-score')
                    score = score_elem.text.strip() if score_elem else "TBD"
                    
                    # Get date
                    date_elem = card.find('div', class_='moment-tz-convert')
                    match_date = None
                    if date_elem and date_elem.get('data-utc-ts'):
                        try:
                            timestamp = int(date_elem['data-utc-ts']) / 1000
                            match_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                        except:
                            match_date = "Unknown"
                    
                    match_data = {
                        'match_url': self.base_url + match_href,
                        'match_id': self._extract_match_id(match_href),
                        'team1': team1,
                        'team2': team2,
                        'score': score,
                        'tournament': tournament,
                        'date': match_date,
                        'completed': score != "TBD" and "-" in score
                    }
                    
                    vct_matches.append(match_data)
                    status = "✅" if match_data['completed'] else "⏳"
                    print(f"  {status} {team1} vs {team2} ({score}) - {tournament}")
                    
                    if len(vct_matches) >= limit:
                        break
                    
            except Exception as e:
                print(f"⚠️  Error parsing match {i+1}: {e}")
                continue
        
        print(f"📊 Found {len(vct_matches)} VCT matches")
        return vct_matches
    
    def _extract_match_id(self, href: str) -> str:
        """Extract clean match ID from href"""
        # Remove leading slash and extract the numeric ID part
        clean_href = href.strip('/')
        parts = clean_href.split('/')
        if parts and parts[0].isdigit():
            return parts[0]
        return clean_href
    
    def _is_main_vct_event(self, tournament_name: str) -> bool:
        """Check if tournament is a main VCT event"""
        if not tournament_name:
            return False
            
        main_vct_keywords = [
            "VCT 2025",
            "Masters",
            "Champions", 
            "EWC 2025",
            "Stage 2",
            "Stage 1"
        ]
        
        # Exclude Challengers and Game Changers
        exclude_keywords = [
            "Challengers",
            "Game Changers", 
            "Academy",
            "Qualifier",
            "Last Chance",
            "LCQ"
        ]
        
        tournament_lower = tournament_name.lower()
        
        # Must contain VCT keywords
        has_vct = any(keyword.lower() in tournament_lower for keyword in main_vct_keywords)
        
        # Must not contain excluded keywords  
        is_excluded = any(keyword.lower() in tournament_lower for keyword in exclude_keywords)
        
        return has_vct and not is_excluded
    
    def get_match_details(self, match_id: str) -> Dict:
        """Get detailed match stats including player performance"""
        url = f"{self.base_url}/{match_id}"
        print(f"🔍 Getting details for match {match_id}...")
        
        soup = self._get_page(url)
        if not soup:
            return {}
        
        match_data = {
            'match_id': match_id,
            'teams': {},
            'maps': [],
            'overview': {}
        }
        
        try:
            # Get team names from header
            team_names = []
            team_headers = soup.find_all('div', class_='match-header-vs-team')
            for header in team_headers:
                name_elem = header.find('div', class_='wf-title-med')
                if name_elem:
                    team_names.append(name_elem.text.strip())
            
            if len(team_names) >= 2:
                match_data['teams'] = {
                    'team1': team_names[0],
                    'team2': team_names[1]
                }
                print(f"  Teams: {team_names[0]} vs {team_names[1]}")
            
            # Get overall match score
            score_elem = soup.find('div', class_='match-header-vs-score')
            if score_elem:
                match_data['overview']['final_score'] = score_elem.text.strip()
            
            # Get map results
            map_sections = soup.find_all('div', class_='vm-stats-game')
            print(f"  Found {len(map_sections)} maps")
            
            for i, map_section in enumerate(map_sections):
                map_data = self._parse_map_data(map_section, team_names)
                if map_data:
                    match_data['maps'].append(map_data)
                    print(f"    Map {i+1}: {map_data['map_name']} ({map_data['score']})")
        
        except Exception as e:
            print(f"❌ Error parsing match details: {e}")
        
        return match_data
    
    def _parse_map_data(self, map_section, team_names: List[str]) -> Dict:
        """Parse individual map statistics"""
        try:
            # Get map name and score
            map_header = map_section.find('div', class_='map')
            map_name = map_header.text.strip() if map_header else "Unknown"
            
            score_elem = map_section.find('div', class_='score')
            score = score_elem.text.strip() if score_elem else "0-0"
            
            map_data = {
                'map_name': map_name,
                'score': score,
                'players': {}
            }
            
            # Get player stats tables
            stat_tables = map_section.find_all('table', class_='wf-table-inset')
            
            for table_idx, table in enumerate(stat_tables):
                if table_idx >= len(team_names):
                    break
                
                team_name = team_names[table_idx]
                map_data['players'][team_name] = []
                
                # Parse each player row
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 7:  # Basic required columns
                        # Extract agent from image or text
                        agent_cell = cells[1]
                        agent_img = agent_cell.find('img')
                        agent = agent_img.get('alt', '') if agent_img else agent_cell.text.strip()
                        
                        player_stats = {
                            'name': cells[0].text.strip(),
                            'agent': agent,
                            'rating': self._safe_float(cells[2].text.strip()),
                            'acs': self._safe_float(cells[3].text.strip()),
                            'kills': self._safe_int(cells[4].text.strip()),
                            'deaths': self._safe_int(cells[5].text.strip()),
                            'assists': self._safe_int(cells[6].text.strip()),
                        }
                        
                        # Calculate K/D ratio
                        if player_stats['deaths'] > 0:
                            player_stats['kd'] = round(player_stats['kills'] / player_stats['deaths'], 2)
                        else:
                            player_stats['kd'] = player_stats['kills']
                        
                        # Add additional stats if available
                        if len(cells) > 7:
                            kast_text = cells[7].text.strip().replace('%', '')
                            player_stats['kast'] = self._safe_float(kast_text)
                        if len(cells) > 8:
                            player_stats['adr'] = self._safe_float(cells[8].text.strip())
                        if len(cells) > 9:
                            player_stats['first_kills'] = self._safe_int(cells[9].text.strip())
                        if len(cells) > 10:
                            player_stats['first_deaths'] = self._safe_int(cells[10].text.strip())
                        
                        map_data['players'][team_name].append(player_stats)
            
            return map_data
            
        except Exception as e:
            print(f"⚠️  Error parsing map data: {e}")
            return {}
    
    def _safe_float(self, value: str) -> float:
        """Safely convert string to float"""
        try:
            return float(value.replace(',', '').replace('%', ''))
        except:
            return 0.0
    
    def _safe_int(self, value: str) -> int:
        """Safely convert string to int"""
        try:
            return int(value.replace(',', ''))
        except:
            return 0
    
    def get_current_tournaments(self) -> List[str]:
        """Get list of current VCT tournaments"""
        print("🏆 Getting current VCT tournaments...")
        url = f"{self.base_url}/events"
        
        soup = self._get_page(url)
        if not soup:
            return []
        
        events = []
        # Look for event cards
        event_cards = soup.find_all('div', class_='events-container-col')
        
        for card in event_cards:
            try:
                title_elem = card.find('div', class_='event-item-title')
                if title_elem:
                    title = title_elem.text.strip()
                    if self._is_main_vct_event(title):
                        events.append(title)
                        print(f"  🎯 {title}")
            except:
                continue
        
        return events
    
    def get_team_recent_form(self, team_name: str, matches_limit: int = 10) -> List[Dict]:
        """Get recent matches for a specific team"""
        print(f"🔍 Getting recent form for {team_name}...")
        
        # Get recent matches and filter for the team
        recent_matches = self.get_vct_matches(limit=50)
        
        team_matches = []
        for match in recent_matches:
            team1_match = team_name.lower() in match['team1'].lower()
            team2_match = team_name.lower() in match['team2'].lower()
            
            if team1_match or team2_match:
                # Add result info
                if match['completed'] and '-' in match['score']:
                    scores = match['score'].split('-')
                    if len(scores) == 2:
                        score1, score2 = int(scores[0]), int(scores[1])
                        if team1_match:
                            match['result'] = 'win' if score1 > score2 else 'loss'
                        else:
                            match['result'] = 'win' if score2 > score1 else 'loss'
                else:
                    match['result'] = 'pending'
                
                team_matches.append(match)
                print(f"  {match['result'].upper()}: vs {match['team2'] if team1_match else match['team1']} ({match['score']})")
                
                if len(team_matches) >= matches_limit:
                    break
        
        return team_matches
    
    def get_qualified_teams(self) -> Dict:
        """Get teams that have qualified for Champions 2025"""
        # This would ideally scrape from VCT standings pages
        # For now, return known qualified teams based on current info
        qualified = {
            'americas': [
                {'name': 'G2 Esports', 'status': 'qualified', 'points': 11},
                {'name': 'Sentinels', 'status': 'qualified', 'points': 10}
            ],
            'emea': [
                {'name': 'Fnatic', 'status': 'qualified', 'points': 8},
                {'name': 'Team Heretics', 'status': 'likely', 'points': 5}  # EWC winners
            ],
            'pacific': [
                {'name': 'Paper Rex', 'status': 'qualified', 'points': 9},
                {'name': 'T1', 'status': 'qualified', 'points': 8},
                {'name': 'Rex Regum Qeon', 'status': 'qualified', 'points': 6}
            ],
            'china': [
                {'name': 'EDward Gaming', 'status': 'likely', 'points': 5},
                {'name': 'Trace Esports', 'status': 'contender', 'points': 4}
            ]
        }
        return qualified
    
    def save_data(self, data: Dict, filename: str):
        """Save scraped data to JSON file"""
        try:
            with open(f"data/raw/{filename}", 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"💾 Data saved to data/raw/{filename}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")


def main():
    """Test the scraper with comprehensive data collection"""
    print("🚀 Starting VCT Data Collection...")
    print("=" * 50)
    
    scraper = VCTScraper()
    
    # Get current tournaments
    print("\n🏆 Current VCT Tournaments:")
    tournaments = scraper.get_current_tournaments()
    if not tournaments:
        print("  ⚠️  No current VCT tournaments found")
    
    # Get recent VCT matches
    print("\n📊 Recent VCT Matches:")
    matches = scraper.get_vct_matches(limit=15)
    
    # Get detailed stats for completed matches
    detailed_matches = []
    if matches:
        completed_matches = [m for m in matches if m['completed']][:3]
        
        if completed_matches:
            print(f"\n🔍 Getting detailed stats for {len(completed_matches)} completed matches...")
            
            for match in completed_matches:
                details = scraper.get_match_details(match['match_id'])
                if details and details.get('maps'):
                    detailed_matches.append(details)
                    time.sleep(2)  # Be extra respectful
        else:
            print("\n⚠️  No completed VCT matches found for detailed analysis")
    
    # Get qualified teams info
    print("\n🎯 Champions 2025 Qualification Status:")
    qualified_teams = scraper.get_qualified_teams()
    for region, teams in qualified_teams.items():
        print(f"  {region.upper()}:")
        for team in teams:
            status_emoji = "✅" if team['status'] == 'qualified' else "🔥" if team['status'] == 'likely' else "⚡"
            print(f"    {status_emoji} {team['name']} ({team['points']} pts) - {team['status']}")
    
    # Save all collected data
    if matches or tournaments or detailed_matches:
        all_data = {
            'collection_info': {
                'scraped_at': datetime.now().isoformat(),
                'total_matches': len(matches),
                'detailed_matches': len(detailed_matches),
                'tournaments_found': len(tournaments)
            },
            'tournaments': tournaments,
            'recent_matches': matches,
            'detailed_matches': detailed_matches,
            'qualified_teams': qualified_teams
        }
        
        scraper.save_data(all_data, 'vct_data.json')
        
        # Print summary
        print(f"\n📈 Collection Summary:")
        print(f"  • {len(tournaments)} current tournaments")
        print(f"  • {len(matches)} recent VCT matches")
        print(f"  • {len(detailed_matches)} matches with detailed stats")
        print(f"  • {sum(len(teams) for teams in qualified_teams.values())} teams tracked for Champions")
        
        # Show sample detailed data
        if detailed_matches:
            sample = detailed_matches[0]
            print(f"\n🎮 Sample Match Details:")
            print(f"  Teams: {sample['teams']}")
            if sample['maps']:
                for i, map_data in enumerate(sample['maps']):
                    print(f"  Map {i+1}: {map_data['map_name']} ({map_data['score']})")
                    # Show top performer
                    for team, players in map_data['players'].items():
                        if players:
                            top_player = max(players, key=lambda p: p.get('rating', 0))
                            print(f"    {team} MVP: {top_player['name']} ({top_player['agent']}) - {top_player['rating']} rating")
    
    print("\n✅ VCT data collection complete!")
    print("🔥 Ready for ML model training!")

if __name__ == "__main__":
    main()