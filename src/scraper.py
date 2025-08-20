# src/scraper.py

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Optional

class VCTScraper:
    """Enhanced VCT data scraper for historical and current tournament data"""
    
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
    
    def get_historical_tournaments(self) -> List[Dict]:
        """Get historical tournament data with completed matches"""
        
        historical_tournaments = [
            {"name": "VCT 2024 Champions Seoul", "event_id": "2177"},
            {"name": "VCT 2025 Masters Toronto", "event_id": "2283"}, 
            {"name": "VCT 2025 Masters Bangkok", "event_id": "2264"},
            {"name": "EWC 2025", "event_id": "2290"},
            {"name": "VCT 2024 Masters Shanghai", "event_id": "2145"},
            {"name": "VCT 2024 Masters Madrid", "event_id": "2097"},
            {"name": "VCT 2025 Stage 1 Americas", "event_id": "2347"},
            {"name": "VCT 2025 Stage 1 EMEA", "event_id": "2348"},
            {"name": "VCT 2025 Stage 1 Pacific", "event_id": "2379"},
            {"name": "VCT 2024 Stage 2 Americas", "event_id": "2156"},
            {"name": "VCT 2024 Stage 2 EMEA", "event_id": "2157"},
            {"name": "VCT 2024 Stage 2 Pacific", "event_id": "2158"},
        ]
        
        all_matches = []
        
        for tournament in historical_tournaments:
            print(f"🔍 Scraping {tournament['name']}...")
            
            # Try different URL patterns
            urls_to_try = [
                f"{self.base_url}/event/{tournament['event_id']}/matches",
                f"{self.base_url}/event/{tournament['event_id']}",
            ]
            
            matches = []
            for url in urls_to_try:
                soup = self._get_page(url)
                if soup:
                    matches = self._parse_tournament_matches(soup, tournament['name'])
                    if matches:
                        break
            
            if matches:
                all_matches.extend(matches)
                print(f"  ✅ Found {len(matches)} completed matches")
            else:
                print(f"  ⚠️  No matches found for {tournament['name']}")
                
        return all_matches
    
    def _parse_tournament_matches(self, soup: BeautifulSoup, tournament_name: str) -> List[Dict]:
        """Parse matches from a specific tournament page"""
        matches = []
        
        # Look for match cards with different possible classes
        match_selectors = [
            'a.wf-module-item',
            'a[href*="/"]',
            '.match-item',
            '.wf-card'
        ]
        
        match_cards = []
        for selector in match_selectors:
            match_cards = soup.select(selector)
            if match_cards:
                break
        
        for card in match_cards:
            try:
                # Skip if not a match URL
                href = card.get('href', '')
                if not href or not re.search(r'/\d+/', href):
                    continue
                
                # Check if match has a score (completed)
                score_elem = card.find('div', class_='match-item-score')
                if not score_elem:
                    # Try alternative score selectors
                    score_elem = card.find('div', class_='score') or card.find('.match-score')
                
                if not score_elem:
                    continue
                    
                score = score_elem.text.strip()
                
                # Skip if not completed
                if score in ['TBD', '-', '', 'vs']:
                    continue
                
                # Must have actual numbers in score
                if not re.search(r'\d', score):
                    continue
                
                # Get teams
                team_elements = card.find_all('div', class_='text-of')
                if not team_elements:
                    # Try alternative team selectors
                    team_elements = card.find_all('.team-name') or card.find_all('.match-team')
                
                if len(team_elements) >= 2:
                    team1 = team_elements[0].text.strip()
                    team2 = team_elements[1].text.strip()
                    
                    # Get date if available
                    date_elem = card.find('div', class_='moment-tz-convert')
                    match_date = None
                    if date_elem and date_elem.get('data-utc-ts'):
                        try:
                            timestamp = int(date_elem['data-utc-ts']) / 1000
                            match_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                        except:
                            match_date = "Unknown"
                    
                    match_data = {
                        'match_id': href.strip('/').split('/')[-1],
                        'match_url': self.base_url + href if not href.startswith('http') else href,
                        'team1': team1,
                        'team2': team2,
                        'score': score,
                        'tournament': tournament_name,
                        'date': match_date,
                        'completed': True
                    }
                    matches.append(match_data)
                    
            except Exception as e:
                continue
                
        return matches
    
    def get_vct_matches(self, limit: int = 20) -> List[Dict]:
        """Get recent VCT matches (filtering out Challengers)"""
        print(f"🔍 Fetching current VCT matches...")
        url = f"{self.base_url}/matches/results"
        
        soup = self._get_page(url)
        if not soup:
            return []
        
        # Find all match cards
        match_cards = soup.find_all('a', class_='wf-module-item')
        print(f"📋 Found {len(match_cards)} total recent matches")
        
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
                continue
        
        print(f"📊 Found {len(vct_matches)} current VCT matches")
        return vct_matches
    
    def _extract_match_id(self, href: str) -> str:
        """Extract clean match ID from href"""
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
            "VCT 2024", 
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
    
    def get_qualified_teams(self) -> Dict:
        """Get teams that have qualified for Champions 2025"""
        # This would ideally scrape from VCT standings pages
        # For now, return current championship points from config
        from config import CURRENT_CHAMPIONSHIP_POINTS
        
        return CURRENT_CHAMPIONSHIP_POINTS
    
    def save_data(self, data: Dict, filename: str):
        """Save scraped data to JSON file"""
        try:
            with open(f"data/raw/{filename}", 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"💾 Data saved to data/raw/{filename}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")


def main():
    """Enhanced data collection with historical tournaments"""
    print("🚀 Starting Enhanced VCT Data Collection...")
    print("=" * 60)
    
    scraper = VCTScraper()
    
    # Get historical tournament data
    print("\n📚 Collecting Historical Tournament Data:")
    print("This may take a few minutes due to rate limiting...")
    historical_matches = scraper.get_historical_tournaments()
    
    # Get current matches  
    print(f"\n📊 Collecting Current VCT Matches:")
    current_matches = scraper.get_vct_matches(limit=20)
    
    # Combine all data
    all_matches = historical_matches + current_matches
    completed_matches = [m for m in all_matches if m.get('completed', False)]
    
    print(f"\n📈 Data Collection Summary:")
    print(f"  • Historical matches: {len(historical_matches)}")
    print(f"  • Current matches: {len(current_matches)}")
    print(f"  • Total matches: {len(all_matches)}")
    print(f"  • Completed matches: {len(completed_matches)}")
    
    # Show sample completed matches
    if completed_matches:
        print(f"\n🎮 Sample Completed Matches:")
        for match in completed_matches[:5]:
            print(f"  • {match['team1']} vs {match['team2']} ({match['score']}) - {match['tournament']}")
    
    # Get detailed stats for a few recent completed matches
    detailed_matches = []
    if completed_matches:
        print(f"\n🔍 Getting detailed stats for recent matches...")
        
        # Get details for up to 3 recent completed matches
        recent_completed = [m for m in completed_matches if 'Stage 2' in m.get('tournament', '')][:3]
        
        for match in recent_completed:
            print(f"Getting details for: {match['team1']} vs {match['team2']}")
            details = scraper.get_match_details(match['match_id'])
            if details and details.get('maps'):
                detailed_matches.append(details)
                time.sleep(2)  # Be respectful
    
    # Get qualified teams info
    print(f"\n🎯 Champions 2025 Qualification Status:")
    qualified_teams = scraper.get_qualified_teams()
    for region, teams in qualified_teams.items():
        qualified_count = len([t for t in teams if t.get('status') == 'qualified'])
        print(f"  {region.upper()}: {qualified_count} qualified, {len(teams)} total teams")
    
    # Save all collected data
    all_data = {
        'collection_info': {
            'scraped_at': datetime.now().isoformat(),
            'total_matches': len(all_matches),
            'completed_matches': len(completed_matches),
            'detailed_matches': len(detailed_matches),
            'historical_matches': len(historical_matches),
            'current_matches': len(current_matches)
        },
        'recent_matches': all_matches,  # All matches (historical + current)
        'detailed_matches': detailed_matches,
        'qualified_teams': qualified_teams
    }
    
    scraper.save_data(all_data, 'vct_data.json')
    
    print(f"\n✅ Enhanced data collection complete!")
    print(f"🔥 Ready for ML model training with {len(completed_matches)} completed matches!")

if __name__ == "__main__":
    main()