import requests
from bs4 import BeautifulSoup
import re

def debug_tournament_structure(tournament_name, event_id):
    print(f"\n🔍 Debugging {tournament_name} (ID: {event_id})")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    url = f"https://www.vlr.gg/event/{event_id}"
    
    try:
        response = session.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print(f"✅ Page loaded successfully")
        
        # Check different potential structures
        structures_found = []
        
        # 1. Bracket cards (what worked for Toronto)
        bracket_cards = soup.find_all('div', class_=['wf-card', 'mod-bracket'])
        if bracket_cards:
            structures_found.append(f"Bracket cards: {len(bracket_cards)}")
        
        # 2. Match items
        match_items = soup.find_all('div', class_='match-item')
        if match_items:
            structures_found.append(f"Match items: {len(match_items)}")
        
        # 3. Any div with "match" in class
        match_divs = soup.find_all('div', class_=lambda x: x and 'match' in str(x).lower())
        if match_divs:
            structures_found.append(f"Match-related divs: {len(match_divs)}")
        
        # 4. Look for score patterns in any element
        all_text = soup.get_text()
        score_patterns = len([m for m in re.finditer(r'\d+[-:]\d+', all_text)])
        if score_patterns:
            structures_found.append(f"Score patterns found: {score_patterns}")
        
        # 5. Check for team names
        team_names = ['Paper Rex', 'Fnatic', 'G2 Esports', 'Sentinels', 'T1', 'DRX', 'Gen.G']
        teams_found = []
        for team in team_names:
            if team in all_text:
                teams_found.append(team)
        
        if teams_found:
            structures_found.append(f"Teams found: {teams_found[:3]}...")
        
        print(f"📊 Structures found: {structures_found}")
        
        # Show a sample of the page content
        print(f"📄 Page title: {soup.title.string if soup.title else 'No title'}")
        
        # Look for any links to matches
        match_links = soup.find_all('a', href=lambda x: x and '/matches/' in str(x))
        if match_links:
            print(f"🔗 Found {len(match_links)} match links")
            for link in match_links[:3]:
                print(f"   - {link.get('href')}")
        
        return len(structures_found) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test the tournaments that returned 0
tournaments_to_debug = [
    ("VCT 2025 Masters Bangkok", "2264"),
    ("EWC 2025", "2290"), 
    ("VCT 2024 Champions Seoul", "2177"),
    ("VCT 2024 Masters Shanghai", "2145")
]

if __name__ == "__main__":
    for tournament_name, event_id in tournaments_to_debug:
        debug_tournament_structure(tournament_name, event_id)
        print("-" * 50)
