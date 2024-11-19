import requests
from bs4 import BeautifulSoup

def get_starred_repos(username):
    urls = []
    page = 1
    while True:
        url = f"https://github.com/{username}?tab=stars&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        repos = soup.find_all('a', {'class': 'Link'})
        
        found_repos = False
        for repo in repos:
            href = repo.get('href', '')
            if href.count('/') == 2 and href.startswith('/'):
                found_repos = True
                urls.append(f"https://github.com{href}")
        
        if not found_repos:
            break
            
        page += 1
    
    return urls

# Get all starred repos
urls = get_starred_repos('sosacrazy126')

# Write to file
with open('starred_repos.txt', 'w') as f:
    for url in urls:
        f.write(url + '\n')

print(f"Found {len(urls)} starred repositories. Saved to starred_repos.txt")
